"""
Fixed Training Script — handles CASIA correctly.
Key fixes vs original:
  1. Dataset now verifies masks are non-empty before training
  2. Prints a clear warning if all masks are black (the bug you hit)
  3. Separate LR for encoder vs decoder
  4. Threshold tuning on validation set
  5. Saves best threshold alongside weights
"""

import os, time, argparse, torch, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np, json, sys

sys.path.insert(0, os.path.dirname(__file__))
from model.mobforge_net import MobForgeNet, BoundaryAwareLoss


class CASIADataset(Dataset):
    def __init__(self, data_dir, img_size=256, augment=True):
        self.img_dir  = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.augment  = augment

        all_files = [f for f in os.listdir(self.img_dir)
                     if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.bmp'))]

        self.files = []
        missing = 0
        for f in all_files:
            if os.path.exists(self._mask_path(f)):
                self.files.append(f)
            else:
                missing += 1
        if missing:
            print(f"  [Dataset] Skipped {missing} images — no matching mask.")

        # Sanity check
        n_check  = min(30, len(self.files))
        n_forged = sum(1 for f in self.files[:n_check]
                       if np.array(Image.open(self._mask_path(f)).convert('L')).max() > 127)
        pct = n_forged / n_check * 100
        print(f"  [Dataset] {n_forged}/{n_check} sampled masks have forged pixels ({pct:.0f}%)")
        if pct < 10:
            print("  WARNING: Almost all masks are black — run prepare_casia.py first!\n")

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def _mask_path(self, fname):
        stem = os.path.splitext(fname)[0]
        p = os.path.join(self.mask_dir, stem + '.png')
        if os.path.exists(p): return p
        return os.path.join(self.mask_dir, fname)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img   = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        mask  = Image.open(self._mask_path(fname)).convert('L')
        if self.augment:
            if np.random.rand() > 0.5:
                img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() > 0.5:
                img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        img_t  = self.img_tf(img)
        mask_t = (self.mask_tf(mask) > 0.5).float()
        return img_t, mask_t, fname


def metrics(pred, target, t=0.5):
    p = (pred > t).float()
    tp = (p*target).sum().item(); fp = (p*(1-target)).sum().item()
    fn = ((1-p)*target).sum().item(); tn = ((1-p)*(1-target)).sum().item()
    e = 1e-8
    pr = tp/(tp+fp+e); re = tp/(tp+fn+e)
    return dict(f1=2*pr*re/(pr+re+e), iou=tp/(tp+fp+fn+e),
                dice=2*tp/(2*tp+fp+fn+e), accuracy=(tp+tn)/(tp+tn+fp+fn+e))


def best_threshold(model, dl, device):
    model.eval()
    ps, ms = [], []
    with torch.no_grad():
        for imgs, masks, _ in dl:
            ps.append(model(imgs.to(device)).cpu()); ms.append(masks)
    ps, ms = torch.cat(ps), torch.cat(ms)
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = metrics(ps, ms, t)['f1']
        if f1 > best_f1: best_f1, best_t = f1, t
    return float(best_t), best_f1


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = MobForgeNet(pretrained=True).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")

    print("Loading datasets...")
    train_ds = CASIADataset(os.path.join(args.data_dir,'train'), args.img_size)
    val_ds   = CASIADataset(os.path.join(args.data_dir,'val'),   args.img_size, augment=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=2)

    criterion = BoundaryAwareLoss()
    enc_params = list(model.encoder_rgb.parameters())
    other      = [p for p in model.parameters() if not any(p is e for e in enc_params)]
    optimizer  = optim.AdamW([{'params':enc_params,'lr':args.lr*0.1},
                               {'params':other,'lr':args.lr}], weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = 0.0; thresh = 0.5; history = []

    print(f"{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | {'F1':>6} | {'IoU':>6} | {'Dice':>6} | {'Thr':>5}")
    print('─'*58)

    for ep in range(1, args.epochs+1):
        model.train(); tl = 0.0
        for imgs, masks, _ in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        scheduler.step()

        model.eval(); vl = 0.0; vm = []
        with torch.no_grad():
            for imgs, masks, _ in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                p = model(imgs); vl += criterion(p, masks).item()
                vm.append(metrics(p, masks, thresh))

        avg = {k: np.mean([m[k] for m in vm]) for k in vm[0]}
        avg.update(train_loss=tl/len(train_dl), val_loss=vl/len(val_dl))
        history.append(avg)
        print(f"{ep:>4} | {avg['train_loss']:>8.4f} | {avg['val_loss']:>8.4f} | "
              f"{avg['f1']:>6.4f} | {avg['iou']:>6.4f} | {avg['dice']:>6.4f} | {thresh:>5.2f}")

        if ep % 5 == 0:
            thresh, tf1 = best_threshold(model, val_dl, device)
            print(f"  → Threshold updated to {thresh:.2f} (F1={tf1:.4f})")

        if avg['f1'] > best_f1:
            best_f1 = avg['f1']
            torch.save({'state_dict': model.state_dict(), 'threshold': thresh, 'f1': best_f1,
                        'epoch': ep}, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  ★ Saved best model — F1={best_f1:.4f}")

    torch.save({'state_dict': model.state_dict(), 'threshold': thresh},
               os.path.join(args.save_dir, 'last_model.pth'))
    json.dump(history, open(os.path.join(args.save_dir,'history.json'),'w'), indent=2)
    print(f"\nDone. Best F1={best_f1:.4f}, best threshold={thresh:.2f}")
    print(f"Run: python inference.py --image test.jpg --weights {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='data')
    p.add_argument('--save_dir',   default='checkpoints')
    p.add_argument('--img_size',   type=int,   default=256)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--lr',         type=float, default=1e-3)
    train(p.parse_args())
