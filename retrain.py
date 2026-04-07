"""
Retrain MobForge-Net with optimized hyperparameters for blending/blurring detection
"""
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys

# Add model to path
sys.path.insert(0, os.path.dirname(__file__))
from model.mobforge_net import MobForgeNet, BoundaryAwareLoss


class ForgeryDataset(Dataset):
    def __init__(self, data_dir, img_size=256, augment=True):
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.img_size = img_size

        self.files = [
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert('RGB')

        # Find corresponding mask
        mask_path = os.path.join(self.mask_dir, fname)
        if not os.path.exists(mask_path):
            stem, _ = os.path.splitext(fname)
            candidates = [
                f'{stem}.png', f'{stem}.jpg', f'{stem}.jpeg',
                f'{stem}_mask.png', f'{stem}_mask.jpg', f'{stem}_mask.jpeg'
            ]
            mask_path = None
            for candidate in candidates:
                candidate_path = os.path.join(self.mask_dir, candidate)
                if os.path.exists(candidate_path):
                    mask_path = candidate_path
                    break

        if mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert('L')
                mask.load()
            except:
                mask = Image.fromarray(
                    np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
                )
        else:
            mask = Image.fromarray(
                np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
            )

        # Augmentation
        if self.augment and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if self.augment and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Rotation augmentation (good for blending/blurring)
        if self.augment and np.random.rand() > 0.7:
            angle = np.random.choice([90, 180, 270])
            img = img.rotate(angle, expand=False)
            mask = mask.rotate(angle, expand=False)

        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return img, mask, fname


def train(args):
    print("=" * 70)
    print("🔥 RETRAINING MODEL FOR BLENDING/BLURRING DETECTION")
    print("=" * 70)
    print(f"Hyperparameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: Adam with weight decay")
    print("=" * 70 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Model
    print("📦 Loading model...")
    model = MobForgeNet(pretrained=True).to(device)
    print(f"✓ Model loaded (MobileNetV3-Small backbone)")

    # Dataset
    print(f"\n📂 Loading dataset from {args.data_dir}...")
    train_ds = ForgeryDataset(os.path.join(args.data_dir, 'train'), augment=True)
    val_ds = ForgeryDataset(os.path.join(args.data_dir, 'val'), augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"✓ Training: {len(train_ds)} images")
    print(f"✓ Validation: {len(val_ds)} images")

    # Loss + optimizer (with weight decay for regularization)
    criterion = BoundaryAwareLoss(lambda1=1.0, lambda2=1.0, lambda3=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler (reduce LR if no improvement)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_f1 = 0
    best_metrics = {}
    patience_counter = 0
    patience_limit = 15  # Stop if no improvement for 15 epochs

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70 + "\n")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_pixels = 0

        with tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for imgs, masks, _ in pbar:
                imgs, masks = imgs.to(device), masks.to(device)

                preds = model(imgs)
                loss = criterion(preds, masks)

                # Calculate accuracy
                pred_binary = (preds > 0.5).float()
                correct = (pred_binary == masks).sum().item()
                pixels = masks.numel()
                total_correct += correct
                total_pixels += pixels

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.4f}'})

        avg_train_loss = total_loss / len(train_dl)
        avg_train_acc = (total_correct / total_pixels) * 100 if total_pixels > 0 else 0

        # Validation
        model.eval()
        total_val_loss = 0
        tp, fp, tn, fn = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, masks, _ in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)

                preds = model(imgs)
                loss = criterion(preds, masks)
                total_val_loss += loss.item()

                pred_binary = (preds > 0.5).float()
                tp += ((pred_binary == 1) & (masks == 1)).sum().item()
                fp += ((pred_binary == 1) & (masks == 0)).sum().item()
                tn += ((pred_binary == 0) & (masks == 0)).sum().item()
                fn += ((pred_binary == 0) & (masks == 1)).sum().item()

        avg_val_loss = total_val_loss / len(val_dl)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0

        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # Save best model based on F1
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'f1': f1, 'iou': iou, 'dice': f1, 'precision': precision,
                'recall': recall, 'accuracy': accuracy, 'train_loss': avg_train_loss,
                'val_loss': avg_val_loss, 'epoch': epoch+1
            }
            patience_counter = 0
            script_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_dir = os.path.join(script_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"  💾 Best model saved (F1: {f1:.4f})")
        else:
            patience_counter += 1

        # Update learning rate
        scheduler.step(f1)

        # Early stopping
        if patience_counter >= patience_limit:
            print(f"\n⏹️  Early stopping (no improvement for {patience_limit} epochs)")
            break

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Best model saved at epoch {best_metrics.get('epoch', 'N/A')}")
    print(f"  F1 Score:  {best_metrics.get('f1', 0):.4f}")
    print(f"  IoU:       {best_metrics.get('iou', 0):.4f}")
    print(f"  Precision: {best_metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {best_metrics.get('recall', 0):.4f}")
    print(f"  Accuracy:  {best_metrics.get('accuracy', 0):.4f}")
    print("=" * 70)
    print("✓ Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain MobForge-Net')
    parser.add_argument('--data_dir', default='data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    # Make data_dir absolute if relative
    if not os.path.isabs(args.data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_dir = os.path.join(script_dir, args.data_dir)
    
    # Validate paths
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        exit(1)

    train(args)
