import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model.mobforge_net import MobForgeNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights='checkpoints/best_model.pth'
print('weights exists:', os.path.exists(weights))
model=MobForgeNet(pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(weights,map_location=DEVICE))
model.eval()
transform=T.Compose([T.Resize((256,256)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
for name in ['1_authentic.jpg','2_copymove.jpg','3_splicing.jpg','4_complex.jpg']:
    path=os.path.join('test_images',name)
    img=Image.open(path).convert('RGB')
    t=transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out=model(t)[0,0].cpu().numpy()
    mean=np.mean(out); mx=np.max(out); mn=np.min(out)
    p50=(out>0.5).mean()*100; p10=(out>0.1).mean()*100; p01=(out>0.01).mean()*100
    print(name, f'mean={mean:.6f}', f'max={mx:.6f}', f'min={mn:.6f}', f'>0.5={p50:.2f}%', f'>0.1={p10:.2f}%', f'>0.01={p01:.2f}%')
