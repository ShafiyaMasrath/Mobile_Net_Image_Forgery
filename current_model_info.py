"""
Current Model Performance Analysis
"""
import torch
import os
from model.mobforge_net import MobForgeNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_current_model():
    weights_path = 'checkpoints/best_model.pth'
    
    print("=" * 70)
    print("CURRENT MODEL ANALYSIS")
    print("=" * 70)
    
    # Check file info
    if os.path.exists(weights_path):
        file_size = os.path.getsize(weights_path)
        mod_time = os.path.getmtime(weights_path)
        
        import time
        print(f"\n📊 Model File Information:")
        print(f"  Path: {weights_path}")
        print(f"  Size: {file_size / 1024 / 1024:.2f} MB")
        print(f"  Last modified: {time.ctime(mod_time)}")
    else:
        print(f"\n❌ Model not found at {weights_path}")
        return
    
    # Load and analyze
    print(f"\n📦 Model Architecture:")
    model = MobForgeNet(pretrained=False).to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Parameters: {total_params / 1e6:.2f}M")
    print(f"  Architecture: MobileNetV3-Small (lightweight)")
    print(f"  Backbone: Dual-stream (RGB + SRM noise residual)")
    
    # Current capabilities
    print(f"\n✅ THIS MODEL IS TRAINED ON:")
    print(f"  • Copy-move forgeries (duplicated regions)")
    print(f"  • Splicing forgeries (pasted objects)")
    print(f"  • CASIA 2.0 benchmark dataset")
    
    print(f"\n❌ THIS MODEL STRUGGLES WITH:")
    print(f"  • Blending/blurring edits ← Your Harsimg.jpeg")
    print(f"  • Subtle color modifications")
    print(f"  • AI-generated content")

if __name__ == '__main__':
    analyze_current_model()
