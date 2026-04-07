"""
Diagnose why CASIA-trained model doesn't detect your forged image
"""
import torch
import os
from model.mobforge_net import MobForgeNet
from PIL import Image
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def diagnose():
    print("=" * 70)
    print("DIAGNOSING MODEL PERFORMANCE")
    print("=" * 70)
    
    weights_path = 'checkpoints/best_model.pth'
    
    # Check checkpoint file info
    if os.path.exists(weights_path):
        file_size = os.path.getsize(weights_path)
        print(f"\n✓ Model file found: {weights_path}")
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Check file modification time
        import time
        mod_time = os.path.getmtime(weights_path)
        print(f"  Last modified: {time.ctime(mod_time)}")
    else:
        print(f"\n❌ Model file NOT found: {weights_path}")
        return False
    
    # Load and check model
    print("\n📊 Model Information:")
    model = MobForgeNet(pretrained=False).to(DEVICE)
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable params: {trainable / 1e6:.2f}M")
    
    # Load weights
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("  ✓ Weights loaded")
    
    # Check if data folder exists with training metrics
    print("\n📂 Checking for training data:")
    
    data_dir = 'data'
    if os.path.exists(data_dir):
        train_images = len([f for f in os.listdir('data/train/images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists('data/train/images') else 0
        train_masks = len([f for f in os.listdir('data/train/masks') if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) if os.path.exists('data/train/masks') else 0
        val_images = len([f for f in os.listdir('data/val/images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists('data/val/images') else 0
        val_masks = len([f for f in os.listdir('data/val/masks') if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) if os.path.exists('data/val/masks') else 0
        
        print(f"  Training set: {train_images} images, {train_masks} masks")
        print(f"  Validation set: {val_images} images, {val_masks} masks")
        
        if train_images > 0 and train_masks > 0:
            print(f"  ✓ Training data found ({train_images + val_images} total images)")
        else:
            print(f"  ⚠️ WARNING: No training data found in data/ folder!")
    else:
        print(f"  ⚠️ No 'data' folder found")
    
    # Try to estimate training quality from weights
    print("\n🔍 Analyzing model weights:")
    
    # Check weight magnitudes
    weight_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            w = param.data.cpu().numpy()
            weight_stats[name] = {
                'mean': w.mean(),
                'std': w.std(),
                'min': w.min(),
                'max': w.max()
            }
    
    # Print a few sample weights
    sample_names = list(weight_stats.keys())[:3]
    for name in sample_names:
        stats = weight_stats[name]
        print(f"  {name}:")
        print(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
        print(f"    Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    
    # Check if weights look like they were trained
    avg_weight_magnitude = np.mean([abs(w['mean']) for w in weight_stats.values()])
    print(f"\n  Average weight magnitude: {avg_weight_magnitude:.6f}")
    
    if avg_weight_magnitude < 0.001:
        print("  ⚠️ WARNING: Weights are very close to initialization!")
        print("     This suggests the model was NOT properly trained.")
    else:
        print("  ✓ Weights show signs of training")
    
    # Final diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 70)
    
    print("\nPossible issues:")
    print("1. Model was trained but underfits (can't detect your specific forgery type)")
    print("2. CASIA dataset doesn't cover your forgery technique")
    print("3. Model needs more training epochs")
    print("4. Learning rate or batch size was too high/low")
    print("5. Model is overfitting to CASIA patterns")
    
    print("\nRecommendations:")
    print("1. RETRAIN with better hyperparameters:")
    print("   python train.py --data_dir data --epochs 150 --batch_size 8 --lr 1e-4")
    print("\n2. Or try TRANSFER LEARNING from a pre-trained forgery model")
    print("\n3. Analyze your forgery type:")
    print("   - Is it COPY-MOVE forgery?")
    print("   - Is it SPLICING?")
    print("   - Something else?")
    
    return True

if __name__ == '__main__':
    diagnose()
