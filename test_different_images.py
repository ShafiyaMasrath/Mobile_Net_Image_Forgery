"""
Test script to verify the model produces different outputs for different images.
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model.mobforge_net import MobForgeNet
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_different_images():
    """Test model with different image patterns."""
    
    # Load model
    weights_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pth')
    model = MobForgeNet(pretrained=False).to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("=" * 70)
    print("TESTING MODEL WITH DIFFERENT IMAGE TYPES")
    print("=" * 70)
    
    # Create test images properly
    test_images = {}
    
    # 1. Solid black image
    test_images['solid_black'] = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 2. Solid white image
    test_images['solid_white'] = np.full((256, 256, 3), 255, dtype=np.uint8)
    
    # 3. Random noise
    test_images['random_noise'] = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # 4. Gradient (left to right)
    gradient = np.linspace(0, 255, 256).astype(np.uint8)
    test_images['gradient'] = np.tile(gradient[np.newaxis, :, np.newaxis], (256, 1, 3))
    
    # 5. Checkerboard pattern
    board = ((np.arange(256)//32 + np.arange(256)[:,np.newaxis]//32) % 2) * 255
    test_images['checkerboard'] = np.stack([board, board, board], axis=2).astype(np.uint8)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = {}
    outputs_raw = {}
    
    with torch.no_grad():
        for name, img_array in test_images.items():
            img = Image.fromarray(img_array)
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            output = model(tensor)
            
            forgery_pct = (output > 0.5).float().mean().item() * 100
            mean_prob = output.mean().item()
            max_prob = output.max().item()
            
            results[name] = {
                'forgery_pct': forgery_pct,
                'mean_prob': mean_prob,
                'max_prob': max_prob
            }
            outputs_raw[name] = output.squeeze().cpu().numpy()
            
            print(f"\n{name:20} | Forgery%: {forgery_pct:6.2f}% | Mean: {mean_prob:.4f} | Max: {max_prob:.4f}")
    
    # Check if outputs are identical
    print("\n" + "=" * 70)
    print("COMPARING OUTPUTS")
    print("=" * 70)
    
    all_outputs = list(outputs_raw.values())
    for i in range(len(all_outputs)-1):
        for j in range(i+1, len(all_outputs)):
            name1 = list(outputs_raw.keys())[i]
            name2 = list(outputs_raw.keys())[j]
            are_identical = np.allclose(all_outputs[i], all_outputs[j], atol=1e-5)
            status = "❌ IDENTICAL" if are_identical else "✓ DIFFERENT"
            print(f"{name1:20} vs {name2:20} : {status}")
    
    # Check if results are all the same
    values = [r['forgery_pct'] for r in results.values()]
    print("\n" + "=" * 70)
    if len(set(values)) == 1:
        print("❌ ERROR: All images return the SAME forgery percentage!")
        print("   This indicates the model is not processing the images correctly.")
        return False
    else:
        print(f"✓ GOOD: Model returns DIFFERENT results for different images")
        print(f"   Range: {min(values):.2f}% - {max(values):.2f}%")
        print(f"   Values: {[f'{v:.2f}%' for v in values]}")
        return True

if __name__ == '__main__':
    try:
        success = test_different_images()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

