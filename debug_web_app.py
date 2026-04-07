"""
Test the web app inference endpoint directly
"""
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model.mobforge_net import MobForgeNet
import io
import base64

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_web_app_inference():
    """Simulate web app inference to see what's happening"""
    
    # Load model same way app.py does
    weights_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pth')
    print(f"Loading model from: {weights_path}")
    print(f"File exists: {os.path.exists(weights_path)}")
    
    model = MobForgeNet(pretrained=False).to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create test images
    print("\n" + "=" * 70)
    print("SIMULATING WEB APP INFERENCE")
    print("=" * 70)
    
    # Save test images to temp files
    os.makedirs('test_uploads', exist_ok=True)
    
    # 1. Black image
    black_img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    black_path = 'test_uploads/black.jpg'
    black_img.save(black_path)
    
    # 2. White image
    white_img = Image.fromarray(np.full((256, 256, 3), 255, dtype=np.uint8))
    white_path = 'test_uploads/white.jpg'
    white_img.save(white_path)
    
    # 3. Half black half white
    half = np.hstack([
        np.zeros((256, 128, 3), dtype=np.uint8),
        np.full((256, 128, 3), 255, dtype=np.uint8)
    ])
    half_img = Image.fromarray(half)
    half_path = 'test_uploads/half.jpg'
    half_img.save(half_path)
    
    test_files = [
        ('Black image', black_path),
        ('White image', white_path),
        ('Half-half image', half_path),
    ]
    
    results = []
    
    for test_name, image_path in test_files:
        print(f"\n{test_name}: {image_path}")
        print("-" * 70)
        
        # Load and preprocess (same as app.py)
        original_img = Image.open(image_path).convert('RGB')
        original_size = original_img.size
        
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor = transform(original_img).unsqueeze(0).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            pred = model(tensor)
        
        prob_map = pred[0, 0].cpu().numpy()  # [H, W]
        
        # Convert to 0-255 like the app does
        prob_map_uint8 = (prob_map * 255).astype(np.uint8)
        
        # Resize back to original
        prob_resized = Image.fromarray(prob_map_uint8)
        prob_resized = np.array(prob_resized.resize(original_size, Image.BILINEAR))
        prob_resized_normalized = prob_resized.astype(np.float32) / 255.0
        
        # Calculate forgery percentage (using threshold 0.5 like app)
        forgery_pct_050 = (prob_resized_normalized > 0.5).mean() * 100
        
        # Calculate using threshold 127 (like binary mask in visualization)
        forgery_pct_127 = (prob_resized > 127).mean() * 100
        
        print(f"  Original size: {original_size}")
        print(f"  Probability map range: [{prob_map.min():.6f}, {prob_map.max():.6f}]")
        print(f"  After uint8 conversion: [{prob_map_uint8.min()}, {prob_map_uint8.max()}]")
        print(f"  After resize: [{prob_resized.min()}, {prob_resized.max()}]")
        print(f"  Forgery % (threshold 0.5): {forgery_pct_050:.2f}%")
        print(f"  Forgery % (threshold 127): {forgery_pct_127:.2f}%")
        print(f"  Mean probability: {prob_map.mean():.6f}")
        print(f"  Mean prob_resized: {prob_resized.mean():.2f}")
        
        results.append({
            'name': test_name,
            'forgery_pct': forgery_pct_050,
            'prob_map_min': prob_map.min(),
            'prob_map_max': prob_map.max(),
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for r in results:
        print(f"{r['name']:20}: {r['forgery_pct']:6.2f}% | Range: [{r['prob_map_min']:.6f}, {r['prob_map_max']:.6f}]")
    
    # Check if all are identical
    pcts = [r['forgery_pct'] for r in results]
    if len(set([round(p, 2) for p in pcts])) == 1:
        print("\n❌ WARNING: All test images show the SAME forgery percentage!")
        print("   This could mean:")
        print("   1. Model outputs are constant/incorrect")
        print("   2. Thresholding logic is wrong")
        print("   3. Model needs retraining")
    else:
        print("\n✓ GOOD: Different images show different forgery percentages")
    
    # Cleanup
    for _, path in test_files:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists('test_uploads') and not os.listdir('test_uploads'):
        os.rmdir('test_uploads')

if __name__ == '__main__':
    test_web_app_inference()
