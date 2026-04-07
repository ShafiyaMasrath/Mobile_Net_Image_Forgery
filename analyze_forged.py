"""
Detailed analysis of model output for the forged image
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model.mobforge_net import MobForgeNet
import os
import sys

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_forged_image(image_path, weights_path):
    """Analyze what the model outputs for the forged image"""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    print("=" * 70)
    print("ANALYZING FORGED IMAGE")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"File exists: {os.path.exists(image_path)}")
    
    # Load model
    print(f"\nLoading model from {weights_path}...")
    model = MobForgeNet(pretrained=False).to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model loaded")
    
    # Load image
    original_img = Image.open(image_path).convert('RGB')
    print(f"\n📷 Image size: {original_img.size}")
    
    # Preprocess
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(original_img).unsqueeze(0).to(DEVICE)
    
    # Run inference
    print("\n🔍 Running inference...")
    with torch.no_grad():
        pred = model(tensor)
    
    prob_map = pred[0, 0].cpu().numpy()  # [256, 256]
    
    print("\n" + "=" * 70)
    print("DETAILED OUTPUT ANALYSIS")
    print("=" * 70)
    
    print(f"\nProbability map statistics (at 256x256 model output):")
    print(f"  Min value:  {prob_map.min():.7f}")
    print(f"  Max value:  {prob_map.max():.7f}")
    print(f"  Mean value: {prob_map.mean():.7f}")
    print(f"  Median:     {np.median(prob_map):.7f}")
    print(f"  Std dev:    {prob_map.std():.7f}")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(prob_map, p)
        print(f"  {p:2d}th: {val:.7f}")
    
    # Count pixels above different thresholds
    print(f"\nPixels above threshold:")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        count = (prob_map > threshold).sum()
        pct = count / prob_map.size * 100
        print(f"  > {threshold:.1f}: {count:6d} pixels ({pct:5.2f}%)")
    
    # Resize to original size
    prob_resized = Image.fromarray((prob_map * 255).astype(np.uint8))
    prob_resized = np.array(prob_resized.resize(original_img.size, Image.BILINEAR))
    prob_resized_norm = prob_resized.astype(np.float32) / 255.0
    
    print(f"\nAfter resizing to original size {original_img.size}:")
    print(f"  Min value:  {prob_resized_norm.min():.7f}")
    print(f"  Max value:  {prob_resized_norm.max():.7f}")
    print(f"  Mean value: {prob_resized_norm.mean():.7f}")
    
    # Using app.py logic (threshold 0.5)
    forgery_pct_app = (prob_resized_norm > 0.5).mean() * 100
    print(f"\nUsing app.py threshold (0.5): {forgery_pct_app:.2f}% forged")
    
    # Try different thresholds
    print(f"\nWith different thresholds:")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        pct = (prob_resized_norm > thresh).mean() * 100
        print(f"  Threshold {thresh}: {pct:.2f}% forged")
    
    # Histogram
    print(f"\nHistogram of probability values (256x256):")
    hist, _ = np.histogram(prob_map, bins=20, range=(0, 1))
    for i, count in enumerate(hist):
        bin_start = i / 20
        bin_end = (i + 1) / 20
        bar = "█" * int(count / hist.max() * 30) if hist.max() > 0 else ""
        print(f"  [{bin_start:.2f}-{bin_end:.2f}): {bar} ({count})")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if prob_map.max() < 0.1:
        print("⚠️  Very low probabilities detected!")
        print("   The model is outputting values very close to 0 (authentic).")
        print("   Likely causes:")
        print("   1. Model not trained properly on this type of forgery")
        print("   2. Model needs retraining with better data")
        print("   3. Forgery type is too subtle for current model")
    else:
        print(f"✓ Model is detecting some regions with probability > 0.1")
    
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_forged.py <image_path>")
        print("Example: python analyze_forged.py /path/to/Harsimg.jpeg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    weights_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pth')
    
    analyze_forged_image(image_path, weights_path)
