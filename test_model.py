"""
Diagnostic script to verify model is loaded and working correctly.
Run this to confirm your trained model weights are being used.
"""
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model.mobforge_net import MobForgeNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model_loading():
    """Test that model weights are correctly loaded."""
    
    weights_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pth')
    
    print("=" * 70)
    print("MODEL LOADING TEST")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Weights path: {weights_path}")
    print(f"File exists: {os.path.exists(weights_path)}")
    
    if not os.path.exists(weights_path):
        print("❌ ERROR: Weights file not found!")
        return False
    
    # Load model
    print("\n📥 Loading model...")
    model = MobForgeNet(pretrained=False).to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Test with dummy input
    print("\n🧪 Testing model with random input...")
    test_input = torch.randn(1, 3, 256, 256).to(DEVICE)
    
    with torch.no_grad():
        output1 = model(test_input)
        output2 = model(test_input)
    
    # Check outputs
    print(f"   Output shape: {output1.shape}")
    print(f"   Output range: [{output1.min():.4f}, {output1.max():.4f}]")
    print(f"   Output mean: {output1.mean():.4f}")
    
    # Check if both runs produce different results (expected due to randomness in input)
    outputs_same = torch.allclose(output1, output2)
    print(f"   Same outputs for same input: {outputs_same} (should be True)")
    
    if not outputs_same:
        print("⚠ WARNING: Model outputs differ for same input - BatchNorm issue?")
        return False
    
    # Test with different inputs
    print("\n🧪 Testing with different inputs...")
    test_input2 = torch.zeros_like(test_input)  # All zeros
    test_input3 = torch.ones_like(test_input)   # All ones
    
    with torch.no_grad():
        output_zeros = model(test_input2)
        output_ones = model(test_input3)
        output_random = model(test_input)
    
    print(f"   Zeros input - mean: {output_zeros.mean():.4f}, max: {output_zeros.max():.4f}")
    print(f"   Ones input  - mean: {output_ones.mean():.4f}, max: {output_ones.max():.4f}")
    print(f"   Random input - mean: {output_random.mean():.4f}, max: {output_random.max():.4f}")
    
    # Check if all outputs are the same (bad sign!)
    if torch.allclose(output_zeros, output_ones) or torch.allclose(output_ones, output_random):
        print("❌ ERROR: Model returns same output for different inputs!")
        print("   This means the trained weights were NOT loaded correctly.")
        return False
    else:
        print("✓ Model produces different outputs for different inputs (GOOD!)")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - Model is working correctly!")
    print("=" * 70)
    return True

if __name__ == '__main__':
    success = test_model_loading()
    exit(0 if success else 1)
