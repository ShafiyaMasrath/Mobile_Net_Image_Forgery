"""
Check if the model is truly processing inputs or returning constant outputs
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model.mobforge_net import MobForgeNet
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model_processing():
    """Test if model actually processes different inputs"""
    
    weights_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pth')
    model = MobForgeNet(pretrained=False).to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("=" * 70)
    print("CHECKING IF MODEL PROCESSES INPUTS CORRECTLY")
    print("=" * 70)
    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Test 1: Multiple runs with SAME input
    print("\nTest 1: Same input, multiple times")
    print("-" * 70)
    test_img = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))
    
    outputs_same = []
    with torch.no_grad():
        for i in range(3):
            tensor = transform(test_img).unsqueeze(0).to(DEVICE)
            output = model(tensor)
            outputs_same.append(output.squeeze().cpu().numpy())
            print(f"  Run {i+1}: min={output.min():.6f}, max={output.max():.6f}, mean={output.mean():.6f}")
    
    # Check if outputs are identical
    identical_runs = np.allclose(outputs_same[0], outputs_same[1]) and np.allclose(outputs_same[1], outputs_same[2])
    print(f"  All outputs identical: {identical_runs} (expected: True for same input)")
    
    # Test 2: DIFFERENT inputs
    print("\nTest 2: Different inputs")
    print("-" * 70)
    
    inputs_info = [
        ('All zeros', np.zeros((256, 256, 3), dtype=np.uint8)),
        ('All ones (255)', np.full((256, 256, 3), 255, dtype=np.uint8)),
        ('All 128', np.full((256, 256, 3), 128, dtype=np.uint8)),
        ('Random', np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)),
        ('Random 2', np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)),
    ]
    
    outputs_diff = {}
    with torch.no_grad():
        for name, img_array in inputs_info:
            test_img = Image.fromarray(img_array)
            tensor = transform(test_img).unsqueeze(0).to(DEVICE)
            output = model(tensor)
            output_np = output.squeeze().cpu().numpy()
            outputs_diff[name] = output_np
            print(f"  {name:20}: min={output.min():.7f}, max={output.max():.7f}, mean={output.mean():.7f}, std={output.std():.7f}")
    
    # Compare all outputs
    print("\nTest 3: Comparing different outputs")
    print("-" * 70)
    names = list(outputs_diff.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            are_close = np.allclose(outputs_diff[names[i]], outputs_diff[names[j]], atol=1e-6)
            status = "SAME   ❌" if are_close else "DIFF   ✓"
            print(f"  {names[i]:20} vs {names[j]:20}: {status}")
    
    # Test 4: Check spatial variation
    print("\nTest 4: Checking if output has spatial variation")
    print("-" * 70)
    random_output = outputs_diff['Random']
    print(f"  Output shape: {random_output.shape}")
    print(f"  Output min value: {random_output.min():.7f}")
    print(f"  Output max value: {random_output.max():.7f}")
    print(f"  Output mean: {random_output.mean():.7f}")
    print(f"  Output std: {random_output.std():.7f}")
    
    # Count unique values (with some tolerance)
    unique_approx = len(np.unique(np.round(random_output * 1e6)))
    print(f"  Approx unique values (rounded): {unique_approx}")
    
    if random_output.std() < 0.0001:
        print(f"  ⚠️  WARNING: Output has very low spatial variance!")
        print(f"     This suggests the model might be BROKEN or not processing input")
    else:
        print(f"  ✓ Output has reasonable spatial variance")
    
    # Final check
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    all_very_similar = all(np.allclose(outputs_diff[names[i]], outputs_diff[names[j]], atol=1e-5) 
                           for i in range(len(names)) for j in range(i+1, len(names)))
    
    if all_very_similar:
        print("❌ MODEL IS BROKEN: All inputs produce nearly identical outputs!")
        print("   Possible causes:")
        print("   1. Model weights aren't loaded correctly")
        print("   2. Model isn't actually being used (constant initialization)")
        print("   3. Input preprocessing has issue")
        return False
    else:
        print("✓ Model processes inputs correctly (outputs vary with input)")
        return True

if __name__ == '__main__':
    success = test_model_processing()
    exit(0 if success else 1)
