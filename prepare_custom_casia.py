"""
Prepare simplified CASIA dataset (Forged/Original folders)
"""
import os
import shutil
import random
import argparse
from pathlib import Path

def prepare_custom_casia(dataset_path, output_dir, train_ratio=0.7):
    """
    Prepare CASIA with structure:
    dataset/
      ├── Forged/  (images with forgeries)
      └── Original/ (authentic images)
    """
    
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    
    forged_dir = dataset_path / "Forged"
    original_dir = dataset_path / "Original"
    
    # Check paths
    if not forged_dir.exists():
        print(f"❌ Forged folder not found: {forged_dir}")
        return False
    if not original_dir.exists():
        print(f"❌ Original folder not found: {original_dir}")
        return False
    
    # Create output structure
    output_train_img = output_dir / "train" / "images"
    output_train_msk = output_dir / "train" / "masks"
    output_val_img = output_dir / "val" / "images"
    output_val_msk = output_dir / "val" / "masks"
    
    for p in [output_train_img, output_train_msk, output_val_img, output_val_msk]:
        p.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PREPARING CUSTOM CASIA DATASET")
    print("=" * 70)
    
    # Get files
    forged_files = [f for f in os.listdir(forged_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    original_files = [f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"\nFound:")
    print(f"  Forged images: {len(forged_files)}")
    print(f"  Original images: {len(original_files)}")
    
    # Split into train/val
    total_forged = len(forged_files)
    total_original = len(original_files)
    
    train_forged = int(total_forged * train_ratio)
    train_original = int(total_original * train_ratio)
    
    random.shuffle(forged_files)
    random.shuffle(original_files)
    
    train_forged_files = forged_files[:train_forged]
    val_forged_files = forged_files[train_forged:]
    train_original_files = original_files[:train_original]
    val_original_files = original_files[train_original:]
    
    print(f"\nSplit ({train_ratio*100:.0f}% train / {(1-train_ratio)*100:.0f}% val):")
    print(f"  Train: {len(train_forged_files)} forged + {len(train_original_files)} authentic = {len(train_forged_files) + len(train_original_files)}")
    print(f"  Val:   {len(val_forged_files)} forged + {len(val_original_files)} authentic = {len(val_forged_files) + len(val_original_files)}")
    
    # Copy train forged images (mask = all white indicating full forgery)
    print("\n📋 Copying training data...")
    for fname in train_forged_files:
        src = forged_dir / fname
        dst_img = output_train_img / fname
        shutil.copy(src, dst_img)
        
        # Create white mask (full image is forged)
        from PIL import Image
        img = Image.open(src)
        mask = Image.new('L', img.size, 255)  # All white
        mask_name = fname.rsplit('.', 1)[0] + '.png'
        mask.save(output_train_msk / mask_name)
    
    # Copy train original images (mask = all black indicating no forgery)
    for fname in train_original_files:
        src = original_dir / fname
        dst_img = output_train_img / fname
        shutil.copy(src, dst_img)
        
        # Create black mask (no forgery)
        from PIL import Image
        img = Image.open(src)
        mask = Image.new('L', img.size, 0)  # All black
        mask_name = fname.rsplit('.', 1)[0] + '.png'
        mask.save(output_train_msk / mask_name)
    
    # Copy val forged images
    for fname in val_forged_files:
        src = forged_dir / fname
        dst_img = output_val_img / fname
        shutil.copy(src, dst_img)
        
        from PIL import Image
        img = Image.open(src)
        mask = Image.new('L', img.size, 255)
        mask_name = fname.rsplit('.', 1)[0] + '.png'
        mask.save(output_val_msk / mask_name)
    
    # Copy val original images
    for fname in val_original_files:
        src = original_dir / fname
        dst_img = output_val_img / fname
        shutil.copy(src, dst_img)
        
        from PIL import Image
        img = Image.open(src)
        mask = Image.new('L', img.size, 0)
        mask_name = fname.rsplit('.', 1)[0] + '.png'
        mask.save(output_val_msk / mask_name)
    
    print("\n✓ Dataset preparation complete!")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    train/")
    print(f"      images/  ({len(os.listdir(output_train_img))} files)")
    print(f"      masks/   ({len(os.listdir(output_train_msk))} files)")
    print(f"    val/")
    print(f"      images/  ({len(os.listdir(output_val_img))} files)")
    print(f"      masks/   ({len(os.listdir(output_val_msk))} files)")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Path to CASIA dataset')
    parser.add_argument('--output_dir', default='data', help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train/val split ratio')
    args = parser.parse_args()
    
    prepare_custom_casia(args.dataset_path, args.output_dir, args.train_ratio)
