"""
QUICK RETRAINING GUIDE FOR YOUR FORGED IMAGE

Since your current model doesn't detect forgeries, you need to retrain it.

Option 1: Use CASIA 2.0 Dataset (Recommended)
─────────────────────────────────────────────
1. Download from: http://forensics.idealtest.org/
2. Extract to a folder (e.g., ~/datasets/CASIA_2.0_Full)
3. Run:
   python prepare_casia.py --dataset_path ~/datasets/CASIA_2.0_Full --output_dir data
   python train.py --data_dir data --epochs 100 --batch_size 16

Option 2: Use Your Own Dataset (Faster)
─────────────────────────────────────────
Create this folder structure with your images:

data/
├── train/
│   ├── images/          ← Your training images (authentic + forged)
│   └── masks/           ← Binary masks (white=forged, black=authentic)
├── val/
│   ├── images/          ← Validation images
│   └── masks/           ← Validation masks

IMPORTANT: Each image MUST have a corresponding mask!
- authentic.jpg → authentic.png (all black if no forgery)
- forged.jpg → forged.png (white regions = forged areas)

Then train with:
   python train.py --data_dir data --epochs 50 --batch_size 8

Option 3: Quick Test with Minimal Data
───────────────────────────────────────
For testing, you need at least:
- 10 authentic images + 10 masks (0% forged)
- 10 forged images + 10 masks (marked regions)

Questions for you:
1. Do you have labeled data (images + forgery masks)?
2. Do you want to download CASIA 2.0 (2.5GB)?
3. What type of forgeries are in your images? (copy-move, splicing, etc.)

For now, check if your data folder exists:
"""

import os
data_path = os.path.join(os.path.dirname(__file__), 'data')
print(f"\nData directory exists: {os.path.exists(data_path)}")
if os.path.exists(data_path):
    for root, dirs, files in os.walk(data_path):
        level = root.replace(data_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        print(f'{subindent}Files: {len(files)}')
else:
    print("⚠️ Data directory NOT found. You need to create it to train.")
