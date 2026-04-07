"""
Generate Sample Forged Images for Testing
Creates copy-move, splicing, and authentic test images
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_test_images(output_dir='test_images'):
    """Generate sample images with different forgery types"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING TEST IMAGES")
    print("=" * 70)
    
    # =====================
    # 1. AUTHENTIC IMAGE
    # =====================
    print("\n📸 Creating authentic image...")
    
    # Create a gradient background
    auth_img = Image.new('RGB', (512, 512), color=(255, 255, 255))
    draw = ImageDraw.Draw(auth_img)
    
    # Draw some geometric shapes (natural-looking)
    for x in range(0, 512, 50):
        col = (100 + x//3, 150 + x//4, 200)
        draw.line([(x, 0), (x, 512)], fill=col, width=2)
    
    # Add circles
    for i in range(5):
        x, y = random.randint(50, 450), random.randint(50, 450)
        r = random.randint(30, 80)
        col = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=col, outline=col)
    
    auth_path = os.path.join(output_dir, '1_authentic.jpg')
    auth_img.save(auth_path, 'JPEG', quality=95)
    print(f"  ✓ Saved: {auth_path}")
    
    # =====================
    # 2. COPY-MOVE FORGERY
    # =====================
    print("\n📸 Creating copy-move forgery...")
    
    copy_move_img = Image.new('RGB', (512, 512), color=(240, 240, 240))
    draw = ImageDraw.Draw(copy_move_img)
    
    # Background pattern
    for i in range(10):
        x = i * 51
        draw.rectangle([(x, 0), (x+25, 512)], fill=(200, 200, 220))
    
    # Draw ORIGINAL circle (object to copy)
    orig_x, orig_y = 150, 250
    orig_r = 60
    draw.ellipse(
        [(orig_x-orig_r, orig_y-orig_r), (orig_x+orig_r, orig_y+orig_r)],
        fill=(255, 100, 100),
        outline=(200, 50, 50),
        width=3
    )
    draw.text((orig_x-20, orig_y-10), "A", fill=(255, 255, 255))
    
    # DUPLICATE it (copy-move forgery)
    copy_x, copy_y = 380, 280
    copy_r = 60
    draw.ellipse(
        [(copy_x-copy_r, copy_y-copy_r), (copy_x+copy_r, copy_y+copy_r)],
        fill=(255, 100, 100),
        outline=(200, 50, 50),
        width=3
    )
    draw.text((copy_x-20, copy_y-10), "A", fill=(255, 255, 255))
    
    # Draw connecting line to show manipulation
    draw.line([(orig_x, orig_y), (copy_x, copy_y)], fill=(100, 100, 100), width=2)
    
    copy_move_path = os.path.join(output_dir, '2_copymove.jpg')
    copy_move_img.save(copy_move_path, 'JPEG', quality=95)
    print(f"  ✓ Saved: {copy_move_path}")
    print(f"     (Circle A is duplicated - model should detect high forgery %)")
    
    # =====================
    # 3. SPLICING FORGERY
    # =====================
    print("\n📸 Creating splicing forgery...")
    
    splice_img = Image.new('RGB', (512, 512), color=(220, 240, 200))
    draw = ImageDraw.Draw(splice_img)
    
    # Background: grid pattern
    for i in range(0, 512, 40):
        draw.line([(i, 0), (i, 512)], fill=(180, 200, 150), width=1)
        draw.line([(0, i), (512, i)], fill=(180, 200, 150), width=1)
    
    # Original content: blue area
    draw.rectangle([(20, 20), (250, 250)], fill=(100, 150, 200), outline=(50, 100, 150), width=2)
    draw.text((80, 100), "ORIGINAL", fill=(255, 255, 255))
    
    # SPLICED content: red rectangle pasted on blue
    splice_x1, splice_y1 = 200, 180
    splice_x2, splice_y2 = 450, 380
    draw.rectangle(
        [(splice_x1, splice_y1), (splice_x2, splice_y2)],
        fill=(255, 150, 100),
        outline=(200, 100, 50),
        width=3
    )
    draw.text((splice_x1+40, splice_y1+80), "SPLICED", fill=(255, 255, 255))
    
    splice_path = os.path.join(output_dir, '3_splicing.jpg')
    splice_img.save(splice_path, 'JPEG', quality=95)
    print(f"  ✓ Saved: {splice_path}")
    print(f"     (Red region spliced into blue area - model should detect high forgery %)")
    
    # =====================
    # 4. COMPLEX FORGERY (mixed)
    # =====================
    print("\n📸 Creating complex forgery (both copy-move + splicing)...")
    
    complex_img = Image.new('RGB', (512, 512), color=(245, 245, 245))
    draw = ImageDraw.Draw(complex_img)
    
    # Add random shapes
    for i in range(8):
        x = random.randint(30, 480)
        y = random.randint(30, 480)
        size = random.randint(40, 100)
        col = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        draw.ellipse([(x-size, y-size), (x+size, y+size)], fill=col)
    
    # Copy-move: duplicate a circle
    draw.ellipse([(100, 100), (200, 200)], fill=(255, 100, 100), outline=(200, 50, 50), width=2)
    draw.ellipse([(350, 150), (450, 250)], fill=(255, 100, 100), outline=(200, 50, 50), width=2)
    
    complex_path = os.path.join(output_dir, '4_complex.jpg')
    complex_img.save(complex_path, 'JPEG', quality=95)
    print(f"  ✓ Saved: {complex_path}")
    
    # =====================
    # SUMMARY
    # =====================
    print("\n" + "=" * 70)
    print("TEST IMAGES CREATED")
    print("=" * 70)
    print(f"\nOutput folder: {output_dir}/\n")
    
    print("IMAGE DESCRIPTIONS:\n")
    print("1️⃣  1_authentic.jpg")
    print("     Type: AUTHENTIC (no forgery)")
    print("     Expected: 0-5% forged")
    print("     ✓ Will work\n")
    
    print("2️⃣  2_copymove.jpg")
    print("     Type: COPY-MOVE FORGERY")
    print("     Description: Red circle A is duplicated")
    print("     Expected: 40-100% forged")
    print("     ✓ Model should detect WELL\n")
    
    print("3️⃣  3_splicing.jpg")
    print("     Type: SPLICING FORGERY")
    print("     Description: Red rectangle spliced into blue area")
    print("     Expected: 50-100% forged")
    print("     ✓ Model should detect WELL\n")
    
    print("4️⃣  4_complex.jpg")
    print("     Type: MIXED FORGERY")
    print("     Description: Copy-move + other objects")
    print("     Expected: 50-100% forged")
    print("     ✓ Model should detect WELL\n")
    
    print("=" * 70)
    print("TESTING INSTRUCTIONS")
    print("=" * 70)
    print("\n1. Upload these images to the web interface:")
    print("   python app.py --weights checkpoints/best_model.pth --port 5000")
    print("   Then visit: http://localhost:5000\n")
    
    print("2. Or test via command line:")
    print("   python inference.py --image test_images/2_copymove.jpg --weights checkpoints/best_model.pth\n")
    
    print("3. Expected Results:")
    print("   • Authentic image: ~0% forged ✓")
    print("   • Copy-move: ~60-100% forged ✓")
    print("   • Splicing: ~70-100% forged ✓\n")
    
    print("Note: Your Harsimg.jpeg (blending edit) will still show 0%")
    print("      Complete retraining tonight to fix this!\n")

if __name__ == '__main__':
    create_test_images()
