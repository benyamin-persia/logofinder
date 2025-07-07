#!/usr/bin/env python3
"""
Test script to demonstrate the unified image comparison approach
Shows the difference between inconsistent vs consistent preprocessing
"""

import os
import sys
import subprocess
import tempfile
from PIL import Image
import numpy as np

def create_test_images():
    """Create test images with different formats and sizes"""
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a simple reference image (red square)
    ref_img = Image.new('RGB', (100, 100), color='red')
    ref_path = os.path.join(test_dir, "reference.png")
    ref_img.save(ref_path)
    
    # Create similar images in different formats and sizes
    test_images = []
    
    # Same image, different format
    img1 = Image.new('RGB', (100, 100), color='red')
    img1_path = os.path.join(test_dir, "same_red.jpg")
    img1.save(img1_path, 'JPEG', quality=95)
    test_images.append(("same_red.jpg", "Same image, JPEG format"))
    
    # Same image, different size
    img2 = Image.new('RGB', (200, 200), color='red')
    img2 = img2.resize((50, 50), Image.Resampling.LANCZOS)
    img2_path = os.path.join(test_dir, "same_red_small.png")
    img2.save(img2_path)
    test_images.append(("same_red_small.png", "Same image, smaller size"))
    
    # Similar image (slightly different red)
    img3 = Image.new('RGB', (100, 100), color=(255, 0, 10))  # Slightly different red
    img3_path = os.path.join(test_dir, "similar_red.png")
    img3.save(img3_path)
    test_images.append(("similar_red.png", "Similar image, slightly different red"))
    
    # Different image (blue square)
    img4 = Image.new('RGB', (100, 100), color='blue')
    img4_path = os.path.join(test_dir, "different_blue.png")
    img4.save(img4_path)
    test_images.append(("different_blue.png", "Different image, blue square"))
    
    return ref_path, test_images

def test_unified_comparison():
    """Test the unified comparison approach"""
    print("Testing Unified Image Comparison with Consistent Preprocessing")
    print("=" * 60)
    
    # Create test images
    print("Creating test images...")
    ref_path, test_images = create_test_images()
    
    print(f"Reference image: {ref_path}")
    print("Test images:")
    for filename, description in test_images:
        print(f"  - {filename}: {description}")
    print()
    
    # Test unified comparison
    print("Running unified comparison (all methods)...")
    try:
        result = subprocess.run([
            sys.executable, "unified_image_comparison.py",
            "--reference", ref_path,
            "--folder", "test_images",
            "--topk", "5",
            "--method", "all",
            "--size", "256"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Unified comparison completed successfully!")
            print("\nResults:")
            print(result.stdout)
        else:
            print("❌ Unified comparison failed!")
            print("Error:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Unified comparison timed out!")
    except Exception as e:
        print(f"❌ Error running unified comparison: {e}")
    
    print("\n" + "=" * 60)
    print("Key Benefits of Unified Comparison:")
    print("1. All images converted to PNG format")
    print("2. All images resized to identical dimensions (256x256)")
    print("3. Consistent normalization for each method")
    print("4. True perceptual differences are revealed")
    print("5. No format/size bias in similarity scores")
    print("=" * 60)

if __name__ == "__main__":
    test_unified_comparison() 