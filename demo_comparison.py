#!/usr/bin/env python3
"""
Demonstration script for Image Comparison System
Shows how to find similar images with different thresholds
"""

import asyncio
import os
from image_comparison import ImageComparator

async def demo_lower_threshold():
    """Demonstrate comparison with a lower threshold to find more similar images"""
    
    print("🎬 DEMO: Image Comparison with Lower Threshold")
    print("=" * 60)
    
    # Check if we have downloaded images
    if not os.path.exists("downloaded_images"):
        print("❌ No downloaded images found. Please run the image downloader first.")
        return False
    
    # Find a sample image
    sample_image = None
    for root, dirs, files in os.walk("downloaded_images"):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                sample_image = os.path.join(root, file)
                break
        if sample_image:
            break
    
    if not sample_image:
        print("❌ No sample images found.")
        return False
    
    print(f"🔍 Using sample image: {os.path.relpath(sample_image, os.getcwd())}")
    
    try:
        # Test with different thresholds
        thresholds = [0.80, 0.85, 0.90, 0.95]
        
        for threshold in thresholds:
            print(f"\n📊 Testing with {threshold*100:.0f}% similarity threshold:")
            print("-" * 50)
            
            # Initialize comparator with current threshold
            comparator = ImageComparator(similarity_threshold=threshold)
            
            # Load the sample image
            reference_image = comparator.load_image_from_file(sample_image)
            
            # Find similar images
            print("🔍 Searching for similar images...")
            similar_images = comparator.find_similar_images(reference_image, "downloaded_images")
            
            # Display results
            print(f"Found {len(similar_images)} images with ≥{threshold*100:.0f}% similarity")
            
            if similar_images:
                print("Top matches:")
                for i, (img_path, similarity) in enumerate(similar_images[:5], 1):
                    rel_path = os.path.relpath(img_path, os.getcwd())
                    print(f"  {i}. {rel_path} ({similarity*100:.1f}%)")
            else:
                print("  No similar images found")
        
        print("\n" + "=" * 60)
        print("💡 Tips for using the image comparison system:")
        print("• Lower thresholds (80-85%) find more similar images")
        print("• Higher thresholds (95%+) find very close matches")
        print("• Use the launcher (option 8) for interactive comparison")
        print("• Command line: python image_comparison.py --threshold 0.85")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

async def main():
    """Main demonstration function"""
    print("Image Comparison System Demonstration")
    print("=" * 60)
    
    result = await demo_lower_threshold()
    
    print("\n" + "=" * 60)
    if result:
        print("🎉 Demonstration completed successfully!")
    else:
        print("⚠️  Demonstration failed. Check the output above.")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1) 