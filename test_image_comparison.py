#!/usr/bin/env python3
"""
Test script for Image Comparison System
"""

import asyncio
import os
import tempfile
from image_comparison import ImageComparator

async def test_image_comparison():
    """Test the image comparison system"""
    
    print("🧪 Testing Image Comparison System")
    print("=" * 50)
    
    # Create a test comparator
    comparator = ImageComparator(similarity_threshold=0.90)  # 90% threshold for testing
    
    # Check if we have downloaded images to test with
    if not os.path.exists("downloaded_images"):
        print("❌ No downloaded images found. Please run the image downloader first.")
        print("   You can use option 4 in the launcher to download images from URLs.csv")
        return False
    
    # Count available images
    image_count = 0
    for root, dirs, files in os.walk("downloaded_images"):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_count += 1
    
    print(f"📁 Found {image_count} images in downloaded_images folder")
    
    if image_count == 0:
        print("❌ No images found to compare with.")
        print("   Please run the image downloader first to download some images.")
        return False
    
    # Show available images
    print("\n📋 Available images for comparison:")
    print("-" * 40)
    for root, dirs, files in os.walk("downloaded_images"):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                rel_path = os.path.relpath(os.path.join(root, file), os.getcwd())
                print(f"   • {rel_path}")
    
    print("\n" + "=" * 50)
    print("🎯 To test image comparison:")
    print("1. Run the launcher: python launcher.py")
    print("2. Choose option 8: 'Compare images (find similar images)'")
    print("3. Provide a reference image (file or URL)")
    print("4. The system will find images with >95% similarity")
    
    print("\n" + "=" * 50)
    print("📖 Usage Examples:")
    print("-" * 30)
    print("• Direct command line:")
    print("  python image_comparison.py --reference path/to/image.jpg")
    print("  python image_comparison.py --url https://example.com/image.png")
    print("  python image_comparison.py --threshold 0.90")
    
    print("\n• Interactive mode:")
    print("  python image_comparison.py")
    
    return True

async def demo_comparison():
    """Demonstrate comparison with a sample image if available"""
    
    print("\n🎬 DEMO: Image Comparison")
    print("=" * 40)
    
    # Look for a sample image in the downloaded_images folder
    sample_image = None
    for root, dirs, files in os.walk("downloaded_images"):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                sample_image = os.path.join(root, file)
                break
        if sample_image:
            break
    
    if sample_image:
        print(f"🔍 Using sample image: {os.path.relpath(sample_image, os.getcwd())}")
        
        try:
            # Initialize comparator
            comparator = ImageComparator(similarity_threshold=0.95)
            
            # Load the sample image
            reference_image = comparator.load_image_from_file(sample_image)
            
            # Find similar images
            print("🔍 Searching for similar images...")
            similar_images = comparator.find_similar_images(reference_image, "downloaded_images")
            
            # Display results
            comparator.display_results(similar_images, sample_image)
            
            return True
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            return False
    else:
        print("❌ No sample images found for demo.")
        return False

async def main():
    """Main test function"""
    print("Image Comparison System Test")
    print("=" * 60)
    
    # Test 1: Basic functionality
    result1 = await test_image_comparison()
    
    # Test 2: Demo comparison
    result2 = await demo_comparison()
    
    print("\n" + "=" * 60)
    if result1 and result2:
        print("🎉 All tests passed! Image comparison system is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return result1 and result2

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1) 