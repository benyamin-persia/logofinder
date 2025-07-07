#!/usr/bin/env python3
"""
Test script specifically for Oracle.com to verify enhanced image extraction
"""

import asyncio
import os
import tempfile
from image_downloader import ImageDownloader

async def test_oracle_extraction():
    """Test image extraction from Oracle.com"""
    
    # Create a temporary test CSV file with Oracle URL
    test_csv_content = """URL
https://www.oracle.com/
"""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        test_csv_path = os.path.join(temp_dir, "oracle_test.csv")
        test_output_dir = os.path.join(temp_dir, "oracle_images")
        
        # Write test CSV
        with open(test_csv_path, 'w') as f:
            f.write(test_csv_content)
        
        print("Testing Enhanced Image Extraction on Oracle.com...")
        print(f"Test CSV: {test_csv_path}")
        print(f"Output directory: {test_output_dir}")
        print("This may take a few minutes as Oracle.com is a complex site...")
        
        # Create and run downloader
        async with ImageDownloader(csv_file=test_csv_path, output_dir=test_output_dir) as downloader:
            await downloader.run()
        
        # Check if files were downloaded
        if os.path.exists(test_output_dir):
            oracle_folder = os.path.join(test_output_dir, "oracle_com")
            if os.path.exists(oracle_folder):
                files = os.listdir(oracle_folder)
                print(f"\nDownloaded {len(files)} images from Oracle.com:")
                for i, file in enumerate(files[:10], 1):  # Show first 10 files
                    print(f"  {i}. {file}")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more files")
                
                if files:
                    print(f"\n✅ Test passed! Successfully extracted {len(files)} images from Oracle.com")
                    return True
                else:
                    print("\n❌ Test failed! No images were extracted from Oracle.com")
                    return False
            else:
                print("\n❌ Test failed! Oracle folder was not created")
                return False
        else:
            print("\n❌ Test failed! Output directory was not created")
            return False

if __name__ == "__main__":
    print("Oracle.com Image Extraction Test")
    print("=" * 50)
    result = asyncio.run(test_oracle_extraction())
    exit(0 if result else 1) 