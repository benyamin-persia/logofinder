#!/usr/bin/env python3
"""
Test script for the Image Downloader application
"""

import asyncio
import os
import tempfile
import shutil
from image_downloader import ImageDownloader

async def test_downloader():
    """Test the image downloader with a simple website"""
    
    # Create a temporary test CSV file
    test_csv_content = """URL
https://httpbin.org/image/png
https://httpbin.org/image/jpeg
"""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        test_csv_path = os.path.join(temp_dir, "test_urls.csv")
        test_output_dir = os.path.join(temp_dir, "test_output")
        
        # Write test CSV
        with open(test_csv_path, 'w') as f:
            f.write(test_csv_content)
        
        print("Testing Image Downloader...")
        print(f"Test CSV: {test_csv_path}")
        print(f"Output directory: {test_output_dir}")
        
        # Create and run downloader
        async with ImageDownloader(csv_file=test_csv_path, output_dir=test_output_dir) as downloader:
            await downloader.run()
        
        # Check if files were downloaded
        if os.path.exists(test_output_dir):
            files = os.listdir(test_output_dir)
            print(f"Downloaded files: {files}")
            
            if files:
                print("✅ Test passed! Files were downloaded successfully.")
                return True
            else:
                print("❌ Test failed! No files were downloaded.")
                return False
        else:
            print("❌ Test failed! Output directory was not created.")
            return False

if __name__ == "__main__":
    result = asyncio.run(test_downloader())
    exit(0 if result else 1) 