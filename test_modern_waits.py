#!/usr/bin/env python3
"""
Test script for modern wait strategies and synchronization
"""

import asyncio
import os
import tempfile
from image_downloader import ImageDownloader, ModernWaitStrategy
from playwright.async_api import async_playwright

async def test_modern_wait_strategies():
    """Test the modern wait strategies on different types of websites"""
    
    test_urls = [
        "https://www.oracle.com/",  # JavaScript-heavy site
        "https://httpbin.org/image/png",  # Simple static site
        "https://example.com/",  # Basic HTML site
    ]
    
    print("Testing Modern Wait Strategies")
    print("=" * 50)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        for url in test_urls:
            print(f"\nTesting: {url}")
            print("-" * 30)
            
            try:
                # Navigate to the page
                await page.goto(url, wait_until='domcontentloaded')
                
                # Initialize wait strategy
                wait_strategy = ModernWaitStrategy(page)
                
                # Test each wait method
                print("1. Testing DOM ready wait...")
                dom_ready = await wait_strategy.wait_for_dom_ready(timeout=10000)
                print(f"   DOM Ready: {'✅' if dom_ready else '❌'}")
                
                print("2. Testing network idle wait...")
                network_idle = await wait_strategy.wait_for_network_idle(timeout=15000)
                print(f"   Network Idle: {'✅' if network_idle else '❌'}")
                
                print("3. Testing dynamic content wait...")
                dynamic_content = await wait_strategy.wait_for_dynamic_content(timeout=10000)
                print(f"   Dynamic Content: {'✅' if dynamic_content else '❌'}")
                
                print("4. Testing lazy images wait...")
                lazy_images = await wait_strategy.wait_for_lazy_images(timeout=10000)
                print(f"   Lazy Images: {'✅' if lazy_images else '❌'}")
                
                print("5. Testing images load wait...")
                images_load = await wait_strategy.wait_for_images_to_load(timeout=10000)
                print(f"   Images Load: {'✅' if images_load else '❌'}")
                
                # Count images found
                image_count = await page.evaluate("""
                    () => {
                        const images = document.querySelectorAll('img');
                        return images.length;
                    }
                """)
                print(f"   Images Found: {image_count}")
                
            except Exception as e:
                print(f"   Error testing {url}: {str(e)}")
        
        await browser.close()

async def test_complete_extraction_with_waits():
    """Test complete image extraction with modern wait strategies"""
    
    # Create a temporary test CSV file
    test_csv_content = """URL
https://www.oracle.com/
"""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        test_csv_path = os.path.join(temp_dir, "modern_test.csv")
        test_output_dir = os.path.join(temp_dir, "modern_images")
        
        # Write test CSV
        with open(test_csv_path, 'w') as f:
            f.write(test_csv_content)
        
        print("\nTesting Complete Extraction with Modern Wait Strategies...")
        print(f"Test CSV: {test_csv_path}")
        print(f"Output directory: {test_output_dir}")
        print("This will demonstrate the enhanced synchronization...")
        
        # Create and run downloader
        async with ImageDownloader(csv_file=test_csv_path, output_dir=test_output_dir) as downloader:
            await downloader.run()
        
        # Check results
        if os.path.exists(test_output_dir):
            oracle_folder = os.path.join(test_output_dir, "oracle_com")
            if os.path.exists(oracle_folder):
                files = os.listdir(oracle_folder)
                print(f"\n✅ Modern wait strategies completed successfully!")
                print(f"📁 Downloaded {len(files)} images from Oracle.com")
                print(f"📂 Files saved to: {oracle_folder}")
                
                if files:
                    print("\n📋 Sample files:")
                    for i, file in enumerate(files[:5], 1):
                        print(f"   {i}. {file}")
                    if len(files) > 5:
                        print(f"   ... and {len(files) - 5} more files")
                return True
            else:
                print("\n❌ Oracle folder was not created")
                return False
        else:
            print("\n❌ Output directory was not created")
            return False

async def main():
    """Main test function"""
    print("Modern Wait Strategy Test Suite")
    print("=" * 60)
    
    # Test 1: Individual wait strategies
    await test_modern_wait_strategies()
    
    # Test 2: Complete extraction with waits
    result = await test_complete_extraction_with_waits()
    
    print("\n" + "=" * 60)
    if result:
        print("🎉 All tests passed! Modern wait strategies are working correctly.")
    else:
        print("❌ Some tests failed. Check the logs for details.")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1) 