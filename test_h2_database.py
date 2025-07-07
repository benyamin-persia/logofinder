#!/usr/bin/env python3
"""
Diagnostic test script for H2 Database website image extraction
"""

import asyncio
import os
import tempfile
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse

async def diagnose_h2_database():
    """Diagnose why images aren't being found on H2 Database website"""
    
    url = "https://h2database.com/html/main.html"
    
    print("🔍 Diagnosing H2 Database Website")
    print("=" * 50)
    print(f"URL: {url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            # Navigate to the page
            print("\n1. Navigating to page...")
            await page.goto(url, wait_until='domcontentloaded')
            
            # Wait a bit for any dynamic content
            await asyncio.sleep(3)
            
            # Check page title and basic info
            title = await page.title()
            print(f"   Page Title: {title}")
            
            # Method 1: Check for img elements
            print("\n2. Checking for <img> elements...")
            images = await page.query_selector_all('img')
            print(f"   Found {len(images)} <img> elements")
            
            for i, img in enumerate(images[:5], 1):  # Show first 5
                try:
                    src = await img.get_attribute('src')
                    alt = await img.get_attribute('alt')
                    print(f"   Image {i}: src='{src}', alt='{alt}'")
                except Exception as e:
                    print(f"   Image {i}: Error reading attributes - {e}")
            
            # Method 2: Check for background images
            print("\n3. Checking for background images...")
            background_images = await page.evaluate("""
                () => {
                    const images = [];
                    const elements = document.querySelectorAll('*');
                    elements.forEach(el => {
                        const style = window.getComputedStyle(el);
                        const bgImage = style.backgroundImage;
                        if (bgImage && bgImage !== 'none') {
                            images.push(bgImage);
                        }
                    });
                    return [...new Set(images)];
                }
            """)
            print(f"   Found {len(background_images)} background images")
            for bg in background_images[:3]:  # Show first 3
                print(f"   Background: {bg}")
            
            # Method 3: Check page source for image references
            print("\n4. Checking page source for image references...")
            page_content = await page.content()
            
            # Look for common image patterns
            import re
            img_patterns = [
                r'src=["\']([^"\']*\.(?:png|jpg|jpeg|gif|svg|webp))["\']',
                r'background-image:\s*url\(["\']?([^"\')\s]+\.(?:png|jpg|jpeg|gif|svg|webp))["\']?\)',
                r'url\(["\']?([^"\')\s]+\.(?:png|jpg|jpeg|gif|svg|webp))["\']?\)'
            ]
            
            all_found_images = []
            for pattern in img_patterns:
                matches = re.findall(pattern, page_content, re.IGNORECASE)
                all_found_images.extend(matches)
            
            print(f"   Found {len(all_found_images)} image references in source")
            for img in all_found_images[:5]:  # Show first 5
                print(f"   Source reference: {img}")
            
            # Method 4: Test URL resolution
            print("\n5. Testing URL resolution...")
            base_url = "https://h2database.com/html/"
            for img in all_found_images[:3]:
                if img.startswith('http'):
                    absolute_url = img
                else:
                    absolute_url = urljoin(base_url, img)
                print(f"   Relative: {img} -> Absolute: {absolute_url}")
            
            # Method 5: Check if images are actually accessible
            print("\n6. Testing image accessibility...")
            for img in all_found_images[:3]:
                if img.startswith('http'):
                    test_url = img
                else:
                    test_url = urljoin(base_url, img)
                
                try:
                    response = await page.goto(test_url, wait_until='domcontentloaded')
                    if response and response.status == 200:
                        print(f"   ✅ {test_url} - Accessible")
                    else:
                        print(f"   ❌ {test_url} - Status: {response.status if response else 'No response'}")
                except Exception as e:
                    print(f"   ❌ {test_url} - Error: {e}")
            
            # Method 6: Try different base URLs
            print("\n7. Testing different base URLs...")
            possible_bases = [
                "https://h2database.com/html/",
                "https://h2database.com/",
                "https://h2database.com/html/main.html"
            ]
            
            for base in possible_bases:
                print(f"   Testing base: {base}")
                for img in all_found_images[:2]:
                    absolute_url = urljoin(base, img)
                    print(f"     {img} -> {absolute_url}")
            
        except Exception as e:
            print(f"❌ Error during diagnosis: {e}")
        finally:
            await browser.close()

async def test_h2_extraction():
    """Test the actual image extraction on H2 Database"""
    
    # Create a temporary test CSV file
    test_csv_content = """URL
https://h2database.com/html/main.html
"""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        test_csv_path = os.path.join(temp_dir, "h2_test.csv")
        test_output_dir = os.path.join(temp_dir, "h2_images")
        
        # Write test CSV
        with open(test_csv_path, 'w') as f:
            f.write(test_csv_content)
        
        print("\n" + "=" * 50)
        print("Testing H2 Database Image Extraction")
        print("=" * 50)
        
        # Import and run the downloader
        from image_downloader import ImageDownloader
        
        async with ImageDownloader(csv_file=test_csv_path, output_dir=test_output_dir) as downloader:
            await downloader.run()
        
        # Check results
        if os.path.exists(test_output_dir):
            h2_folder = os.path.join(test_output_dir, "h2database_com")
            if os.path.exists(h2_folder):
                files = os.listdir(h2_folder)
                print(f"\n📁 Downloaded {len(files)} images from H2 Database")
                print(f"📂 Files saved to: {h2_folder}")
                
                if files:
                    print("\n📋 Downloaded files:")
                    for i, file in enumerate(files, 1):
                        print(f"   {i}. {file}")
                    return True
                else:
                    print("\n❌ No images were downloaded")
                    return False
            else:
                print("\n❌ H2 Database folder was not created")
                return False
        else:
            print("\n❌ Output directory was not created")
            return False

async def main():
    """Main diagnostic function"""
    print("H2 Database Image Extraction Diagnostic")
    print("=" * 60)
    
    # Step 1: Diagnose the website
    await diagnose_h2_database()
    
    # Step 2: Test actual extraction
    result = await test_h2_extraction()
    
    print("\n" + "=" * 60)
    if result:
        print("✅ H2 Database extraction completed successfully!")
    else:
        print("❌ H2 Database extraction failed. Check the diagnosis above.")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1) 