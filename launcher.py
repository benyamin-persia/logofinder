#!/usr/bin/env python3
"""
Launcher script for Image Downloader with menu options and flexible URL input
"""

import os
import sys
import asyncio
import pandas as pd
from image_downloader import ImageDownloader
import subprocess
import time

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("           IMAGE DOWNLOADER WITH PLAYWRIGHT")
    print("=" * 60)
    print()

def print_menu():
    """Print main menu options"""
    print("Choose an option:")
    print("1. Provide a file with URLs (CSV or Excel)")
    print("2. Extract images from URLs and find similar images to my reference (upload or link)")
    print("3. Compare images (LPIPS - perceptual similarity)")
    print("4. Unified Image Comparison (All Methods with Consistent Preprocessing)")
    print("5. Enter a single URL")
    print("6. Enter multiple URLs (comma or newline separated)")
    print("7. Run with default settings (URLs.csv)")
    print("8. Run basic test")
    print("9. Test modern wait strategies")
    print("10. View current configuration")
    print("11. Exit")
    print()

def get_user_input(prompt, default=""):
    """Get user input with optional default value"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()

def view_configuration():
    """Display current configuration"""
    try:
        from config import (
            CSV_FILE, OUTPUT_DIR, HEADLESS, BROWSER_TYPE, VIEWPORT_WIDTH, 
            VIEWPORT_HEIGHT, PAGE_TIMEOUT, DOWNLOAD_TIMEOUT, MAX_CONCURRENT_DOWNLOADS,
            DELAY_BETWEEN_WEBSITES, LOG_LEVEL, LOG_FILE
        )
        print("\nCurrent Configuration:")
        print("-" * 30)
        print(f"CSV File: {CSV_FILE}")
        print(f"Output Directory: {OUTPUT_DIR}")
        print(f"Headless Mode: {HEADLESS}")
        print(f"Browser Type: {BROWSER_TYPE}")
        print(f"Viewport: {VIEWPORT_WIDTH}x{VIEWPORT_HEIGHT}")
        print(f"Page Timeout: {PAGE_TIMEOUT}ms")
        print(f"Download Timeout: {DOWNLOAD_TIMEOUT}ms")
        print(f"Max Concurrent Downloads: {MAX_CONCURRENT_DOWNLOADS}")
        print(f"Delay Between Websites: {DELAY_BETWEEN_WEBSITES}s")
        print(f"Log Level: {LOG_LEVEL}")
        print(f"Log File: {LOG_FILE}")
        print()
    except ImportError:
        print("Configuration file not found!")

def write_urls_to_csv(urls, csv_path):
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['URL'])
        for url in urls:
            writer.writerow([url.strip()])

def extract_urls_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    urls = []
    if ext in ['.csv']:
        df = pd.read_csv(file_path)
        if 'URL' in df.columns:
            urls = df['URL'].dropna().astype(str).tolist()
        else:
            urls = df.iloc[:,0].dropna().astype(str).tolist()
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
        if 'URL' in df.columns:
            urls = df['URL'].dropna().astype(str).tolist()
        else:
            urls = df.iloc[:,0].dropna().astype(str).tolist()
    else:
        raise ValueError('Unsupported file type. Please provide a CSV or Excel file.')
    return urls

async def run_downloader_with_urls(urls):
    temp_csv = "_user_urls.csv"
    write_urls_to_csv(urls, temp_csv)
    try:
        async with ImageDownloader(csv_file=temp_csv) as downloader:
            await downloader.run()
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

async def run_downloader(csv_file=None, output_dir=None):
    """Run the image downloader with specified parameters"""
    try:
        # Use default values if not specified
        if csv_file is None:
            from config import CSV_FILE
            csv_file = CSV_FILE
        if output_dir is None:
            from config import OUTPUT_DIR
            output_dir = OUTPUT_DIR
        
        # Check if CSV file exists
        if not os.path.exists(csv_file):
            print(f"ERROR: CSV file '{csv_file}' not found!")
            return False
        
        print(f"Starting downloader...")
        print(f"CSV File: {csv_file}")
        print(f"Output Directory: {output_dir}")
        print()
        
        async with ImageDownloader(csv_file=csv_file, output_dir=output_dir) as downloader:
            await downloader.run()
        
        print("\nSUCCESS: Download completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Error running downloader: {str(e)}")
        return False

async def run_test():
    """Run the test script"""
    try:
        from test_downloader import test_downloader
        print("Running test...")
        result = await test_downloader()
        return result
    except Exception as e:
        print(f"ERROR: Error running test: {str(e)}")
        return False

async def run_modern_wait_test():
    """Run the modern wait strategies test script"""
    try:
        from test_modern_waits import main as test_modern_waits
        print("Running modern wait strategies test...")
        result = await test_modern_waits()
        return result
    except Exception as e:
        print(f"ERROR: Error running modern wait test: {str(e)}")
        return False

async def run_extract_and_compare():
    print("=== Extract Images and Find Similar Images ===")
    
    # Step 1: Ask for reference image first
    print("\nStep 1: Reference Image")
    ref_path = input("Enter the path or URL to your reference image: ").strip()
    if not ref_path:
        print("ERROR: Reference image path is required!")
        return False
    
    # Step 2: Ask for headless mode
    print("\nStep 2: Browser Mode")
    print("Headless mode: Faster, no visible browser window")
    print("Non-headless mode: Slower, but you can see the browser working")
    headless_choice = input("Use headless mode? (y/n, default: y): ").strip().lower()
    headless_mode = headless_choice != 'n'
    
    # Update configuration
    import config
    config.HEADLESS = headless_mode
    print(f"Browser mode: {'Headless' if headless_mode else 'Non-headless'}")
    
    # Step 3: Ask for thread count
    print("\nStep 3: Thread Configuration")
    while True:
        thread_input = input("Enter number of threads for downloading (1-10, default: 5): ").strip()
        if not thread_input:
            thread_count = 5
            break
        try:
            thread_count = int(thread_input)
            if 1 <= thread_count <= 10:
                break
            else:
                print("ERROR: Thread count must be between 1 and 10!")
        except ValueError:
            print("ERROR: Please enter a valid number!")
    
    # Step 4: Ask for top matches
    print("\nStep 4: Comparison Settings")
    topk_input = input("How many top matches to show? (default: 10): ").strip()
    topk = int(topk_input) if topk_input.isdigit() else 10
    
    print(f"\n=== Starting Process ===")
    print(f"Reference Image: {ref_path}")
    print(f"Browser Mode: {'Headless' if headless_mode else 'Non-headless'}")
    print(f"Thread Count: {thread_count}")
    print(f"Top Matches: {topk}")
    print(f"Starting download process...")
    
    # Step 5: Run the downloader with multi-threading
    start_time = time.time()
    download_stats = await run_downloader_with_threading(thread_count)
    download_time = time.time() - start_time

    # Step 5.5: Convert all images to PNG, RGB, and consistent size
    print(f"\n=== Unifying Downloaded Images: Converting to PNG, RGB, and {256}x{256} ===")
    import sys
    conversion_result = subprocess.run([
        sys.executable, "convert_all_to_png.py",
        "--folder", "downloaded_images",
        "--size", "256",
        "--delete-originals"
    ], capture_output=True, text=True)
    if conversion_result.returncode == 0:
        print(conversion_result.stdout)
    else:
        print("ERROR during PNG conversion:")
        print(conversion_result.stderr)
        return False

    # Step 6: Run image comparison
    print(f"\n=== Running Unified Image Comparison ===")
    print("Using consistent preprocessing for accurate results...")
    comparison_start = time.time()
    
    # Use unified comparison for best accuracy with consistent preprocessing
    result = subprocess.run([
        sys.executable, "unified_image_comparison.py", 
        "--reference", ref_path, 
        "--folder", "downloaded_images", 
        "--topk", str(topk),
        "--method", "all",
        "--size", "256"
    ], capture_output=True, text=True)
    comparison_time = time.time() - comparison_start
    
    # Step 7: Generate comprehensive report
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE REPORT")
    print(f"{'='*60}")
    
    print(f"\nTIMING:")
    print(f"   Download Time: {download_time:.2f} seconds")
    print(f"   Comparison Time: {comparison_time:.2f} seconds")
    print(f"   Total Time: {download_time + comparison_time:.2f} seconds")
    
    print(f"\nDOWNLOAD STATISTICS:")
    print(f"   Browser Mode: {'Headless' if headless_mode else 'Non-headless'}")
    print(f"   Threads Used: {thread_count}")
    print(f"   URLs Processed: {download_stats.get('urls_processed', 0)}")
    print(f"   Successful Downloads: {download_stats.get('successful', 0)}")
    print(f"   Failed Downloads: {download_stats.get('failed', 0)}")
    print(f"   Total Images Downloaded: {download_stats.get('total_images', 0)}")
    
    print(f"\nCOMPARISON RESULTS:")
    if result.returncode == 0:
        print(f"   Status: SUCCESS - Completed successfully")
        print(f"   Reference Image: {ref_path}")
        print(f"   Images Compared: {download_stats.get('total_images', 0)}")
        print(f"   Top Matches Requested: {topk}")
        if result.stdout:
            print(f"\n   Top Similar Images:")
            lines = result.stdout.strip().split('\n')
            for line in lines[-topk:]:  # Show last topk lines
                if line.strip():
                    print(f"     {line.strip()}")
    else:
        print(f"   Status: FAILED")
        print(f"   Error: {result.stderr}")
    
    print(f"\n{'='*60}")
    return True

async def run_extract_and_compare_lpips_only():
    print("=== Extract Images and Find Similar Images (LPIPS Only) ===")
    # Prompt for all comparison parameters first
    ref_path = input("Enter the path or URL to your reference image: ").strip()
    if not ref_path:
        print("ERROR: Reference image path is required!")
        return False
    topk = input("How many top matches to show? (default: 10): ").strip()
    topk = int(topk) if topk.isdigit() else 10
    threshold = input("LPIPS distance threshold (0.0-1.0, lower=more similar, default: 0.3): ").strip()
    threshold = float(threshold) if threshold and threshold.replace('.', '').isdigit() else 0.3
    # Step 1: Download images
    print("\nStep 1: Download Images from URLs")
    while True:
        thread_input = input("Enter number of threads for downloading (1-10, default: 5): ").strip()
        if not thread_input:
            thread_count = 5
            break
        try:
            thread_count = int(thread_input)
            if 1 <= thread_count <= 10:
                break
            else:
                print("ERROR: Thread count must be between 1 and 10!")
        except ValueError:
            print("ERROR: Please enter a valid number!")
    print(f"Starting download process with {thread_count} threads...")
    start_time = time.time()
    download_stats = await run_downloader_with_threading(thread_count)
    download_time = time.time() - start_time
    print(f"Download completed in {download_time:.2f} seconds.")
    # Step 2: Convert all images to PNG, RGB, and consistent size (DISABLED TEMPORARILY)
    # print(f"\n=== Unifying Downloaded Images: Converting to PNG, RGB, and 256x256 ===")
    # import sys
    # conversion_result = subprocess.run([
    #     sys.executable, "convert_all_to_png.py",
    #     "--folder", "downloaded_images",
    #     "--size", "256",
    #     "--delete-originals"
    # ], capture_output=True, text=True)
    # if conversion_result.returncode == 0:
    #     print(conversion_result.stdout)
    # else:
    #     print("ERROR during PNG conversion:")
    #     print(conversion_result.stderr)
    #     return False
    # Step 3: Run LPIPS comparison with pre-collected arguments
    print("\n=== Running LPIPS Image Comparison (perceptual similarity) ===")
    subprocess.run([
        sys.executable, "lpips_image_comparison.py",
        "--reference", ref_path,
        "--folder", "downloaded_images",
        "--topk", str(topk),
        "--threshold", str(threshold)
    ])
    print("=== LPIPS Image Comparison Finished ===\n")
    return True

async def run_downloader_with_threading(thread_count):
    """Run the downloader with multi-threading support"""
    stats = {
        'urls_processed': 0,
        'successful': 0,
        'failed': 0,
        'total_images': 0
    }
    
    try:
        # Read URLs from CSV or default file
        urls = []
        if os.path.exists('URLs.csv'):
            urls = extract_urls_from_file('URLs.csv')
        else:
            print("ERROR: URLs.csv not found. Please create it with your URLs.")
            return stats
        
        if not urls:
            print("ERROR: No URLs found in the file.")
            return stats
        
        print(f"Found {len(urls)} URLs to process")
        
        # Create thread pool for downloading
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # Process URLs in batches to avoid overwhelming the system
        batch_size = max(1, len(urls) // thread_count)
        batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
        
        print(f"Processing {len(batches)} batches with {thread_count} threads...")
        
        async def process_batch(batch_urls, batch_num):
            batch_stats = {'successful': 0, 'failed': 0, 'images': 0}
            
            for url in batch_urls:
                try:
                    print(f"   Processing: {url}")
                    # Create a unique temporary CSV file for this URL
                    temp_csv = f"_temp_urls_{batch_num}_{hash(url)}.csv"
                    write_urls_to_csv([url], temp_csv)
                    
                    try:
                        # Run the downloader for this URL
                        async with ImageDownloader(csv_file=temp_csv) as downloader:
                            await downloader.run()
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_csv):
                            os.remove(temp_csv)
                    
                    # Count images in the downloaded folder for this URL
                    # Use the same naming convention as the image downloader
                    from urllib.parse import urlparse
                    import re
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    website_name = re.sub(r'[^\w\-_.]', '_', domain)
                    url_folder = os.path.join('downloaded_images', website_name)
                    
                    if os.path.exists(url_folder):
                        images = [f for f in os.listdir(url_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'))]
                        batch_stats['images'] += len(images)
                        batch_stats['successful'] += 1
                        print(f"   SUCCESS: Downloaded {len(images)} images from {url}")
                    else:
                        batch_stats['failed'] += 1
                        print(f"   FAILED: No images downloaded from {url}")
                        
                except Exception as e:
                    print(f"   ERROR: Error processing {url}: {e}")
                    batch_stats['failed'] += 1
            
            return batch_stats
        
        # Run batches concurrently
        tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for batch_result in batch_results:
            if isinstance(batch_result, dict):
                stats['successful'] += batch_result.get('successful', 0)
                stats['failed'] += batch_result.get('failed', 0)
                stats['total_images'] += batch_result.get('images', 0)
        
        stats['urls_processed'] = len(urls)
        
        print(f"Download completed!")
        print(f"   Successful: {stats['successful']}")
        print(f"   Failed: {stats['failed']}")
        print(f"   Total Images: {stats['total_images']}")
        
    except Exception as e:
        print(f"ERROR: Error in threaded download: {e}")
    
    return stats

async def run_lpips_comparison():
    """Run the LPIPS image comparison system (perceptual similarity)"""
    try:
        print("Starting LPIPS Image Comparison (perceptual similarity)...")
        ref_path = input("Enter the path or URL to your reference image: ").strip()
        topk = input("How many top matches to show? (default: 10): ").strip()
        topk = int(topk) if topk.isdigit() else 10
        threshold = input("LPIPS distance threshold (0.0-1.0, lower=more similar, default: 0.3): ").strip()
        threshold = float(threshold) if threshold and threshold.replace('.', '').isdigit() else 0.3
        import sys
        subprocess.run([sys.executable, "lpips_image_comparison.py", "--reference", ref_path, "--folder", "downloaded_images", "--topk", str(topk), "--threshold", str(threshold)])
        return True
    except Exception as e:
        print(f"ERROR: Error running LPIPS comparison: {str(e)}")
        return False

async def run_unified_comparison():
    """Run the unified image comparison system with consistent preprocessing"""
    try:
        print("Starting Unified Image Comparison (All Methods with Consistent Preprocessing)...")
        print("This ensures all images are converted to PNG, resized identically, and normalized consistently")
        print("for accurate similarity comparison across all methods.")
        print()
        
        ref_path = input("Enter the path or URL to your reference image: ").strip()
        topk = input("How many top matches to show? (default: 10): ").strip()
        topk = int(topk) if topk.isdigit() else 10
        
        print("\nComparison Methods:")
        print("- resnet: ResNet50 (low-level visual features)")
        print("- clip: CLIP (semantic similarity)")
        print("- lpips: LPIPS (perceptual similarity)")
        print("- all: All methods combined (recommended)")
        method = input("Enter comparison method (resnet/clip/lpips/all, default: all): ").strip()
        if not method or method not in ['resnet', 'clip', 'lpips', 'all']:
            method = "all"
        
        size = input("Enter target image size for preprocessing (default: 256): ").strip()
        size = int(size) if size.isdigit() else 256
        
        print(f"\nRunning unified comparison with {method} method, size {size}x{size}...")
        import sys
        subprocess.run([sys.executable, "unified_image_comparison.py", "--reference", ref_path, "--folder", "downloaded_images", "--topk", str(topk), "--method", method, "--size", str(size)])
        return True
    except Exception as e:
        print(f"ERROR: Error running unified comparison: {str(e)}")
        return False

async def main():
    """Main launcher function"""
    print_banner()
    
    while True:
        print_menu()
        choice = get_user_input("Enter your choice (1-11)")
        
        if choice == "1":
            file_path = get_user_input("Enter the path to your CSV or Excel file")
            if file_path and os.path.exists(file_path):
                try:
                    urls = extract_urls_from_file(file_path)
                    if urls:
                        write_urls_to_csv(urls, 'URLs.csv')
                        print(f"Saved {len(urls)} URLs to URLs.csv. You can now use the other options to process these URLs.")
                    else:
                        print("No URLs found in the file.")
                except Exception as e:
                    print(f"Error reading file: {e}")
            else:
                print("File not found.")
        elif choice == "2":
            await run_extract_and_compare_lpips_only()
        elif choice == "3":
            await run_lpips_comparison()
        elif choice == "4":
            await run_unified_comparison()
        elif choice == "5":
            url = get_user_input("Enter the URL")
            if url:
                await run_downloader_with_urls([url])
        elif choice == "6":
            urls_input = get_user_input("Enter URLs (comma or newline separated)")
            if ',' in urls_input:
                urls = [u.strip() for u in urls_input.split(',') if u.strip()]
            else:
                urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
            if urls:
                await run_downloader_with_urls(urls)
        elif choice == "7":
            await run_downloader()
        elif choice == "8":
            await run_test()
        elif choice == "9":
            await run_modern_wait_test()
        elif choice == "10":
            view_configuration()
        elif choice == "11":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 11.")
        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nERROR: Unexpected error: {str(e)}")
        sys.exit(1) 