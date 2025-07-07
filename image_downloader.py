import asyncio
import os
import re
import csv
import aiohttp
import aiofiles
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import logging
from pathlib import Path
import time
from typing import List, Set, Dict
import hashlib
from config import *

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

class ModernWaitStrategy:
    """Modern wait strategy class with explicit and implicit waits"""
    
    def __init__(self, page, timeout=30000):
        self.page = page
        self.timeout = timeout
    
    async def wait_for_element(self, selector: str, timeout: int = None) -> bool:
        """Explicit wait for element to be present"""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout or self.timeout)
            return True
        except PlaywrightTimeoutError:
            logging.warning(f"Element {selector} not found within timeout")
            return False
    
    async def wait_for_images_to_load(self, timeout: int = None) -> bool:
        """Wait for all images to load"""
        try:
            await self.page.wait_for_function("""
                () => {
                    const images = document.querySelectorAll('img');
                    return Array.from(images).every(img => img.complete);
                }
            """, timeout=timeout or self.timeout)
            return True
        except PlaywrightTimeoutError:
            logging.warning("Not all images loaded within timeout")
            return False
    
    async def wait_for_network_idle(self, timeout: int = None) -> bool:
        """Wait for network to be idle"""
        try:
            await self.page.wait_for_load_state('networkidle', timeout=timeout or self.timeout)
            return True
        except PlaywrightTimeoutError:
            logging.warning("Network did not become idle within timeout")
            return False
    
    async def wait_for_dom_ready(self, timeout: int = None) -> bool:
        """Wait for DOM to be ready"""
        try:
            await self.page.wait_for_load_state('domcontentloaded', timeout=timeout or self.timeout)
            return True
        except PlaywrightTimeoutError:
            logging.warning("DOM did not become ready within timeout")
            return False
    
    async def wait_for_lazy_images(self, timeout: int = None) -> bool:
        """Wait for lazy-loaded images to appear"""
        try:
            await self.page.wait_for_function("""
                () => {
                    const lazyImages = document.querySelectorAll('img[data-src], img[data-lazy-src], img[data-original]');
                    return lazyImages.length === 0 || Array.from(lazyImages).some(img => img.src && img.src !== '');
                }
            """, timeout=timeout or self.timeout)
            return True
        except PlaywrightTimeoutError:
            logging.warning("Lazy images did not load within timeout")
            return False
    
    async def wait_for_dynamic_content(self, timeout: int = None) -> bool:
        """Wait for dynamic content to load"""
        try:
            await self.page.wait_for_function("""
                () => {
                    // Wait for common dynamic content indicators
                    const indicators = [
                        document.querySelector('.loading') === null,
                        document.querySelector('.spinner') === null,
                        document.querySelector('[data-loading]') === null
                    ];
                    return indicators.every(indicator => indicator);
                }
            """, timeout=timeout or self.timeout)
            return True
        except PlaywrightTimeoutError:
            logging.warning("Dynamic content did not load within timeout")
            return False

class ImageDownloader:
    def __init__(self, csv_file: str = CSV_FILE, output_dir: str = OUTPUT_DIR):
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.downloaded_urls: Set[str] = set()
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_website_name(self, url: str) -> str:
        """Extract website name from URL for folder naming"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        # Replace dots and special characters with underscores
        domain = re.sub(r'[^\w\-_.]', '_', domain)
        return domain
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility"""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        return filename
    
    async def download_image(self, image_url: str, folder_path: str, website_name: str) -> bool:
        """Download a single image"""
        try:
            if image_url in self.downloaded_urls:
                return False
                
            # Skip data URLs and invalid URLs
            if image_url.startswith('data:') or not image_url.startswith(('http://', 'https://')):
                return False
            
            # Enhanced headers to avoid HTTP 406 errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'image',
                'Sec-Fetch-Mode': 'no-cors',
                'Sec-Fetch-Site': 'same-origin',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            async with self.session.get(image_url, headers=headers, timeout=DOWNLOAD_TIMEOUT/1000) as response:
                if response.status != 200:
                    logging.warning(f"Failed to download {image_url}: HTTP {response.status}")
                    # Try fallback method using browser for problematic sites
                    return await self.download_image_via_browser(image_url, folder_path, website_name)
                
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    logging.warning(f"Skipping {image_url}: Not an image ({content_type})")
                    return False
                
                # Get filename from URL or content-type
                filename = os.path.basename(urlparse(image_url).path)
                if not filename or '.' not in filename:
                    # Generate filename from content type
                    ext = content_type.split('/')[-1].split(';')[0]
                    if ext in ['jpeg', 'jpg']:
                        ext = 'jpg'
                    filename = f"image_{len(os.listdir(folder_path)) + 1}.{ext}"
                
                filename = self.sanitize_filename(filename)
                file_path = os.path.join(folder_path, filename)
                
                # Check if file already exists
                if os.path.exists(file_path):
                    # Generate unique filename
                    name, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(file_path):
                        filename = f"{name}_{counter}{ext}"
                        file_path = os.path.join(folder_path, filename)
                        counter += 1
                
                # Download and save image
                content = await response.read()
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(content)
                
                self.downloaded_urls.add(image_url)
                logging.info(f"Downloaded: {filename} from {image_url}")
                return True
                
        except Exception as e:
            logging.error(f"Error downloading {image_url}: {str(e)}")
            return False
    
    async def download_image_via_browser(self, image_url: str, folder_path: str, website_name: str) -> bool:
        """Fallback method to download images using browser when HTTP requests fail"""
        try:
            logging.info(f"Attempting browser-based download for {image_url}")
            
            # Create a temporary browser instance for this download
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)  # Headless for faster download
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                
                try:
                    page = await context.new_page()
                    
                    # Navigate to the image URL
                    response = await page.goto(image_url, wait_until='domcontentloaded')
                    
                    if response and response.status == 200:
                        # Get the image content
                        content = await page.content()
                        
                        # Extract image data from the page
                        image_data = await page.evaluate("""
                            () => {
                                const img = document.querySelector('img');
                                if (img) {
                                    // Try to get image data from canvas
                                    const canvas = document.createElement('canvas');
                                    const ctx = canvas.getContext('2d');
                                    canvas.width = img.naturalWidth || img.width;
                                    canvas.height = img.naturalHeight || img.height;
                                    ctx.drawImage(img, 0, 0);
                                    return canvas.toDataURL('image/png');
                                }
                                return null;
                            }
                        """)
                        
                        if image_data and image_data.startswith('data:image/'):
                            # Convert data URL to binary
                            import base64
                            header, encoded = image_data.split(",", 1)
                            content = base64.b64decode(encoded)
                            
                            # Generate filename
                            filename = os.path.basename(urlparse(image_url).path)
                            if not filename or '.' not in filename:
                                filename = f"image_{len(os.listdir(folder_path)) + 1}.png"
                            
                            filename = self.sanitize_filename(filename)
                            file_path = os.path.join(folder_path, filename)
                            
                            # Check if file already exists
                            if os.path.exists(file_path):
                                name, ext = os.path.splitext(filename)
                                counter = 1
                                while os.path.exists(file_path):
                                    filename = f"{name}_{counter}{ext}"
                                    file_path = os.path.join(folder_path, filename)
                                    counter += 1
                            
                            # Save the image
                            async with aiofiles.open(file_path, 'wb') as f:
                                await f.write(content)
                            
                            self.downloaded_urls.add(image_url)
                            logging.info(f"Browser download successful: {filename} from {image_url}")
                            return True
                        else:
                            logging.warning(f"Could not extract image data from {image_url}")
                            return False
                    else:
                        logging.warning(f"Browser failed to access {image_url}: HTTP {response.status if response else 'No response'}")
                        return False
                        
                finally:
                    await browser.close()
                    
        except Exception as e:
            logging.error(f"Browser download failed for {image_url}: {str(e)}")
            return False
    
    async def wait_for_page_ready(self, page, wait_strategy: ModernWaitStrategy) -> bool:
        """Modern wait strategy for page readiness"""
        if not WAIT_STRATEGY_ENABLED:
            logging.info("Wait strategy disabled, using basic wait...")
            await asyncio.sleep(5)
            return True
            
        logging.info("Using modern wait strategy for page readiness...")
        
        # Step 1: Wait for DOM to be ready
        if not await wait_strategy.wait_for_dom_ready(timeout=DOM_READY_TIMEOUT):
            logging.warning("DOM ready timeout, continuing anyway...")
        
        # Step 2: Wait for network to be idle
        if not await wait_strategy.wait_for_network_idle(timeout=NETWORK_IDLE_TIMEOUT_WAIT):
            logging.warning("Network idle timeout, continuing anyway...")
        
        # Step 3: Wait for dynamic content
        if not await wait_strategy.wait_for_dynamic_content(timeout=DYNAMIC_CONTENT_TIMEOUT):
            logging.warning("Dynamic content timeout, continuing anyway...")
        
        # Step 4: Wait for lazy images
        if not await wait_strategy.wait_for_lazy_images(timeout=LAZY_IMAGES_TIMEOUT):
            logging.warning("Lazy images timeout, continuing anyway...")
        
        # Step 5: Wait for images to load
        if not await wait_strategy.wait_for_images_to_load(timeout=IMAGES_LOAD_TIMEOUT):
            logging.warning("Images load timeout, continuing anyway...")
        
        # Additional wait for JavaScript execution
        await asyncio.sleep(3)
        
        return True
    
    async def extract_images_from_page(self, page, base_url: str) -> List[Dict]:
        """Extract all image URLs and their HTML element information from a webpage"""
        image_data = []
        
        try:
            # Initialize modern wait strategy
            wait_strategy = ModernWaitStrategy(page, timeout=NETWORK_IDLE_TIMEOUT)
            
            # Wait for page to be completely ready
            await self.wait_for_page_ready(page, wait_strategy)
            
            # Get the actual page URL for better base URL resolution
            current_url = page.url
            logging.info(f"Current page URL: {current_url}")
            logging.info(f"Base URL for resolution: {base_url}")
            
            # Use the current page URL as base if it's different from the provided base_url
            if current_url != base_url:
                # Extract the directory from current URL for better relative path resolution
                parsed_current = urlparse(current_url)
                if parsed_current.path.endswith('.html'):
                    # If it's a file, use the directory as base
                    base_url = f"{parsed_current.scheme}://{parsed_current.netloc}{os.path.dirname(parsed_current.path)}/"
                else:
                    base_url = current_url
                logging.info(f"Adjusted base URL: {base_url}")
            
            # Method 1: Get all image elements with various attributes
            images = await page.query_selector_all('img')
            
            for img in images:
                try:
                    # Get all possible image source attributes
                    src = await img.get_attribute('src')
                    srcset = await img.get_attribute('srcset')
                    data_src = await img.get_attribute('data-src')
                    data_srcset = await img.get_attribute('data-srcset')
                    data_lazy_src = await img.get_attribute('data-lazy-src')
                    data_original = await img.get_attribute('data-original')
                    data_image = await img.get_attribute('data-image')
                    data_thumb = await img.get_attribute('data-thumb')
                    data_full = await img.get_attribute('data-full')
                    
                    # Get element information
                    element_info = await page.evaluate("""
                        (element) => {
                            const rect = element.getBoundingClientRect();
                            return {
                                tagName: element.tagName.toLowerCase(),
                                className: element.className || '',
                                id: element.id || '',
                                alt: element.alt || '',
                                title: element.title || '',
                                width: rect.width,
                                height: rect.height,
                                position: {
                                    x: rect.left,
                                    y: rect.top
                                },
                                parentTag: element.parentElement ? element.parentElement.tagName.toLowerCase() : '',
                                parentClass: element.parentElement ? element.parentElement.className || '' : '',
                                parentId: element.parentElement ? element.parentElement.id || '' : ''
                            };
                        }
                    """, img)
                    
                    # Collect all possible image sources
                    sources = []
                    for attr_value in [src, data_src, data_lazy_src, data_original, data_image, data_thumb, data_full]:
                        if attr_value:
                            sources.append(attr_value)
                    
                    # Parse srcset for multiple resolutions
                    if srcset and ENABLE_SRCSET_PARSING:
                        srcset_urls = self.parse_srcset(srcset)
                        sources.extend(srcset_urls)
                    
                    if data_srcset and ENABLE_SRCSET_PARSING:
                        srcset_urls = self.parse_srcset(data_srcset)
                        sources.extend(srcset_urls)
                    
                    # Convert relative URLs to absolute and store with element info
                    for source in sources:
                        if source:
                            absolute_url = urljoin(base_url, source)
                            # Check if URL already exists to avoid duplicates
                            if not any(img_data['url'] == absolute_url for img_data in image_data):
                                image_data.append({
                                    'url': absolute_url,
                                    'element': element_info,
                                    'source_type': 'img_tag'
                                })
                                
                except Exception as e:
                    logging.warning(f"Error extracting image source: {str(e)}")
                    continue
            
            # Method 2: Extract background images from CSS
            if ENABLE_BACKGROUND_IMAGES:
                try:
                    background_elements = await page.evaluate("""
                        () => {
                            const elements = [];
                            const allElements = document.querySelectorAll('*');
                            allElements.forEach(el => {
                                const style = window.getComputedStyle(el);
                                const bgImage = style.backgroundImage;
                                if (bgImage && bgImage !== 'none') {
                                    const matches = bgImage.match(/url\\(['"]?([^'"]+)['"]?\\)/g);
                                    if (matches) {
                                        matches.forEach(match => {
                                            const url = match.replace(/url\\(['"]?([^'"]+)['"]?\\)/, '$1');
                                            if (url && !url.startsWith('data:')) {
                                                const rect = el.getBoundingClientRect();
                                                elements.push({
                                                    url: url,
                                                    element: {
                                                        tagName: el.tagName.toLowerCase(),
                                                        className: el.className || '',
                                                        id: el.id || '',
                                                        width: rect.width,
                                                        height: rect.height,
                                                        position: {
                                                            x: rect.left,
                                                            y: rect.top
                                                        },
                                                        parentTag: el.parentElement ? el.parentElement.tagName.toLowerCase() : '',
                                                        parentClass: el.parentElement ? el.parentElement.className || '' : '',
                                                        parentId: el.parentElement ? el.parentElement.id || '' : ''
                                                    }
                                                });
                                            }
                                        });
                                    }
                                }
                            });
                            return elements;
                        }
                    """)
                    
                    for bg_element in background_elements:
                        absolute_url = urljoin(base_url, bg_element['url'])
                        if not any(img_data['url'] == absolute_url for img_data in image_data):
                            image_data.append({
                                'url': absolute_url,
                                'element': bg_element['element'],
                                'source_type': 'background_image'
                            })
                            
                except Exception as e:
                    logging.warning(f"Error extracting background images: {str(e)}")
            
            # Method 3: Extract from picture elements
            if ENABLE_PICTURE_ELEMENTS:
                try:
                    picture_elements = await page.evaluate("""
                        () => {
                            const elements = [];
                            const pictures = document.querySelectorAll('picture source');
                            pictures.forEach(source => {
                                const srcset = source.srcset;
                                if (srcset) {
                                    const urls = srcset.split(',').map(s => s.trim().split(' ')[0]);
                                    const rect = source.getBoundingClientRect();
                                    urls.forEach(url => {
                                        elements.push({
                                            url: url,
                                            element: {
                                                tagName: source.tagName.toLowerCase(),
                                                className: source.className || '',
                                                id: source.id || '',
                                                width: rect.width,
                                                height: rect.height,
                                                position: {
                                                    x: rect.left,
                                                    y: rect.top
                                                },
                                                parentTag: source.parentElement ? source.parentElement.tagName.toLowerCase() : '',
                                                parentClass: source.parentElement ? source.parentElement.className || '' : '',
                                                parentId: source.parentElement ? source.parentElement.id || '' : ''
                                            }
                                        });
                                    });
                                }
                            });
                            return elements;
                        }
                    """)
                    
                    for pic_element in picture_elements:
                        absolute_url = urljoin(base_url, pic_element['url'])
                        if not any(img_data['url'] == absolute_url for img_data in image_data):
                            image_data.append({
                                'url': absolute_url,
                                'element': pic_element['element'],
                                'source_type': 'picture_element'
                            })
                            
                except Exception as e:
                    logging.warning(f"Error extracting picture images: {str(e)}")
            
            # Method 4: Extract from CSS files and stylesheets
            try:
                css_elements = await page.evaluate("""
                    () => {
                        const elements = [];
                        const styleSheets = Array.from(document.styleSheets);
                        styleSheets.forEach(sheet => {
                            try {
                                const rules = Array.from(sheet.cssRules || sheet.rules || []);
                                rules.forEach(rule => {
                                    if (rule.style && rule.style.backgroundImage) {
                                        const bgImage = rule.style.backgroundImage;
                                        const matches = bgImage.match(/url\\(['"]?([^'"]+)['"]?\\)/g);
                                        if (matches) {
                                            matches.forEach(match => {
                                                const url = match.replace(/url\\(['"]?([^'"]+)['"]?\\)/, '$1');
                                                if (url && !url.startsWith('data:')) {
                                                    elements.push({
                                                        url: url,
                                                        element: {
                                                            tagName: 'css_rule',
                                                            className: rule.selectorText || '',
                                                            id: '',
                                                            width: 0,
                                                            height: 0,
                                                            position: { x: 0, y: 0 },
                                                            parentTag: '',
                                                            parentClass: '',
                                                            parentId: ''
                                                        }
                                                    });
                                                }
                                            });
                                        }
                                    }
                                });
                            } catch (e) {
                                // CORS issues with external stylesheets
                            }
                        });
                        return elements;
                    }
                """)
                
                for css_element in css_elements:
                    absolute_url = urljoin(base_url, css_element['url'])
                    if not any(img_data['url'] == absolute_url for img_data in image_data):
                        image_data.append({
                            'url': absolute_url,
                            'element': css_element['element'],
                            'source_type': 'css_rule'
                        })
                        
            except Exception as e:
                logging.warning(f"Error extracting CSS images: {str(e)}")
            
            # Method 5: Extract from JSON-LD and other structured data
            try:
                structured_data_elements = await page.evaluate("""
                    () => {
                        const elements = [];
                        const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                        scripts.forEach(script => {
                            try {
                                const data = JSON.parse(script.textContent);
                                function extractImages(obj) {
                                    if (typeof obj === 'object' && obj !== null) {
                                        Object.keys(obj).forEach(key => {
                                            if (key === 'image' || key === 'logo' || key === 'thumbnail') {
                                                if (typeof obj[key] === 'string') {
                                                    elements.push({
                                                        url: obj[key],
                                                        element: {
                                                            tagName: 'script',
                                                            className: 'json-ld',
                                                            id: script.id || '',
                                                            width: 0,
                                                            height: 0,
                                                            position: { x: 0, y: 0 },
                                                            parentTag: '',
                                                            parentClass: '',
                                                            parentId: ''
                                                        }
                                                    });
                                                } else if (Array.isArray(obj[key])) {
                                                    obj[key].forEach(item => {
                                                        if (typeof item === 'string') {
                                                            elements.push({
                                                                url: item,
                                                                element: {
                                                                    tagName: 'script',
                                                                    className: 'json-ld',
                                                                    id: script.id || '',
                                                                    width: 0,
                                                                    height: 0,
                                                                    position: { x: 0, y: 0 },
                                                                    parentTag: '',
                                                                    parentClass: '',
                                                                    parentId: ''
                                                                }
                                                            });
                                                        }
                                                    });
                                                }
                                            } else if (typeof obj[key] === 'object') {
                                                extractImages(obj[key]);
                                            }
                                        });
                                    }
                                }
                                extractImages(data);
                            } catch (e) {
                                // Invalid JSON
                            }
                        });
                        return elements;
                    }
                """)
                
                for structured_element in structured_data_elements:
                    absolute_url = urljoin(base_url, structured_element['url'])
                    if not any(img_data['url'] == absolute_url for img_data in image_data):
                        image_data.append({
                            'url': absolute_url,
                            'element': structured_element['element'],
                            'source_type': 'structured_data'
                        })
                        
            except Exception as e:
                logging.warning(f"Error extracting structured data images: {str(e)}")
            
            logging.info(f"Extracted {len(image_data)} unique images with element information")
            return image_data
            
        except Exception as e:
            logging.error(f"Error extracting images from page: {str(e)}")
            return []
    
    def parse_srcset(self, srcset: str) -> List[str]:
        """Parse srcset attribute to extract image URLs"""
        urls = []
        try:
            # Split by comma and extract URLs
            parts = srcset.split(',')
            for part in parts:
                part = part.strip()
                # Extract URL (everything before the first space)
                url = part.split(' ')[0]
                if url:
                    urls.append(url)
        except Exception as e:
            logging.warning(f"Error parsing srcset '{srcset}': {str(e)}")
        return urls
    
    async def process_website(self, url: str) -> Dict[str, int]:
        """Process a single website and download all images"""
        stats = {'total_images': 0, 'downloaded': 0, 'failed': 0}
        
        try:
            website_name = self.get_website_name(url)
            folder_path = os.path.join(self.output_dir, website_name)
            
            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Create metadata file for this website
            metadata_file = os.path.join(folder_path, 'image_metadata.json')
            metadata = {
                'website_url': url,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'images': []
            }
            
            async with async_playwright() as p:
                browser_type = getattr(p, BROWSER_TYPE)
                browser = await browser_type.launch(
                    headless=HEADLESS,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-accelerated-2d-canvas',
                        '--no-first-run',
                        '--no-zygote',
                        '--disable-gpu'
                    ]
                )
                
                context = await browser.new_context(
                    viewport={'width': VIEWPORT_WIDTH, 'height': VIEWPORT_HEIGHT},
                    user_agent=USER_AGENT
                )
                
                page = await context.new_page()
                
                # Set timeouts
                page.set_default_timeout(PAGE_TIMEOUT)
                
                try:
                    logging.info(f"Navigating to {url}")
                    await page.goto(url, wait_until='domcontentloaded')
                    
                    # Scroll the page to load lazy images
                    await self.scroll_page(page)
                    
                    # Extract all image URLs with modern synchronization
                    image_data = await self.extract_images_from_page(page, url)
                    stats['total_images'] = len(image_data)
                    
                    logging.info(f"Found {len(image_data)} images on {website_name}")
                    
                    # Download all images with concurrency limit
                    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
                    
                    async def download_with_semaphore(img_data):
                        async with semaphore:
                            success = await self.download_image(img_data['url'], folder_path, website_name)
                            if success:
                                # Add metadata for successfully downloaded image
                                filename = os.path.basename(urlparse(img_data['url']).path)
                                if not filename or '.' not in filename:
                                    # Generate filename if none exists
                                    ext = 'jpg'  # default extension
                                    filename = f"image_{len(metadata['images']) + 1}.{ext}"
                                
                                # Find the actual downloaded file (with potential counter suffix)
                                actual_filename = None
                                for f in os.listdir(folder_path):
                                    if f.startswith(os.path.splitext(filename)[0]) and f.endswith(os.path.splitext(filename)[1]):
                                        actual_filename = f
                                        break
                                
                                if actual_filename:
                                    metadata['images'].append({
                                        'filename': actual_filename,
                                        'original_url': img_data['url'],
                                        'element_info': img_data['element'],
                                        'source_type': img_data['source_type'],
                                        'download_time': time.strftime('%Y-%m-%d %H:%M:%S')
                                    })
                            
                            return success
                    
                    download_tasks = []
                    for img_data in image_data:
                        task = download_with_semaphore(img_data)
                        download_tasks.append(task)
                    
                    # Wait for all downloads to complete
                    results = await asyncio.gather(*download_tasks, return_exceptions=True)
                    
                    # Count results
                    for result in results:
                        if isinstance(result, bool):
                            if result:
                                stats['downloaded'] += 1
                            else:
                                stats['failed'] += 1
                        else:
                            stats['failed'] += 1
                            logging.error(f"Download error: {result}")
                    
                    # Save metadata file
                    try:
                        import json
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                        logging.info(f"Saved metadata to {metadata_file}")
                    except Exception as e:
                        logging.error(f"Failed to save metadata: {e}")
                    
                except Exception as e:
                    logging.error(f"Error processing {url}: {str(e)}")
                    stats['failed'] = stats['total_images']
                
                finally:
                    await browser.close()
            
            # Add delay between websites
            if DELAY_BETWEEN_WEBSITES > 0:
                await asyncio.sleep(DELAY_BETWEEN_WEBSITES)
            
            logging.info(f"Completed {website_name}: {stats['downloaded']}/{stats['total_images']} images downloaded")
            return stats
            
        except Exception as e:
            logging.error(f"Error processing website {url}: {str(e)}")
            return {'total_images': 0, 'downloaded': 0, 'failed': 1}
    
    async def scroll_page(self, page):
        """Scroll the page to trigger lazy loading with modern wait strategy"""
        try:
            await page.evaluate(f"""
                () => {{
                    return new Promise((resolve) => {{
                        let totalHeight = 0;
                        const distance = 100;
                        const timer = setInterval(() => {{
                            const scrollHeight = document.body.scrollHeight;
                            window.scrollBy(0, distance);
                            totalHeight += distance;
                            
                            if(totalHeight >= scrollHeight){{
                                clearInterval(timer);
                                window.scrollTo(0, 0);
                                resolve();
                            }}
                        }}, {SCROLL_DELAY});
                    }});
                }}
            """)
        except Exception as e:
            logging.warning(f"Error scrolling page: {str(e)}")
    
    async def run(self):
        """Main execution method"""
        # Read URLs from CSV
        urls = []
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if 'URL' in row and row['URL'].strip():
                        urls.append(row['URL'].strip())
        except Exception as e:
            logging.error(f"Error reading CSV file: {str(e)}")
            return
        
        if not urls:
            logging.warning("No URLs found in CSV file")
            return
        
        logging.info(f"Found {len(urls)} URLs to process")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Process each website
        total_stats = {
            'total_images': 0,
            'downloaded': 0,
            'failed': 0
        }
        
        for i, url in enumerate(urls, 1):
            logging.info(f"Processing website {i}/{len(urls)}: {url}")
            stats = await self.process_website(url)
            
            total_stats['total_images'] += stats['total_images']
            total_stats['downloaded'] += stats['downloaded']
            total_stats['failed'] += stats['failed']
            
            # Small delay between websites
            await asyncio.sleep(DELAY_BETWEEN_WEBSITES)
        
        logging.info(f"Download completed! Total: {total_stats['downloaded']}/{total_stats['total_images']} images downloaded")
        logging.info(f"Failed: {total_stats['failed']} images")

async def main():
    """Main function"""
    async with ImageDownloader() as downloader:
        await downloader.run()

if __name__ == "__main__":
    asyncio.run(main()) 