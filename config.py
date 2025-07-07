# Configuration file for Image Downloader

# File paths
CSV_FILE = "URLs.csv"
OUTPUT_DIR = "downloaded_images"

# Browser settings
HEADLESS = True  # Set to True for headless mode (faster, no visible browser)
BROWSER_TYPE = "chromium"  # Options: chromium, firefox, webkit
VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Timeout settings (in milliseconds)
PAGE_TIMEOUT = 120000  # 2 minutes
DOWNLOAD_TIMEOUT = 60000  # 1 minute
NETWORK_IDLE_TIMEOUT = 60000  # 1 minute

# Download settings
MAX_CONCURRENT_DOWNLOADS = 5  # Reduced for better reliability
DELAY_BETWEEN_WEBSITES = 3  # seconds
SCROLL_DELAY = 200  # milliseconds - slower scrolling for better loading

# Image filtering
MIN_IMAGE_SIZE = 1024  # bytes (1KB)
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # bytes (50MB)
ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico']

# Logging
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FILE = "image_downloader.log"

# Advanced settings
ENABLE_BACKGROUND_IMAGES = True
ENABLE_PICTURE_ELEMENTS = True
ENABLE_SRCSET_PARSING = True
ENABLE_LAZY_LOADING = True

# Modern Wait Strategy settings
WAIT_STRATEGY_ENABLED = True
DOM_READY_TIMEOUT = 15000  # milliseconds
NETWORK_IDLE_TIMEOUT_WAIT = 30000  # milliseconds
DYNAMIC_CONTENT_TIMEOUT = 20000  # milliseconds
LAZY_IMAGES_TIMEOUT = 25000  # milliseconds
IMAGES_LOAD_TIMEOUT = 20000  # milliseconds
ELEMENT_WAIT_TIMEOUT = 10000  # milliseconds 