# Image Downloader with Playwright

A powerful Python application that downloads all images from websites using Playwright in non-headless mode. The application organizes downloaded images into folders named after each website.

## Features

- **Comprehensive Image Detection**: Finds images from multiple sources:
  - Standard `<img>` tags with `src` attributes
  - Lazy-loaded images with `data-src` attributes
  - High-resolution images from `srcset` attributes
  - Background images from CSS
  - Images in `<picture>` elements
  - Multiple resolution variants

- **Smart Organization**: Creates separate folders for each website
- **Non-headless Mode**: Runs with visible browser for debugging
- **Concurrent Downloads**: Downloads multiple images simultaneously
- **Error Handling**: Robust error handling and logging
- **Duplicate Prevention**: Avoids downloading the same image twice
- **File Naming**: Sanitizes filenames for filesystem compatibility
- **Image Comparison**: Find images with >95% similarity to a reference image
- **Multiple Comparison Methods**: SSIM, MSE, Histogram, and SIFT feature matching

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Playwright browsers**:
   ```bash
   playwright install
   ```

## Usage

### Basic Usage

1. **Prepare your URLs**: Make sure your `URLs.csv` file contains the websites you want to process:
   ```csv
   URL
   https://example.com
   https://another-site.com
   ```

2. **Run the application**:
   ```bash
   python launcher.py
   ```

### Image Comparison

After downloading images, you can find similar images using the comparison system:

1. **Interactive Mode**:
   ```bash
   python launcher.py
   # Choose option 8: "Compare images (find similar images)"
   ```

2. **Command Line Mode**:
   ```bash
   # Compare with a reference image file
   python image_comparison.py --reference path/to/image.jpg
   
   # Compare with an image URL
   python image_comparison.py --url https://example.com/image.png
   
   # Adjust similarity threshold (default: 95%)
   python image_comparison.py --threshold 0.90
   ```

3. **Direct Script**:
   ```bash
   python image_comparison.py
   ```

### Configuration

You can customize the application by modifying the `ImageDownloader` class initialization:

```python
# Custom CSV file and output directory
downloader = ImageDownloader(
    csv_file="my_urls.csv",
    output_dir="my_images"
)
```

## Output Structure

The application creates the following folder structure:

```
downloaded_images/
├── example_com/
│   ├── logo.png
│   ├── banner.jpg
│   ├── icon.svg
│   └── ...
├── another_site_com/
│   ├── header.png
│   ├── product1.jpg
│   └── ...
└── ...
```

## Features in Detail

### Image Detection Methods

1. **Standard Images**: Extracts from `<img src="...">` tags
2. **Lazy-loaded Images**: Finds images in `data-src` attributes
3. **High-resolution Images**: Parses `srcset` attributes for multiple resolutions
4. **Background Images**: Extracts images from CSS `background-image` properties
5. **Picture Elements**: Finds images in `<picture><source>` elements

### Image Comparison Methods

The comparison system uses multiple algorithms to find similar images:

1. **Structural Similarity Index (SSIM)**: Compares structural information (40% weight)
2. **Mean Squared Error (MSE)**: Pixel-by-pixel comparison (30% weight)
3. **Histogram Comparison**: Color distribution analysis (20% weight)
4. **SIFT Feature Matching**: Scale-invariant feature detection (10% weight)

**Similarity Threshold**: Default 95% similarity threshold, adjustable via command line.

### Browser Behavior

- **Non-headless Mode**: Browser window is visible during operation
- **User Agent**: Uses a realistic Chrome user agent
- **Viewport**: Sets 1920x1080 viewport for consistent rendering
- **Scrolling**: Automatically scrolls pages to trigger lazy loading
- **Wait Times**: Waits for network idle and dynamic content loading

### Download Features

- **Concurrent Downloads**: Downloads multiple images simultaneously using asyncio
- **Content Type Validation**: Only downloads actual image files
- **Duplicate Prevention**: Tracks downloaded URLs to avoid duplicates
- **Filename Sanitization**: Creates safe filenames for all operating systems
- **Error Recovery**: Continues processing even if individual downloads fail

## Logging

The application provides comprehensive logging:

- **Console Output**: Real-time progress updates
- **Log File**: Detailed logs saved to `image_downloader.log`
- **Statistics**: Summary of downloaded vs failed images per website

## Error Handling

The application handles various error scenarios:

- **Network Errors**: Timeouts and connection failures
- **Invalid URLs**: Malformed or inaccessible URLs
- **File System Errors**: Permission issues and disk space problems
- **Browser Errors**: Playwright-specific issues

## Performance

- **Concurrent Processing**: Downloads multiple images simultaneously
- **Efficient Parsing**: Uses Playwright's efficient DOM querying
- **Memory Management**: Proper cleanup of browser instances and sessions
- **Rate Limiting**: Small delays between websites to be respectful

## Troubleshooting

### Common Issues

1. **Playwright not installed**:
   ```bash
   playwright install
   ```

2. **Permission errors**: Ensure write permissions to the output directory

3. **Network timeouts**: Some websites may be slow to load, the app handles this automatically

4. **Browser crashes**: The app automatically restarts browser instances

### Debug Mode

The application runs in non-headless mode by default, so you can see what's happening in the browser window. This helps with debugging issues.

## License

This project is open source and available under the MIT License.
