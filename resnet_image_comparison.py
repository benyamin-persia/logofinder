#!/usr/bin/env python3
"""
Free Deep Learning Image Comparison (ResNet50, torchvision, no login required)
"""
import os
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from pathlib import Path
import argparse
import numpy as np
import requests
from io import BytesIO

def load_image(path):
    if path.startswith('http://') or path.startswith('https://'):
        # Add headers to avoid 403 errors
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
            'Sec-Fetch-Site': 'cross-site',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        response = requests.get(path, headers=headers, timeout=30)
        response.raise_for_status()
        content = response.content
        
        # Check if it's an SVG
        if path.lower().endswith('.svg') or b'<svg' in content[:1000]:
            try:
                import cairosvg
                # Convert SVG to PNG in memory
                png_data = cairosvg.svg2png(bytestring=content, output_width=256, output_height=256)
                return Image.open(BytesIO(png_data)).convert("RGB")
            except ImportError:
                print("Warning: cairosvg not installed. Install with: pip install cairosvg")
                raise Exception("SVG conversion requires cairosvg package")
            except Exception as e:
                print(f"Error converting SVG: {e}")
                raise
        
        return Image.open(BytesIO(content)).convert("RGB")
    
    # Local file
    if path.lower().endswith('.svg'):
        try:
            import cairosvg
            with open(path, 'rb') as f:
                svg_content = f.read()
            png_data = cairosvg.svg2png(bytestring=svg_content, output_width=256, output_height=256)
            return Image.open(BytesIO(png_data)).convert("RGB")
        except ImportError:
            print("Warning: cairosvg not installed. Install with: pip install cairosvg")
            raise Exception("SVG conversion requires cairosvg package")
        except Exception as e:
            print(f"Error converting SVG: {e}")
            raise
    
    return Image.open(path).convert("RGB")

def get_image_files(folder, exts={'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.svg'}):
    files = []
    for root, dirs, fs in os.walk(folder):
        for f in fs:
            if Path(f).suffix.lower() in exts:
                files.append(os.path.join(root, f))
    return files

def get_embedding(model, transform, image, device):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img_tensor)
    emb = emb.cpu().numpy().flatten()
    emb = emb / np.linalg.norm(emb)
    return emb

def get_original_url_from_path(image_path):
    """Extract the original URL from the image path"""
    try:
        # Path structure: downloaded_images/WEBSITE_NAME/image.jpg
        # WEBSITE_NAME is created by get_website_name() function
        parts = image_path.split(os.sep)
        if len(parts) >= 3 and parts[0] == 'downloaded_images':
            website_name = parts[1]
            
            # Try to find the metadata file to get the original URL
            folder_path = os.path.dirname(image_path)
            metadata_file = os.path.join(folder_path, 'image_metadata.json')
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return metadata.get('website_url', f"Unknown URL (folder: {website_name})")
            
            # Fallback: try to reconstruct URL from folder name
            # This is less reliable but works as backup
            return f"https://{website_name.replace('_', '.')}"
            
    except Exception as e:
        print(f"Warning: Could not extract URL from path {image_path}: {e}")
    return "Unknown URL"

def get_element_info_from_metadata(image_path):
    """Get HTML element information from metadata file"""
    try:
        # Find the metadata file in the same folder as the image
        folder_path = os.path.dirname(image_path)
        metadata_file = os.path.join(folder_path, 'image_metadata.json')
        
        if os.path.exists(metadata_file):
            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Find the image in metadata
            filename = os.path.basename(image_path)
            for img_data in metadata.get('images', []):
                if img_data.get('filename') == filename:
                    return img_data.get('element_info', {}), img_data.get('source_type', 'unknown')
        
    except Exception as e:
        print(f"Warning: Could not read metadata for {image_path}: {e}")
    
    return {}, 'unknown'

def format_element_info(element_info, source_type):
    """Format element information for display"""
    if not element_info:
        return "No element info available"
    
    tag_name = element_info.get('tagName', 'unknown')
    class_name = element_info.get('className', '')
    element_id = element_info.get('id', '')
    alt_text = element_info.get('alt', '')
    title_text = element_info.get('title', '')
    width = element_info.get('width', 0)
    height = element_info.get('height', 0)
    parent_tag = element_info.get('parentTag', '')
    parent_class = element_info.get('parentClass', '')
    parent_id = element_info.get('parentId', '')
    
    # Build element description
    element_desc = f"<{tag_name}"
    if element_id:
        element_desc += f" id=\"{element_id}\""
    if class_name:
        element_desc += f" class=\"{class_name}\""
    if alt_text:
        element_desc += f" alt=\"{alt_text}\""
    if title_text:
        element_desc += f" title=\"{title_text}\""
    element_desc += ">"
    
    # Add parent element info
    if parent_tag:
        parent_desc = f"<{parent_tag}"
        if parent_id:
            parent_desc += f" id=\"{parent_id}\""
        if parent_class:
            parent_desc += f" class=\"{parent_class}\""
        parent_desc += ">"
        element_desc = f"{parent_desc} -> {element_desc}"
    
    # Add dimensions and source type
    if width > 0 and height > 0:
        element_desc += f" ({width}x{height}px)"
    
    element_desc += f" [{source_type}]"
    
    return element_desc

def main():
    parser = argparse.ArgumentParser(description="Free Deep Learning Image Comparison (ResNet50)")
    parser.add_argument('--reference', type=str, required=True, help='Path to reference image')
    parser.add_argument('--folder', type=str, default='downloaded_images', help='Folder to search for similar images')
    parser.add_argument('--topk', type=int, default=10, help='Number of top matches to show')
    args = parser.parse_args()

    print("Loading ResNet50 model (no login required)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove classifier, keep features
    model.eval()
    model.to(device)
    transform = weights.transforms()

    # Load reference image
    print(f"Loading reference image: {args.reference}")
    ref_img = load_image(args.reference)
    ref_emb = get_embedding(model, transform, ref_img, device)

    # Gather dataset images
    print(f"Scanning folder: {args.folder}")
    image_files = get_image_files(args.folder)
    print(f"Found {len(image_files)} images to compare.")
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {args.folder}")
        print("Please make sure:")
        print("1. The download process completed successfully")
        print("2. Images were downloaded to the correct folder")
        print("3. The folder path is correct")
        return

    # Compute embeddings for all images
    similarities = []
    for img_path in image_files:
        try:
            img = load_image(img_path)
            emb = get_embedding(model, transform, img, device)
            sim = np.dot(ref_emb, emb)
            # Extract original URL
            original_url = get_original_url_from_path(img_path)
            # Get element information
            element_info, source_type = get_element_info_from_metadata(img_path)
            similarities.append((img_path, sim, original_url, element_info, source_type))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if len(similarities) == 0:
        print("ERROR: No images could be processed for comparison")
        print("This might be due to:")
        print("1. Corrupted image files")
        print("2. Unsupported image formats")
        print("3. Permission issues")
        return

    # Sort and show top matches
    similarities.sort(key=lambda x: x[1], reverse=True)
    print("\nTop similar images:")
    print("=" * 100)
    for i, (img_path, sim, original_url, element_info, source_type) in enumerate(similarities[:args.topk], 1):
        filename = os.path.basename(img_path)
        element_desc = format_element_info(element_info, source_type)
        
        print(f"{i:2d}. {filename}")
        print(f"    URL: {original_url}")
        print(f"    Element: {element_desc}")
        print(f"    Path: {img_path}")
        print(f"    Similarity: {sim*100:.2f}%")
        print()

if __name__ == "__main__":
    main() 