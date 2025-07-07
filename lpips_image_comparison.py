#!/usr/bin/env python3
"""
LPIPS-based Image Comparison for Perceptual Similarity
Uses Learned Perceptual Image Patch Similarity for human-like image comparison
Free, no login required - uses pre-trained AlexNet features
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import lpips
import requests
from io import BytesIO
import time
from torchvision import transforms

def load_image(path):
    """Load image from path or URL with proper headers"""
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
        return Image.open(BytesIO(response.content)).convert("RGB")
    
    return Image.open(path).convert("RGB")

def get_image_files(folder, exts={'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}):
    """Get all image files from folder recursively"""
    files = []
    for root, dirs, fs in os.walk(folder):
        for f in fs:
            if Path(f).suffix.lower() in exts:
                files.append(os.path.join(root, f))
    return files

def preprocess_image(image):
    """Preprocess image for LPIPS"""
    # LPIPS expects images in range [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

def get_original_url_from_path(image_path):
    """Extract the original URL from the image path"""
    try:
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
            
            return f"https://{website_name.replace('_', '.')}"
            
    except Exception as e:
        print(f"Warning: Could not extract URL from path {image_path}: {e}")
    return "Unknown URL"

def get_element_info_from_metadata(image_path):
    """Get HTML element information from metadata file"""
    try:
        folder_path = os.path.dirname(image_path)
        metadata_file = os.path.join(folder_path, 'image_metadata.json')
        
        if os.path.exists(metadata_file):
            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
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
    parser = argparse.ArgumentParser(description="LPIPS-based Perceptual Image Comparison")
    parser.add_argument('--reference', type=str, required=True, help='Path or URL to reference image')
    parser.add_argument('--folder', type=str, default='downloaded_images', help='Folder to search for similar images')
    parser.add_argument('--topk', type=int, default=10, help='Number of top matches to show')
    parser.add_argument('--threshold', type=float, default=0.3, help='LPIPS distance threshold (lower = more similar)')
    args = parser.parse_args()

    print("Loading LPIPS model for perceptual similarity (free, no login)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load LPIPS model
    loss_fn = lpips.LPIPS(net='alex', verbose=False)
    loss_fn.to(device)
    print(f"LPIPS model loaded on {device}")

    # Load reference image
    print(f"Loading reference image: {args.reference}")
    try:
        ref_img = load_image(args.reference)
        ref_tensor = preprocess_image(ref_img).to(device)
        print(f"Reference image loaded successfully")
    except Exception as e:
        print(f"ERROR: Could not load reference image: {e}")
        return

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

    # Compute LPIPS distances for all images
    print("Computing perceptual similarities...")
    similarities = []
    processed = 0
    
    for img_path in image_files:
        try:
            img = load_image(img_path)
            img_tensor = preprocess_image(img).to(device)
            
            # Compute LPIPS distance (lower = more similar)
            with torch.no_grad():
                distance = loss_fn(ref_tensor, img_tensor).item()
            
            # Convert distance to similarity (1 - distance, since LPIPS distance is 0=identical, higher=more different)
            similarity = max(0, 1 - distance)
            
            # Extract metadata
            original_url = get_original_url_from_path(img_path)
            element_info, source_type = get_element_info_from_metadata(img_path)
            
            similarities.append((img_path, similarity, distance, original_url, element_info, source_type))
            processed += 1
            
            if processed % 100 == 0:
                print(f"Processed {processed}/{len(image_files)} images...")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if len(similarities) == 0:
        print("ERROR: No images could be processed for comparison")
        return

    # Sort by similarity (higher similarity first) and filter by threshold
    similarities.sort(key=lambda x: x[1], reverse=True)
    filtered_similarities = [s for s in similarities if s[2] <= (1 - args.threshold)]  # LPIPS distance threshold
    
    print(f"\nLPIPS Perceptual Similarity Results:")
    print(f"Threshold: {args.threshold} (LPIPS distance <= {1-args.threshold:.3f})")
    print(f"Images above threshold: {len(filtered_similarities)}/{len(similarities)}")
    print("=" * 100)
    
    # Show top matches
    top_matches = similarities[:args.topk]
    for i, (img_path, sim, distance, original_url, element_info, source_type) in enumerate(top_matches, 1):
        filename = os.path.basename(img_path)
        element_desc = format_element_info(element_info, source_type)
        
        # Add perceptual interpretation
        if distance <= 0.1:
            perceptual_desc = "VERY SIMILAR (nearly identical perceptually)"
        elif distance <= 0.2:
            perceptual_desc = "SIMILAR (very close perceptual match)"
        elif distance <= 0.3:
            perceptual_desc = "MODERATELY SIMILAR (similar visual perception)"
        elif distance <= 0.5:
            perceptual_desc = "SOMEWHAT SIMILAR (some perceptual similarity)"
        else:
            perceptual_desc = "DISSIMILAR (different visual perception)"
        
        print(f"{i:2d}. {filename}")
        print(f"    URL: {original_url}")
        print(f"    Element: {element_desc}")
        print(f"    Path: {img_path}")
        print(f"    Perceptual Similarity: {sim*100:.2f}%")
        print(f"    LPIPS Distance: {distance:.4f}")
        print(f"    Interpretation: {perceptual_desc}")
        print()

    # Summary statistics
    if filtered_similarities:
        avg_similarity = np.mean([s[1] for s in filtered_similarities])
        avg_distance = np.mean([s[2] for s in filtered_similarities])
        print(f"Average similarity (above threshold): {avg_similarity*100:.2f}%")
        print(f"Average LPIPS distance (above threshold): {avg_distance:.4f}")
    
    print(f"LPIPS perceptual comparison completed successfully!")

if __name__ == "__main__":
    main() 