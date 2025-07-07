#!/usr/bin/env python3
"""
Unified Image Comparison with Consistent Preprocessing
Ensures all images are converted to PNG, resized identically, and normalized consistently
for accurate similarity comparison across all methods
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
import tempfile
import shutil
import concurrent.futures
import sys
import platform

def load_and_preprocess_image(path, target_size=(256, 256), save_as_png=True):
    """Load image and preprocess consistently for all comparison methods"""
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
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to target size with high-quality resampling
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Save as PNG if requested
    if save_as_png:
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file.name, 'PNG', optimize=True)
        temp_file.close()
        return temp_file.name, img
    
    return path, img

def get_image_files(folder, exts={'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}):
    """Get all image files from folder recursively"""
    files = []
    for root, dirs, fs in os.walk(folder):
        for f in fs:
            if Path(f).suffix.lower() in exts:
                files.append(os.path.join(root, f))
    return files

def preprocess_for_resnet(image):
    """Preprocess image for ResNet50"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def preprocess_for_clip(image):
    """Preprocess image for CLIP"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return transform(image).unsqueeze(0)

def preprocess_for_lpips(image):
    """Preprocess image for LPIPS"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
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
    
    if parent_tag:
        parent_desc = f"<{parent_tag}"
        if parent_id:
            parent_desc += f" id=\"{parent_id}\""
        if parent_class:
            parent_desc += f" class=\"{parent_class}\""
        parent_desc += ">"
        element_desc = f"{parent_desc} -> {element_desc}"
    
    if width > 0 and height > 0:
        element_desc += f" ({width}x{height}px)"
    
    element_desc += f" [{source_type}]"
    
    return element_desc

def compare_with_resnet(ref_tensor, img_tensor, model, device):
    """Compare images using ResNet50"""
    with torch.no_grad():
        ref_emb = model(ref_tensor.to(device))
        img_emb = model(img_tensor.to(device))
        
        # Normalize embeddings
        ref_emb = ref_emb / torch.norm(ref_emb)
        img_emb = img_emb / torch.norm(img_emb)
        
        # Compute cosine similarity
        similarity = torch.dot(ref_emb.flatten(), img_emb.flatten()).item()
    
    return similarity

def compare_with_clip(ref_tensor, img_tensor, model, processor, device):
    """Compare images using CLIP"""
    try:
        # Prepare inputs for CLIP
        ref_inputs = processor(images=ref_tensor.squeeze(0), return_tensors="pt", padding=True)
        img_inputs = processor(images=img_tensor.squeeze(0), return_tensors="pt", padding=True)
        
        ref_inputs = {k: v.to(device) for k, v in ref_inputs.items()}
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        
        with torch.no_grad():
            ref_features = model.get_image_features(**ref_inputs)
            img_features = model.get_image_features(**img_inputs)
            
            # Normalize features
            ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = torch.dot(ref_features.flatten(), img_features.flatten()).item()
        
        return similarity
    except Exception as e:
        print(f"CLIP comparison failed: {e}")
        return 0.0

def compare_with_lpips(ref_tensor, img_tensor, loss_fn, device):
    """Compare images using LPIPS"""
    with torch.no_grad():
        distance = loss_fn(ref_tensor.to(device), img_tensor.to(device)).item()
        # Convert distance to similarity (1 - distance)
        similarity = max(0, 1 - distance)
    
    return similarity, distance

def process_image_worker(args_tuple):
    img_path, ref_path, size = args_tuple
    try:
        from PIL import Image
        import numpy as np
        import torch
        from torchvision import transforms
        import lpips
        # Load models inside the process
        from torchvision.models import resnet50, ResNet50_Weights
        from transformers import CLIPProcessor, CLIPModel
        device = "cpu"
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        resnet.eval()
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", token=False)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", token=False, use_fast=True)
        clip_model.eval()
        lpips_model = lpips.LPIPS(net='alex', verbose=False)
        lpips_model.eval()
        # Preprocessing
        def preprocess_for_resnet(image):
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(image).unsqueeze(0)
        def preprocess_for_clip(image):
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                   std=[0.26862954, 0.26130258, 0.27577711])
            ])
            return transform(image).unsqueeze(0)
        def preprocess_for_lpips(image):
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            return transform(image).unsqueeze(0)
        # Load and preprocess reference and candidate
        ref_img = Image.open(ref_path).convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        img = Image.open(img_path).convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        ref_img_np = np.array(ref_img)
        img_np = np.array(img)
        is_pixel_identical = np.array_equal(ref_img_np, img_np)
        similarities = {}
        # ResNet50
        ref_tensor = preprocess_for_resnet(ref_img)
        img_tensor = preprocess_for_resnet(img)
        with torch.no_grad():
            ref_emb = resnet(ref_tensor)
            img_emb = resnet(img_tensor)
            ref_emb = ref_emb / torch.norm(ref_emb)
            img_emb = img_emb / torch.norm(img_emb)
            similarities['resnet'] = torch.dot(ref_emb.flatten(), img_emb.flatten()).item()
        # CLIP
        ref_tensor_clip = preprocess_for_clip(ref_img)
        img_tensor_clip = preprocess_for_clip(img)
        ref_inputs = clip_processor(images=ref_tensor_clip.squeeze(0), return_tensors="pt", padding=True)
        img_inputs = clip_processor(images=img_tensor_clip.squeeze(0), return_tensors="pt", padding=True)
        with torch.no_grad():
            ref_features = clip_model.get_image_features(**ref_inputs)
            img_features = clip_model.get_image_features(**img_inputs)
            ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            similarities['clip'] = torch.dot(ref_features.flatten(), img_features.flatten()).item()
        # LPIPS
        ref_tensor_lpips = preprocess_for_lpips(ref_img)
        img_tensor_lpips = preprocess_for_lpips(img)
        with torch.no_grad():
            distance = lpips_model(ref_tensor_lpips, img_tensor_lpips).item()
            similarities['lpips'] = max(0, 1 - distance)
            similarities['lpips_distance'] = distance
        # Metadata (not available in worker, so just return path)
        return (img_path, similarities, is_pixel_identical)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def run_comparison(args, image_files):
    print("Unified Image Comparison with Consistent Preprocessing")
    print("=" * 60)
    print(f"Target size: {args.size}x{args.size} pixels")
    print(f"Format: PNG with consistent normalization")
    print(f"Method: {args.method}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    if args.method in ['resnet', 'all']:
        print("Loading ResNet50 model...")
        from torchvision.models import resnet50, ResNet50_Weights
        models['resnet'] = resnet50(weights=ResNet50_Weights.DEFAULT)
        models['resnet'] = torch.nn.Sequential(*(list(models['resnet'].children())[:-1]))
        models['resnet'].to(device)
        models['resnet'].eval()
    if args.method in ['clip', 'all']:
        print("Loading CLIP model...")
        try:
            from transformers import CLIPProcessor, CLIPModel
            models['clip_model'] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", token=False)
            models['clip_processor'] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", token=False, use_fast=True)
            models['clip_model'].to(device)
            models['clip_model'].eval()
        except Exception as e:
            print(f"Could not load CLIP: {e}")
            if args.method == 'clip':
                return
    if args.method in ['lpips', 'all']:
        print("Loading LPIPS model...")
        import lpips
        models['lpips'] = lpips.LPIPS(net='alex', verbose=False)
        models['lpips'].to(device)
    print(f"Loading and preprocessing reference image: {args.reference}")
    try:
        ref_path, ref_img = load_and_preprocess_image(args.reference, (args.size, args.size))
        print(f"Reference image preprocessed: {ref_img.size}")
    except Exception as e:
        print(f"ERROR: Could not load reference image: {e}")
        return

    print("Computing similarities with consistent preprocessing (multi-processing)...")
    results = []
    pixel_identical = None
    max_workers = min(8, os.cpu_count() or 4)
    args_list = [(img_path, args.reference, args.size) for img_path in image_files]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, result in enumerate(executor.map(process_image_worker, args_list), 1):
            if result is not None:
                img_path, similarities, is_pixel_identical = result
                if is_pixel_identical:
                    pixel_identical = (img_path, None)
                # Metadata extraction in main process
                original_url = get_original_url_from_path(img_path)
                element_info, source_type = get_element_info_from_metadata(img_path)
                results.append((img_path, similarities, original_url, element_info, source_type))
            if i % 50 == 0:
                print(f"Processed {i}/{len(image_files)} images...")

    if len(results) == 0:
        print("ERROR: No images could be processed for comparison")
        return

    # If a pixel-identical image was found, force it to the top with 100% similarity
    if pixel_identical is not None:
        img_path, _ = pixel_identical
        for i, result in enumerate(results):
            if result[0] == img_path:
                similarities = result[1]
                similarities['resnet'] = 1.0
                similarities['clip'] = 1.0
                similarities['lpips'] = 1.0
                similarities['lpips_distance'] = 0.0
                # Move to top
                results.insert(0, results.pop(i))
                break

    # Display results
    print(f"\nUnified Comparison Results:")
    print(f"Images processed: {len(results)}")
    print("=" * 100)
    
    # Sort by the selected method or average of all methods
    if args.method == 'all':
        # Sort by average similarity across all methods
        for i, result in enumerate(results):
            similarities = result[1]
            valid_sims = [v for k, v in similarities.items() if k != 'lpips_distance']
            result = list(result)
            result.append(np.mean(valid_sims))
            results[i] = result
        results.sort(key=lambda x: x[-1], reverse=True)
    else:
        # Sort by the selected method
        results.sort(key=lambda x: x[1].get(args.method, 0), reverse=True)
    
    # Show top matches with raw scores and strict threshold
    top_matches = results[:args.topk]
    strict_threshold = 0.99
    for i, result in enumerate(top_matches, 1):
        if len(result) == 6:  # Has average score
            img_path, similarities, original_url, element_info, source_type, avg_score = result
        else:
            img_path, similarities, original_url, element_info, source_type = result
            avg_score = None
        filename = os.path.basename(img_path)
        element_desc = format_element_info(element_info, source_type)
        print(f"{i:2d}. {filename}")
        print(f"    URL: {original_url}")
        print(f"    Element: {element_desc}")
        print(f"    Path: {img_path}")
        # Print raw similarity scores
        print(f"    ResNet50: {similarities.get('resnet', 0)*100:.2f}%")
        print(f"    CLIP: {similarities.get('clip', 0)*100:.2f}%")
        print(f"    LPIPS: {similarities.get('lpips', 0)*100:.2f}% (distance: {similarities.get('lpips_distance', 0):.4f})")
        if avg_score is not None:
            print(f"    Average: {avg_score*100:.2f}%")
        # Strict threshold reporting
        if (similarities.get('resnet', 0) >= strict_threshold and
            similarities.get('clip', 0) >= strict_threshold and
            similarities.get('lpips', 0) >= strict_threshold):
            print("    >>> IDENTICAL (all methods above 99%) <<<")
        print()

    print("Unified comparison completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Unified Image Comparison with Consistent Preprocessing")
    parser.add_argument('--reference', type=str, required=True, help='Path or URL to reference image')
    parser.add_argument('--folder', type=str, default='downloaded_images', help='Folder to search for similar images')
    parser.add_argument('--topk', type=int, default=10, help='Number of top matches to show')
    parser.add_argument('--method', type=str, choices=['resnet', 'clip', 'lpips', 'all'], default='all', help='Comparison method to use')
    parser.add_argument('--size', type=int, default=512, help='Target image size for preprocessing (default: 512)')
    parser.add_argument('--single-process', action='store_true', help='Force single-process mode (for subprocess safety)')
    args = parser.parse_args()

    print(f"Scanning folder: {args.folder}")
    image_files = get_image_files(args.folder)
    print(f"Found {len(image_files)} images to compare.")
    if len(image_files) == 0:
        print(f"ERROR: No images found in {args.folder}")
        return

    if args.single_process or (platform.system() == 'Windows' and getattr(sys, 'frozen', False)):
        print("Running in single-process mode for compatibility.")
        results = []
        pixel_identical = None
        for i, img_path in enumerate(image_files, 1):
            result = process_image_worker((img_path, args.reference, args.size))
            if result is not None:
                img_path, similarities, is_pixel_identical = result
                if is_pixel_identical:
                    pixel_identical = (img_path, None)
                original_url = get_original_url_from_path(img_path)
                element_info, source_type = get_element_info_from_metadata(img_path)
                results.append((img_path, similarities, original_url, element_info, source_type))
            if i % 50 == 0:
                print(f"Processed {i}/{len(image_files)} images...")
        # Continue with result sorting and reporting as before
        # ... (reuse the result sorting/reporting code from run_comparison)
    else:
        # Use multi-processing as before
        max_workers = min(8, os.cpu_count() or 4)
        args_list = [(img_path, args.reference, args.size) for img_path in image_files]
        results = []
        pixel_identical = None
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, result in enumerate(executor.map(process_image_worker, args_list), 1):
                if result is not None:
                    img_path, similarities, is_pixel_identical = result
                    if is_pixel_identical:
                        pixel_identical = (img_path, None)
                    original_url = get_original_url_from_path(img_path)
                    element_info, source_type = get_element_info_from_metadata(img_path)
                    results.append((img_path, similarities, original_url, element_info, source_type))
                if i % 50 == 0:
                    print(f"Processed {i}/{len(image_files)} images...")
        # Continue with result sorting and reporting as before
        # ... (reuse the result sorting/reporting code from run_comparison)

if __name__ == "__main__":
    main() 