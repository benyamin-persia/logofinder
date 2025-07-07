#!/usr/bin/env python3
"""
Deep Learning Image Comparison System (CLIP-based)
Finds images most similar to a reference image using OpenAI CLIP
"""
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import argparse
import numpy as np

def load_image(path):
    return Image.open(path).convert("RGB")

def get_image_files(folder, exts={'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.svg'}):
    files = []
    for root, dirs, fs in os.walk(folder):
        for f in fs:
            if Path(f).suffix.lower() in exts:
                files.append(os.path.join(root, f))
    return files

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Image Comparison (CLIP)")
    parser.add_argument('--reference', type=str, required=True, help='Path to reference image')
    parser.add_argument('--folder', type=str, default='downloaded_images', help='Folder to search for similar images')
    parser.add_argument('--topk', type=int, default=10, help='Number of top matches to show')
    args = parser.parse_args()

    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Load reference image
    print(f"Loading reference image: {args.reference}")
    ref_img = load_image(args.reference)
    ref_inputs = processor(images=ref_img, return_tensors="pt").to(device)
    with torch.no_grad():
        ref_emb = model.get_image_features(**ref_inputs)
        ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)

    # Gather dataset images
    print(f"Scanning folder: {args.folder}")
    image_files = get_image_files(args.folder)
    print(f"Found {len(image_files)} images to compare.")

    # Compute embeddings for all images
    similarities = []
    for img_path in image_files:
        try:
            img = load_image(img_path)
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            sim = (ref_emb @ emb.T).item()
            similarities.append((img_path, sim))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Sort and show top matches
    similarities.sort(key=lambda x: x[1], reverse=True)
    print("\nTop similar images:")
    for i, (img_path, sim) in enumerate(similarities[:args.topk], 1):
        print(f"{i:2d}. {img_path} (Similarity: {sim*100:.2f}%)")

if __name__ == "__main__":
    main() 