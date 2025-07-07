#!/usr/bin/env python3
"""
Convert all images in a folder (recursively) to PNG, resize to a fixed size, and convert to RGB.
Originals are kept by default. Optionally delete originals with --delete-originals.
Usage:
    python convert_all_to_png.py --folder downloaded_images --size 256 --delete-originals
"""
import os
from PIL import Image
from pathlib import Path
import argparse

def convert_image_to_png(src_path, dst_path, size):
    try:
        img = Image.open(src_path)
        img = img.convert('RGB')  # Drop alpha, ensure RGB
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        img.save(dst_path, 'PNG', optimize=True)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Convert all images in a folder to PNG, resize, and RGB.")
    parser.add_argument('--folder', type=str, default='downloaded_images', help='Folder to scan for images')
    parser.add_argument('--size', type=int, default=256, help='Target size (width and height)')
    parser.add_argument('--delete-originals', action='store_true', help='Delete original files after conversion')
    args = parser.parse_args()

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.svg'}
    count = 0
    failed = 0
    for root, dirs, files in os.walk(args.folder):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext in exts:
                src_path = os.path.join(root, fname)
                dst_name = Path(fname).stem + '.png'
                dst_path = os.path.join(root, dst_name)
                if os.path.exists(dst_path):
                    # Avoid overwriting existing PNGs
                    dst_path = os.path.join(root, Path(fname).stem + '_converted.png')
                ok, err = convert_image_to_png(src_path, dst_path, args.size)
                if ok:
                    count += 1
                    print(f"Converted: {src_path} -> {dst_path}")
                    if args.delete_originals and src_path != dst_path:
                        try:
                            os.remove(src_path)
                            print(f"Deleted original: {src_path}")
                        except Exception as e:
                            print(f"Failed to delete {src_path}: {e}")
                else:
                    failed += 1
                    print(f"Failed to convert {src_path}: {err}")
    print(f"\nDone! Converted {count} images. Failed: {failed}.")
    if args.delete_originals:
        print("Original files deleted.")

if __name__ == "__main__":
    main() 