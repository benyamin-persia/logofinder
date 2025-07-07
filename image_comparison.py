#!/usr/bin/env python3
"""
Image Comparison System - Find images with >95% similarity to a reference image
"""

import os
import cv2
import numpy as np
import requests
from PIL import Image
import io
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from urllib.parse import urlparse
import argparse
import cairosvg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageComparator:
    def __init__(self, similarity_threshold: float = 0.85, debug: bool = False):
        self.similarity_threshold = similarity_threshold
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.svg'}
        self.debug = debug
        
    def load_image_from_file(self, image_path: str) -> np.ndarray:
        """Load image from file path, with SVG support and forced 256x256 size"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            ext = Path(image_path).suffix.lower()
            if ext == '.svg':
                # Convert SVG to PNG in-memory at 256x256
                png_bytes = cairosvg.svg2png(url=image_path, output_width=256, output_height=256)
                image_array = np.frombuffer(png_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not convert SVG to image: {image_path}")
                return image
            # Standard raster image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            return image
        except Exception as e:
            logging.error(f"Error loading image from file {image_path}: {e}")
            raise
    
    def load_image_from_url(self, image_url: str) -> np.ndarray:
        """Load image from URL, with SVG support and forced 256x256 size"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()
            ext = Path(urlparse(image_url).path).suffix.lower()
            if ext == '.svg' or 'image/svg+xml' in response.headers.get('Content-Type', ''):
                # Convert SVG to PNG in-memory at 256x256
                png_bytes = cairosvg.svg2png(bytestring=response.content, output_width=256, output_height=256)
                image_array = np.frombuffer(png_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not convert SVG to image from URL: {image_url}")
                return image
            # Standard raster image
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode image from URL: {image_url}")
            return image
        except Exception as e:
            logging.error(f"Error loading image from URL {image_url}: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for comparison"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize to standard size for comparison
            resized = cv2.resize(gray, (256, 256))
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            raise
    
    def calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity between two images using multiple methods"""
        try:
            # Method 1: Structural Similarity Index (SSIM)
            ssim_score = self.calculate_ssim(img1, img2)
            
            # Method 2: Mean Squared Error (MSE) - convert to similarity
            mse_score = self.calculate_mse_similarity(img1, img2)
            
            # Method 3: Histogram comparison
            hist_score = self.calculate_histogram_similarity(img1, img2)
            
            # Method 4: Feature matching (SIFT)
            feature_score = self.calculate_feature_similarity(img1, img2)
            
            # Combine scores (weighted average)
            combined_score = (ssim_score * 0.4 + mse_score * 0.3 + hist_score * 0.2 + feature_score * 0.1)
            
            return combined_score
            
        except Exception as e:
            logging.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        try:
            # SSIM calculation
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            sigma1 = np.std(img1)
            sigma2 = np.std(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
            return max(0, min(1, ssim))
        except:
            return 0.0
    
    def calculate_mse_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity based on Mean Squared Error"""
        try:
            mse = np.mean((img1 - img2) ** 2)
            # Convert MSE to similarity (lower MSE = higher similarity)
            similarity = 1 / (1 + mse)
            return max(0, min(1, similarity))
        except:
            return 0.0
    
    def calculate_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity based on histogram comparison"""
        try:
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
            
            # Normalize histograms
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            # Calculate correlation
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0, min(1, (correlation + 1) / 2))  # Convert from [-1,1] to [0,1]
        except:
            return 0.0
    
    def calculate_feature_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity based on SIFT feature matching"""
        try:
            # Ensure images are in correct format for SIFT
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1.astype(np.uint8)
                
            if len(img2.shape) == 3:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = img2.astype(np.uint8)
            
            # Initialize SIFT detector
            sift = cv2.SIFT_create()
            
            # Find keypoints and descriptors
            kp1, des1 = sift.detectAndCompute(img1_gray, None)
            kp2, des2 = sift.detectAndCompute(img2_gray, None)
            
            if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
                return 0.0
            
            # Use FLANN matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Calculate similarity based on number of good matches
            max_matches = min(len(kp1), len(kp2))
            if max_matches == 0:
                return 0.0
            
            similarity = len(good_matches) / max_matches
            return min(1.0, similarity)
            
        except Exception as e:
            # Silently return 0.0 for feature matching failures
            return 0.0
    
    def find_similar_images(self, reference_image: np.ndarray, image_folder: str) -> List[Tuple[str, float]]:
        """Find images similar to reference image in the given folder. If debug, print all scores."""
        similar_images = []
        all_scores = []
        try:
            # Preprocess reference image
            ref_processed = self.preprocess_image(reference_image)
            # Get all image files in the folder
            image_files = []
            for root, dirs, files in os.walk(image_folder):
                for file in files:
                    if Path(file).suffix.lower() in self.supported_formats:
                        image_files.append(os.path.join(root, file))
            logging.info(f"Found {len(image_files)} images to compare")
            # Compare with each image
            for i, image_path in enumerate(image_files):
                try:
                    # Load and preprocess image
                    img = self.load_image_from_file(image_path)
                    img_processed = self.preprocess_image(img)
                    # Calculate similarity
                    similarity = self.calculate_similarity(ref_processed, img_processed)
                    all_scores.append((image_path, similarity))
                    # Add to results if above threshold
                    if similarity >= self.similarity_threshold:
                        similar_images.append((image_path, similarity))
                        logging.info(f"Found similar image: {image_path} (similarity: {similarity:.3f})")
                    # Progress update
                    if (i + 1) % 50 == 0:
                        logging.info(f"Processed {i + 1}/{len(image_files)} images")
                except Exception as e:
                    logging.warning(f"Error processing {image_path}: {e}")
                    continue
            # Sort by similarity (highest first)
            similar_images.sort(key=lambda x: x[1], reverse=True)
            if self.debug:
                print("\n--- DEBUG: All similarity scores ---")
                for path, score in sorted(all_scores, key=lambda x: x[1], reverse=True):
                    print(f"{path}: {score*100:.2f}%")
                print("--- END DEBUG ---\n")
            return similar_images
        except Exception as e:
            logging.error(f"Error finding similar images: {e}")
            return []
    
    def get_user_reference_image(self) -> np.ndarray:
        """Get reference image from user input"""
        print("\n" + "=" * 60)
        print("IMAGE COMPARISON SYSTEM")
        print("=" * 60)
        print("Choose how to provide the reference image:")
        print("1. Upload image file")
        print("2. Provide image URL")
        print("3. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                # File upload
                file_path = input("Enter the path to your image file: ").strip()
                if file_path:
                    try:
                        return self.load_image_from_file(file_path)
                    except Exception as e:
                        print(f"❌ Error loading image: {e}")
                        continue
                else:
                    print("❌ Please provide a valid file path")
                    
            elif choice == "2":
                # URL input
                image_url = input("Enter the image URL: ").strip()
                if image_url:
                    try:
                        return self.load_image_from_url(image_url)
                    except Exception as e:
                        print(f"❌ Error loading image from URL: {e}")
                        continue
                else:
                    print("❌ Please provide a valid URL")
                    
            elif choice == "3":
                print("Exiting...")
                exit(0)
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    def display_results(self, similar_images: List[Tuple[str, float]], reference_path: str = "Reference Image"):
        """Display comparison results"""
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"Reference Image: {reference_path}")
        print(f"Similarity Threshold: {self.similarity_threshold * 100:.1f}%")
        print(f"Found {len(similar_images)} similar images")
        print("-" * 60)
        
        if not similar_images:
            print("❌ No images found with similarity above the threshold.")
            return
        
        print("📋 Similar Images (sorted by similarity):")
        print("-" * 60)
        
        for i, (image_path, similarity) in enumerate(similar_images, 1):
            # Get relative path for display
            rel_path = os.path.relpath(image_path, os.getcwd())
            similarity_percent = similarity * 100
            
            print(f"{i:2d}. {rel_path}")
            print(f"    Similarity: {similarity_percent:.2f}%")
            
            # Add visual indicator
            if similarity_percent >= 98:
                indicator = "🟢 Very High"
            elif similarity_percent >= 95:
                indicator = "🟡 High"
            else:
                indicator = "🔴 Medium"
            
            print(f"    Match Level: {indicator}")
            print()
        
        print("=" * 60)
        print("✅ Comparison completed successfully!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Image Comparison System")
    parser.add_argument("--threshold", type=float, default=None, 
                       help="Similarity threshold as a percentage (0-100, default: 85)")
    parser.add_argument("--folder", type=str, default="downloaded_images",
                       help="Folder containing images to compare (default: downloaded_images)")
    parser.add_argument("--reference", type=str, 
                       help="Path to reference image file (optional)")
    parser.add_argument("--url", type=str,
                       help="URL of reference image (optional)")
    
    args = parser.parse_args()
    
    # Get reference image first
    if args.reference:
        reference_image = ImageComparator().load_image_from_file(args.reference)
        reference_path = args.reference
    elif args.url:
        reference_image = ImageComparator().load_image_from_url(args.url)
        reference_path = args.url
    else:
        reference_image = ImageComparator().get_user_reference_image()
        reference_path = "User provided"
    
    # Prompt for threshold if not provided
    threshold = args.threshold
    if threshold is None:
        while True:
            user_input = input("Enter similarity threshold percentage (0-100, default 85): ").strip()
            if user_input == "":
                threshold = 85.0
                break
            try:
                val = float(user_input)
                if 0.0 <= val <= 100.0:
                    threshold = val
                    break
                else:
                    print("❌ Threshold must be between 0 and 100")
            except ValueError:
                print("❌ Please enter a valid number between 0 and 100")
    # Convert to decimal for internal use
    threshold_decimal = threshold / 100.0
    # Prompt for debug mode
    debug = False
    debug_input = input("Enable debug mode? (y/N): ").strip().lower()
    if debug_input == 'y':
        debug = True
    # Initialize comparator
    comparator = ImageComparator(similarity_threshold=threshold_decimal, debug=debug)
    
    try:
        # Check if comparison folder exists
        if not os.path.exists(args.folder):
            print(f"❌ Comparison folder not found: {args.folder}")
            print("Please run the image downloader first to download images.")
            return
        
        # Find similar images
        print(f"\n🔍 Searching for images similar to reference...")
        similar_images = comparator.find_similar_images(reference_image, args.folder)
        
        # Display results
        comparator.display_results(similar_images, reference_path)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 