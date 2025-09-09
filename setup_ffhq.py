#!/usr/bin/env python3
"""
Script to set up FFHQ dataset for gradient inversion attacks.
Downloads images from Hugging Face and organizes them with JSON metadata.
"""

import os
import json
from datasets import load_dataset
from PIL import Image
import shutil

def setup_ffhq_dataset(output_dir="ffhq_dataset", sample_size=1000):
    """
    Set up FFHQ dataset in the format expected by the gradient inversion repository.
    
    Args:
        output_dir: Directory to save the organized dataset
        sample_size: Number of images to download (None for all 70k)
    """
    print("🚀 Setting up FFHQ dataset...")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(images_dir, exist_ok=True)
    
    # Copy JSON metadata
    print("📁 Copying JSON metadata...")
    if os.path.exists("ffhq-features-dataset/json"):
        if os.path.exists(json_dir):
            shutil.rmtree(json_dir)
        shutil.copytree("ffhq-features-dataset/json", json_dir)
        print(f"✅ Copied JSON metadata to {json_dir}")
    else:
        print("❌ JSON metadata not found. Make sure ffhq-features-dataset is cloned.")
        return False
    
    # Download and save images
    print("📥 Loading FFHQ images from Hugging Face...")
    try:
        dataset = load_dataset("nuwandaa/ffhq128", split="train")
        
        # Determine how many images to process
        total_images = len(dataset)
        if sample_size is None:
            sample_size = total_images
        else:
            sample_size = min(sample_size, total_images)
        
        print(f"💾 Processing {sample_size} out of {total_images} images...")
        
        # Save images with proper naming
        for i in range(sample_size):
            if i % 1000 == 0:
                print(f"  Processed {i}/{sample_size} images...")
            
            # Get image from dataset
            image = dataset[i]['image']
            
            # Save with zero-padded filename (00000.png, 00001.png, etc.)
            filename = f"{i:05d}.png"
            image_path = os.path.join(images_dir, filename)
            image.save(image_path)
        
        print(f"✅ Saved {sample_size} images to {images_dir}")
        
        # Verify setup
        print("\n🔍 Verifying setup...")
        
        # Check if we have matching JSON files
        sample_json = os.path.join(json_dir, "00000.json")
        sample_image = os.path.join(images_dir, "00000.png")
        
        if os.path.exists(sample_json) and os.path.exists(sample_image):
            # Show sample age label
            with open(sample_json, 'r') as f:
                data = json.load(f)
                # JSON contains a list with one dictionary
                age = data[0]['faceAttributes']['age']
                print(f"✅ Sample image 00000.png has age: {age}")
        
        print(f"\n🎉 FFHQ dataset setup complete!")
        print(f"📂 Dataset location: {os.path.abspath(output_dir)}")
        print(f"🖼️  Images: {sample_size} files in {images_dir}")
        print(f"📋 Metadata: JSON files in {json_dir}")
        print(f"\n💡 Use this path in your scripts: --data_path {os.path.abspath(output_dir)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up FFHQ dataset for gradient inversion")
    parser.add_argument("--output_dir", default="ffhq_dataset", help="Output directory")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of images (None for all)")
    parser.add_argument("--full", action="store_true", help="Download all 70k images")
    
    args = parser.parse_args()
    
    sample_size = None if args.full else args.sample_size
    
    success = setup_ffhq_dataset(args.output_dir, sample_size)
    
    if success:
        print("\n🚀 Ready to run gradient inversion attacks with FFHQ!")
    else:
        print("\n❌ Setup failed. Please check the error messages above.") 