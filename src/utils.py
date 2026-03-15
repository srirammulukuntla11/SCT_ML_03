import os
import numpy as np
from tqdm import tqdm
import shutil
import random

def create_directory_structure(base_dir):
    """
    Create directory structure for train/val/test splits
    """
    dirs = [
        os.path.join(base_dir, 'train', 'cats'),
        os.path.join(base_dir, 'train', 'dogs'),
        os.path.join(base_dir, 'validation', 'cats'),
        os.path.join(base_dir, 'validation', 'dogs'),
        os.path.join(base_dir, 'test', 'cats'),
        os.path.join(base_dir, 'test', 'dogs')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return base_dir

def split_data(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train, validation, and test sets
    
    Parameters:
    - source_dir: Path to PetImages folder containing Cat/ and Dog/ subfolders
    """
    test_ratio = 1 - train_ratio - val_ratio
    
    # Handle both 'Cat'/'Dog' (original) and 'cat'/'dog' (lowercase) folder names
    categories = []
    if os.path.exists(os.path.join(source_dir, 'Cat')):
        categories.append(('Cat', 'cats'))
    if os.path.exists(os.path.join(source_dir, 'Dog')):
        categories.append(('Dog', 'dogs'))
    if os.path.exists(os.path.join(source_dir, 'cat')):
        categories.append(('cat', 'cats'))
    if os.path.exists(os.path.join(source_dir, 'dog')):
        categories.append(('dog', 'dogs'))
    
    for source_category, dest_category in categories:
        source_category_dir = os.path.join(source_dir, source_category)
        
        # Get all image files
        images = [f for f in os.listdir(source_category_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
        
        # Filter out zero-length files
        valid_images = []
        corrupted_count = 0
        
        for img in tqdm(images, desc=f"Checking {source_category} images"):
            img_path = os.path.join(source_category_dir, img)
            if os.path.getsize(img_path) > 0:
                valid_images.append(img)
            else:
                corrupted_count += 1
        
        print(f"   Found {len(valid_images)} valid images, {corrupted_count} corrupted/skipped")
        
        # Shuffle images
        random.shuffle(valid_images)
        
        # Calculate split indices
        n_train = int(len(valid_images) * train_ratio)
        n_val = int(len(valid_images) * val_ratio)
        
        train_images = valid_images[:n_train]
        val_images = valid_images[n_train:n_train + n_val]
        test_images = valid_images[n_train + n_val:]
        
        print(f"   Splitting: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
        
        # Copy images to respective directories
        for img in tqdm(train_images, desc=f"Copying {dest_category} training"):
            src = os.path.join(source_category_dir, img)
            dst = os.path.join(train_dir, dest_category, img)
            shutil.copy2(src, dst)
        
        for img in tqdm(val_images, desc=f"Copying {dest_category} validation"):
            src = os.path.join(source_category_dir, img)
            dst = os.path.join(val_dir, dest_category, img)
            shutil.copy2(src, dst)
        
        for img in tqdm(test_images, desc=f"Copying {dest_category} test"):
            src = os.path.join(source_category_dir, img)
            dst = os.path.join(test_dir, dest_category, img)
            shutil.copy2(src, dst)
    
    # Count final distribution
    print("\n📊 Final dataset distribution:")
    for split in ['train', 'validation', 'test']:
        n_cats = len(os.listdir(os.path.join(train_dir.replace('train', split), 'cats'))) if split != 'train' else len(os.listdir(os.path.join(train_dir, 'cats')))
        n_dogs = len(os.listdir(os.path.join(train_dir.replace('train', split), 'dogs'))) if split != 'train' else len(os.listdir(os.path.join(train_dir, 'dogs')))
        print(f"   {split.capitalize()}: {n_cats} cats, {n_dogs} dogs")