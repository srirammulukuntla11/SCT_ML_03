import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Load image, convert to grayscale, and resize
    """
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    resized = cv2.resize(gray, target_size)
    
    return resized

def load_dataset(data_dir, target_size=(128, 128), max_images_per_class=None):
    """
    Load all images from directory structure
    """
    images = []
    labels = []
    paths = []
    
    categories = ['cats', 'dogs']
    label_map = {'cats': 0, 'dogs': 1}
    
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            continue
            
        image_files = os.listdir(category_dir)
        
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]
        
        for img_file in tqdm(image_files, desc=f"Loading {category}"):
            img_path = os.path.join(category_dir, img_file)
            
            # Load and preprocess image
            img = load_and_preprocess_image(img_path, target_size)
            
            if img is not None:
                images.append(img)
                labels.append(label_map[category])
                paths.append(img_path)
    
    return np.array(images), np.array(labels), paths

def normalize_features(features, scaler=None):
    """
    Normalize features using StandardScaler
    """
    if scaler is None:
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        return features_normalized, scaler
    else:
        features_normalized = scaler.transform(features)
        return features_normalized, scaler