from skimage.feature import hog
import numpy as np
from tqdm import tqdm

def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualize=False):
    """
    Extract HOG features from images
    
    Parameters:
    - orientations: Number of orientation bins
    - pixels_per_cell: Size of each cell
    - cells_per_block: Number of cells per block for normalization
    """
    hog_features = []
    hog_images = [] if visualize else None
    
    for img in tqdm(images, desc="Extracting HOG features"):
        # Extract HOG features
        features = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=visualize,
            block_norm='L2-Hys'
        )
        
        hog_features.append(features)
        
        if visualize:
            hog_images.append(features[1])  # The HOG image
    
    return np.array(hog_features), hog_images

def get_feature_info(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Calculate feature vector dimension
    """
    # Calculate feature dimension
    n_cells_x = images[0].shape[1] // pixels_per_cell[0]
    n_cells_y = images[0].shape[0] // pixels_per_cell[0]
    n_blocks_x = n_cells_x - cells_per_block[0] + 1
    n_blocks_y = n_cells_y - cells_per_block[0] + 1
    feature_dim = n_blocks_x * n_blocks_y * cells_per_block[0] * cells_per_block[1] * orientations
    
    return feature_dim