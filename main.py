import os
import numpy as np
from src.utils import create_directory_structure, split_data
from src.data_preprocessing import load_dataset, normalize_features
from src.feature_extraction import extract_hog_features, get_feature_info
from src.train_model import train_svm, evaluate_model, plot_confusion_matrix, save_model
import warnings
warnings.filterwarnings('ignore')

def main():
    # Configuration
    SOURCE_DIR = 'data/raw/PetImages'  # Direct path to PetImages folder
    SPLIT_BASE_PATH = 'data/splits/'
    TARGET_SIZE = (128, 128)
    MAX_IMAGES_PER_CLASS = 500  # Using 500 per class for quick training (you can increase to 5000)
    
    # HOG Parameters 
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)
    
    # SVM Parameters
    SVM_KERNEL = 'rbf'
    SVM_C = 1.0
    
    print("=" * 50)
    print("CATS VS DOGS CLASSIFICATION USING SVM")
    print("=" * 50)
    
    # Step 1: Check if dataset exists
    print("\n[Step 1] Checking dataset...")
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Dataset not found at {SOURCE_DIR}")
        print("Please ensure your PetImages folder is placed at: data/raw/PetImages/")
        return
    
    cat_path = os.path.join(SOURCE_DIR, 'Cat')
    dog_path = os.path.join(SOURCE_DIR, 'Dog')
    
    if not os.path.exists(cat_path) or not os.path.exists(dog_path):
        print("ERROR: Cat or Dog folder not found inside PetImages")
        return
    
    print(f"✅ Dataset found!")
    print(f"   Cat images: {len(os.listdir(cat_path))} files")
    print(f"   Dog images: {len(os.listdir(dog_path))} files")
    
    # Step 2: Create directory structure for splits
    print("\n[Step 2] Creating directory structure for data splits...")
    create_directory_structure(SPLIT_BASE_PATH)
    
    # Step 3: Split data into train/val/test
    print("\n[Step 3] Splitting data...")
    train_dir = os.path.join(SPLIT_BASE_PATH, 'train')
    val_dir = os.path.join(SPLIT_BASE_PATH, 'validation')
    test_dir = os.path.join(SPLIT_BASE_PATH, 'test')
    
    # Check if splitting is needed
    if len(os.listdir(os.path.join(train_dir, 'cats'))) == 0:
        split_data(SOURCE_DIR, train_dir, val_dir, test_dir)
    else:
        print("Data already split, skipping...")
    
    # Step 4: Load and preprocess training data
    print("\n[Step 4] Loading training data...")
    X_train, y_train, train_paths = load_dataset(
        train_dir, 
        target_size=TARGET_SIZE, 
        max_images_per_class=MAX_IMAGES_PER_CLASS
    )
    print(f"✅ Loaded {len(X_train)} training images")
    print(f"   Class distribution: {np.bincount(y_train)}")
    
    # Step 5: Extract HOG features for training
    print("\n[Step 5] Extracting HOG features from training images...")
    feature_dim = get_feature_info([X_train[0]], HOG_ORIENTATIONS, 
                                   HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK)
    print(f"   Feature vector dimension: {feature_dim}")
    
    X_train_features, _ = extract_hog_features(
        X_train,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK
    )
    
    # Step 6: Normalize features
    print("\n[Step 6] Normalizing features...")
    X_train_norm, scaler = normalize_features(X_train_features)
    
    # Step 7: Train SVM model
    print("\n[Step 7] Training SVM model...")
    svm_model = train_svm(
        X_train_norm, 
        y_train, 
        kernel=SVM_KERNEL, 
        C=SVM_C
    )
    
    # Step 8: Load and preprocess test data
    print("\n[Step 8] Loading test data...")
    X_test, y_test, test_paths = load_dataset(
        test_dir, 
        target_size=TARGET_SIZE,
        max_images_per_class=MAX_IMAGES_PER_CLASS // 5  # Smaller test set
    )
    print(f"✅ Loaded {len(X_test)} test images")
    
    # Step 9: Extract HOG features for test
    print("\n[Step 9] Extracting HOG features from test images...")
    X_test_features, _ = extract_hog_features(
        X_test,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK
    )
    
    # Step 10: Normalize test features
    print("\n[Step 10] Normalizing test features...")
    X_test_norm, _ = normalize_features(X_test_features, scaler)
    
    # Step 11: Evaluate model
    print("\n[Step 11] Evaluating model...")
    results = evaluate_model(svm_model, X_test_norm, y_test)
    
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    print(f"🎯 Test Accuracy: {results['accuracy']:.4f}")
    print("\n📋 Classification Report:")
    print(results['classification_report'])
    
    # Step 12: Plot confusion matrix
    print("\n[Step 12] Plotting confusion matrix...")
    plot_confusion_matrix(results['confusion_matrix'], save_path='confusion_matrix.png')
    
    # Step 13: Save model
    print("\n[Step 13] Saving model...")
    save_model(svm_model, scaler, 'models/svm_model.pkl')
    
    print("\n" + "=" * 50)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()