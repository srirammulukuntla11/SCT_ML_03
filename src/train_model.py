from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'):
    """
    Train SVM classifier
    """
    print(f"Training SVM with {kernel} kernel...")
    print(f"Training data shape: {X_train.shape}")
    
    # Initialize and train SVM
    svm_model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        random_state=42,
        probability=True  # Enable probability estimates
    )
    
    svm_model.fit(X_train, y_train)
    
    print("Training complete!")
    return svm_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get classification report
    class_report = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'])
    
    # Get confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred
    }

def plot_confusion_matrix(conf_matrix, save_path=None):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cat', 'Dog'],
                yticklabels=['Cat', 'Dog'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_model(model, scaler, filepath='models/svm_model.pkl'):
    """
    Save trained model and scaler
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler
    }
    
    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath='models/svm_model.pkl'):
    """
    Load trained model and scaler
    """
    model_data = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model_data['model'], model_data['scaler']