# Cats vs Dogs Classification using SVM

A machine learning project that uses Support Vector Machine (SVM) to classify images of cats and dogs.

## 📁 Dataset
- **Source**: Microsoft Cats vs Dogs Dataset
- **Images**: 25,000 total (12,500 cats + 12,500 dogs)
- **Format**: JPG images

## 🛠️ Technologies Used
- Python 3.8+
- scikit-learn (SVM)
- OpenCV (Image processing)
- scikit-image (HOG features)
- NumPy
- Matplotlib/Seaborn (Visualization)



text

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/cats-vs-dogs-svm.git
   cd cats-vs-dogs-svm
Install dependencies

bash

pip install -r requirements.txt
Download dataset

Download from here

Extract and place PetImages folder in data/raw/

Run the project

bash

python main.py

⚙️ Configuration
You can modify these parameters in main.py:

MAX_IMAGES_PER_CLASS: Number of images to use (default: 500)

TARGET_SIZE: Image resize dimensions (default: 128x128)

SVM_C: SVM regularization parameter (default: 1.0)

📊 Results

The model achieves 70-75% accuracy on test data using HOG features + SVM.

Output includes:

Test accuracy

Classification report (precision, recall, F1-score)

Confusion matrix visualization

📝 Requirements

text

numpy

opencv-python

scikit-learn

scikit-image

matplotlib

seaborn

pillow

joblib

tqdm
