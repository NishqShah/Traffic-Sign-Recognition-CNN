# Traffic Sign Recognition using CNNs

This project implements a Convolutional Neural Network (CNN) to classify German traffic signs. The model is trained on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://zenodo.org/records/13741936) dataset and achieves a test accuracy of around **97%**.  

The project was developed in **Google Colab** and the full workflow is available in a single `traffic_sign_recognition.ipynb` notebook that can be directly run on Colab.

---

## About the Project

The goal of this project was to build a deep learning model that can accurately classify 43 different classes of German traffic signs.  

We focused not just on training a performant CNN, but also on understanding practical preprocessing techniques (grayscale conversion, CLAHE, normalization), model regularization strategies (dropout, early stopping), and evaluation using confusion matrices and classification reports.

---

## What's Inside

- `traffic_sign_recognition.ipynb`: The main Colab notebook containing all steps — from data loading, preprocessing, model building, training, to evaluation.
- `traffic_sign_model.h5`: The final trained model saved for reuse.

---

## Key Features

- Preprocessing pipeline with grayscale conversion, CLAHE (contrast enhancement), and normalization  
- CNN model built with TensorFlow/Keras  
- Dropout and EarlyStopping to prevent overfitting  
- Evaluation with classification report and confusion matrix heatmap  
- Achieved ~97% accuracy on the test set  

---

## EDA Highlights

Before training the CNN, we explored and preprocessed the dataset:

- Checked class distribution across 43 traffic sign categories  
- Visualized random samples for inspection  
- Converted images to grayscale to reduce complexity  
- Applied **CLAHE** for contrast enhancement in poor lighting  
- Normalized pixel values from [0–255] to [0–1]  

These preprocessing steps ensured consistency in training data and improved generalization.

---

## Model Architecture

The CNN architecture used is simple but effective:

- **Input Layer**: 30×30 grayscale images (1 channel)  
- **Conv2D + MaxPooling**: 32 filters → 64 filters → 128 filters  
- **Flatten**: Convert 3D feature maps into 1D vector  
- **Dense Layer**: 128 neurons with ReLU  
- **Dropout Layer**: 0.6 to reduce overfitting  
- **Output Layer**: 43 neurons with Softmax activation  

This design balances performance and training efficiency, achieving strong accuracy without being overly complex.

---

## Training Setup

- **Loss Function**: Sparse Categorical Cross-Entropy  
- **Optimizer**: Adam  
- **Epochs**: Up to 100 (with EarlyStopping patience of 5)  
- **Batch Size**: 64  
- **Callbacks**: EarlyStopping with `restore_best_weights=True`  

---

## Performance

- **Final Test Accuracy**: ~97%  
- Confusion matrix shows most signs are classified correctly, with very few misclassifications.  
- Classification report highlights high precision, recall, and F1-scores across classes.

---

## Notes & Decisions

- **Data Augmentation**:  
  We experimented with augmentation (rotations, shifts, zooms). However, since preprocessing (grayscale + CLAHE + normalization) already improved clarity, augmentation provided little accuracy gain and introduced complexity.We decided to keep the pipeline simple.  
  *(If needed in the future, augmentation can be applied before preprocessing via a saved pipeline using joblib.)*
  
---

## What I Learned

- The importance of preprocessing for image-based models (grayscale + CLAHE improved results)  
- How dropout and early stopping stabilize training  
- Trade-offs between augmentation complexity and actual accuracy improvements  
- How to evaluate models beyond accuracy using confusion matrices and reports  

---

## Files in This Repository

### `traffic_sign_recognition.ipynb`  
The main Google Colab notebook with the entire workflow. Can be opened and run directly in Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NishqShah/Traffic-Sign-Recognition-CNN/blob/main/traffic_sign_recognition.ipynb)

### `traffic_sign_model.h5`  
The trained CNN model saved in HDF5 format.

---

## References

- Dataset: [GTSRB on Zenodo](https://zenodo.org/records/13741936)  
- Blog Reference: [Traffic Signs Classification — navoshta.com](https://navoshta.com/traffic-signs-classification/)  

---
