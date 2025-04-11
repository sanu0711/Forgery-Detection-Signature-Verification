# Forgery Detection - Signature Verification System

![GitHub last commit](https://img.shields.io/github/last-commit/sanu0711/Forgery-Detection-Signature-Verification)
![GitHub repo size](https://img.shields.io/github/repo-size/sanu0711/Forgery-Detection-Signature-Verification)

This project is a **Signature Forgery Detection System** built using **Django** and **TensorFlow/Keras**, which leverages **transfer learning** with pre-trained CNN models including **ResNet50**, **VGG16**, **VGG19**, and a **custom CNN model**. The aim is to classify whether a given signature is *genuine* or *forged*.

---

## 📌 Project Objectives

- Detect forged signatures using deep learning
- Use transfer learning with pre-trained models for efficient training
- Train multiple models for comparison and accuracy benchmarking
- Serve predictions through a Django web interface (optional for deployment)

---

## 🧠 Model Architectures

### 1. 🔧 Custom CNN Model

A standard CNN with multiple convolution and pooling layers followed by dense layers.

- Input size: `150x150`
- Layers: Conv2D → MaxPooling → Dense
- Optimizer: Adam (`lr=0.0001`)
- Loss: Binary Crossentropy
- Metrics: Accuracy
- Epochs: `100` (with early stopping)

---

### 2. 🧠 ResNet50 (Transfer Learning)

- Pre-trained on ImageNet (`include_top=False`)
- Input size: `64x64`
- Pooling: `avg`
- Added Layers: Flatten → Dense(512 → 435 → 365 → 1)
- Trainable Layers: All frozen
- Optimizer: Adam
- Epochs: `100` (early stopping with patience 30)

---

### 3. 🧠 VGG16 (Transfer Learning)

- Pre-trained on ImageNet (`include_top=False`)
- Input size: `64x64`
- Pooling: `avg`
- Added Layers: Flatten → Dense(512 → 450 → 260 → 1)
- Trainable Layers: All frozen
- Optimizer: Adam
- Epochs: `100` (early stopping with patience 30)

---

### 4. 🧠 VGG19 (Transfer Learning)

- Pre-trained on ImageNet (`include_top=False`)
- Input size: `64x64`
- Pooling: `avg`
- Added Layers: Flatten → Dense(512 → 455 → 250 → 1)
- Trainable Layers: All frozen
- Optimizer: Adam
- Epochs: `100` (early stopping with patience 30)

---

## 📂 Dataset

Dataset used: [Signature Verification Dataset by robinreni](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset)

Downloaded using `kagglehub`:

```python
import kagglehub

path = kagglehub.dataset_download("robinreni/signature-verification-dataset")
print("Path to dataset files:", path)
```

## 🔗 Download Pre-trained Models

To get started quickly, you can download the pre-trained `.h5` models from the following Google Drive links:

| **Model Name** | **Architecture**     | **Download Link** |
|----------------|----------------------|-------------------|
| Custom CNN     | Custom-built CNN     | [Download](https://drive.google.com/drive/folders/15Dh7d9g2zpmRf7nnKMXb0toMeJWt38QW?usp=sharing) |
| ResNet50       | Transfer Learning    | [Download](https://drive.google.com/drive/folders/15Dh7d9g2zpmRf7nnKMXb0toMeJWt38QW?usp=sharing) |
| VGG16          | Transfer Learning    | [Download](https://drive.google.com/drive/folders/15Dh7d9g2zpmRf7nnKMXb0toMeJWt38QW?usp=sharing) |
| VGG19          | Transfer Learning    | [Download](https://drive.google.com/drive/folders/15Dh7d9g2zpmRf7nnKMXb0toMeJWt38QW?usp=sharing) |


## 🎥 Demo

![Demo](./static/demo/demo.gif)












