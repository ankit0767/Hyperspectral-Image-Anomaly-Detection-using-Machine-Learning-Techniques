# Hyperspectral Image Anomaly Detection using an Autoencoder

An interactive web application built with Streamlit to perform unsupervised anomaly detection on hyperspectral images. This project uses a Principal Component Analysis (PCA) and a deep learning Autoencoder model to identify pixels with unusual spectral signatures.

## 📋 Features

-   **Interactive File Upload:** Easily upload your hyperspectral data and ground truth maps (`.mat` files).
-   **Side-by-Side Visualization:** Instantly compare the original data (as a false-color RGB), the ground truth, and the resulting anomaly map in a clean layout.
-   **Unsupervised Anomaly Detection:** Implements a robust machine learning pipeline (PCA + Autoencoder) that requires no prior labeling of anomalies to train.
-   **Modern UI:** A polished and user-friendly web interface designed with custom styling.

---

## 🛠️ Tech Stack

-   **Backend:** Python, TensorFlow/Keras, Scikit-learn, NumPy, SciPy
-   **Frontend:** Streamlit

---

## ⚙️ How It Works

The application implements a standard and effective pipeline for unsupervised anomaly detection:

1.  **Data Loading:** The hyperspectral data cube is loaded from the user-provided `.mat` file.
2.  **Dimensionality Reduction:** PCA is applied to reduce the high number of spectral bands to a more manageable number of components, capturing the most significant variance in the data. This helps combat the curse of dimensionality and reduces computational load.
3.  **Unsupervised Learning:** A deep autoencoder is trained on the reduced data. The model learns to efficiently reconstruct the "normal" background pixels with very low error.
4.  **Anomaly Scoring:** The model's reconstruction error (Mean Squared Error) is calculated for every pixel. Pixels that the model struggles to reconstruct (i.e., those with high error) are considered anomalous because their spectral signature deviates significantly from the learned norm.
5.  **Thresholding:** A percentile-based threshold is applied to the error map to generate the final binary anomaly map, clearly separating anomalies from the background.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

-   Python 3.8 or newer
-   `pip` and `venv`

### 2. Installation

First, clone the repository to your local machine:
```bash
git clone [https://github.com/ankit0767/Hyperspectral-Image-Anomaly-.git](https://github.com/ankit0767/Hyperspectral-Image-Anomaly-.git)
cd Hyperspectral-Image-Anomaly-
