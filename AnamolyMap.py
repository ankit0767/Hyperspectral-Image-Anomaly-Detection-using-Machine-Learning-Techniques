from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models

def generate_anomaly_map(mat_file):
    """Processes HSI and returns anomaly map."""
    hsi_data = loadmat(mat_file)
    
    # Auto-detect 3D hyperspectral variable
    def get_hsi_variable(mat_dict):
        for key in mat_dict:
            if not key.startswith("__") and isinstance(mat_dict[key], np.ndarray) and mat_dict[key].ndim == 3:
                return mat_dict[key]
        raise ValueError("No 3D hyperspectral data found.")

    hsi = get_hsi_variable(hsi_data)

    # Normalize
    hsi_min = hsi.min(axis=(0, 1), keepdims=True)
    hsi_max = hsi.max(axis=(0, 1), keepdims=True)
    hsi_norm = (hsi - hsi_min) / (hsi_max - hsi_min + 1e-6)

    # Apply PCA for dimensionality reduction
    h, w, bands = hsi_norm.shape
    hsi_reshaped = hsi_norm.reshape(-1, bands)
    pca = PCA(n_components=30)
    hsi_denoised = pca.fit_transform(hsi_reshaped)

    # Define Autoencoder model
    def build_autoencoder(input_shape):
        model = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_shape[0], activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Train Autoencoder
    input_shape = (30,)
    autoencoder = build_autoencoder(input_shape)
    X_train = hsi_denoised.reshape(-1, 30)
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_split=0.2)

    # Compute reconstruction error
    reconstructed = autoencoder.predict(X_train)
    reconstruction_error = np.mean((X_train - reconstructed) ** 2, axis=1)
    threshold = np.percentile(reconstruction_error, 95)
    anomalies = reconstruction_error > threshold

    return anomalies.reshape(h, w)  # Return anomaly map
