import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_image(mat_file):
    """Converts hyperspectral image from .mat file to RGB using PCA."""
    mat = sio.loadmat(mat_file)
    
    # Auto-detect the 3D hyperspectral variable
    def get_hsi_variable(mat_dict):
        for key in mat_dict:
            if not key.startswith("__") and isinstance(mat_dict[key], np.ndarray) and mat_dict[key].ndim == 3:
                return mat_dict[key]
        raise ValueError("No 3D hyperspectral data found.")

    hsi = get_hsi_variable(mat)

    # Apply PCA to convert HSI to RGB
    h, w, d = hsi.shape
    reshaped = hsi.reshape(-1, d)
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(reshaped).reshape(h, w, 3)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    return rgb  # Returning the processed RGB image
