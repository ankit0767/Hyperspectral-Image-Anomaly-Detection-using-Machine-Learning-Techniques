import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def get_ground_truth(mat_file):
    """Loads and processes ground truth from a .mat file."""
    data = sio.loadmat(mat_file)
    
    # Auto-detect ground truth variable
    def find_gt_variable(mat_dict):
        for key in mat_dict:
            if not key.startswith("__") and isinstance(mat_dict[key], np.ndarray) and mat_dict[key].ndim == 2:
                return mat_dict[key]
        raise ValueError("No 2D ground truth data found.")

    ground_truth = find_gt_variable(data)
    
    return ground_truth  # Returning the processed ground truth array


