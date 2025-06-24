import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio  # Added for handling .mat files
from AnamolyMap import generate_anomaly_map  # Anomaly detection function
from Ground_truth import get_ground_truth  # Ground truth processing function
from ViewImage import visualize_image  # Hyperspectral image visualization function

# --- Page Config ---
st.set_page_config(page_title="Hyperspectral Image Analysis", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to right, #000428, #004E92); /* Dark navy to deep blue */
            color: #FFFFFF; /* White text for readability */
        }
        h1 {
            color: #FFD700; /* Gold for a premium look */
            text-align: center;
            font-weight: bold;
            font-size: 32px;
        }
        p {
            color: #DCDCDC; /* Soft gray for contrast */
            font-size: 18px;
        }
        .stFileUploader {
            background: rgba(255, 255, 255, 0.1); /* Slight transparency */
            border: 3px solid #FFD700; /* Gold border for premium feel */
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0px 0px 15px rgba(255, 215, 0, 0.5); /* Gold glow effect */
        }
        .stButton>button {
            background-color: #004E92; /* Deep blue button */
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #002F5E; /* Slightly darker on hover */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🔬 Hyperspectral Image Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: yellow;'>Detecting anomalies, analyzing ground truth, and visualizing hyperspectral data</p>", unsafe_allow_html=True)
st.markdown("---")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Hyperspectral Image (.mat)", type=["mat"])

if uploaded_file is not None:
    try:
        # Create three sections
        col1, col2, col3 = st.columns(3)

        # ---Anomaly Map ---
        with col1:
            st.header("🔍 Anomaly Map")
            
            # Load the uploaded hyperspectral image instead of a fixed filename
            anomaly_map = generate_anomaly_map(uploaded_file)  
            
            # Generate anomaly map visualization
            fig, ax = plt.subplots()
            ax.imshow(anomaly_map, cmap="hot")
            ax.set_title("Detected Anomalies")
            st.pyplot(fig)

        # --- Ground Truth Map ---
        with col2:
            st.header("✅ Ground Truth")

            ground_truth = get_ground_truth("Indian_pines_gt.mat")  
            
            if ground_truth is not None:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(np.squeeze(ground_truth), cmap="nipy_spectral") 
                ax.set_title("Ground Truth Labels")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning("Ground truth data not found.")

        # ---  Image Visualization ---
        with col3:
            st.header("🖼 Hyperspectral RGB View")

            # Convert uploaded hyperspectral image to RGB
            rgb_image = visualize_image(uploaded_file)  
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(rgb_image)
            ax.axis("off")
            ax.set_title("PCA RGB Image")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.warning("Please upload a hyperspectral image to begin analysis!")

# --- Footer ---
st.markdown("<p style='text-align: center; color: gray;'>An AI Solution by Ankit Pal</p>", unsafe_allow_html=True)
