import cv2
import numpy as np
import streamlit as st
from skimage.feature import hog, local_binary_pattern
import joblib  # For loading the saved Random Forest model

def extract_features_from_streamlit(uploaded_file, image_size=32):
    """
    Takes an uploaded image from Streamlit, preprocesses it, and extracts a feature vector.
    The vector includes flattened pixels, LBP histogram, Hu Moments, and HOG features.
    """
    # Load the image from the uploaded file as grayscale
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        st.error("Error: Could not load the image.")
        return None

    # Resize and normalize the image
    image_resized = cv2.resize(image, (image_size, image_size))
    image_normalized = image_resized / 255.0  # Normalize pixel intensities

    # Flatten pixel intensities
    flat_features = image_normalized.flatten()

    # Convert to uint8 for LBP and HOG extraction
    image_uint8 = (image_normalized * 255).astype(np.uint8)

    # Extract Local Binary Pattern (LBP) features
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(image_uint8, n_points, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    lbp_hist_normalized = lbp_hist.astype("float") / lbp_hist.sum()

    # Extract Hu Moments (shape-based features)
    moments = cv2.moments(image_normalized)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Extract HOG features
    hog_features, _ = hog(image_uint8, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)

    # Concatenate all features (pixel intensities, LBP, Hu Moments, and HOG)
    combined_features = np.concatenate([flat_features, lbp_hist_normalized, hu_moments, hog_features])

    # Return the combined feature vector
    return combined_features


# Load your pre-trained Random Forest model (ensure this path points to your model)
model = joblib.load("./Randomforest.pkl")

st.title("Image Classification with Random Forest")
uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Extract features from the uploaded image
    features = extract_features_from_streamlit(uploaded_file)

    if features is not None:
        # Predict the class using the loaded Random Forest model
        predicted_class = model.predict([features])[0]  # Model expects a 2D array (1 sample, n_features)

        # Display the predicted class
        st.success(f"The predicted class is: **{predicted_class}**")
