import cv2
import numpy as np
import streamlit as st
from skimage.feature import hog, local_binary_pattern
import joblib
from PIL import Image
import io
import onnxruntime as ort
from rembg import remove
import torchvision.transforms as transforms

# Load both models
rf_model = joblib.load("./Randomforest.pkl")  # Random Forest
classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
          'Gall Midge', 'Powdery Mildew', 'Sooty Mould']  # DenseNet classes
ort_session = ort.InferenceSession('densenet.onnx')  # ONNX model

# Define transformations for DenseNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(file_bytes, image_size=32):
    """Feature extraction for Random Forest"""
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        st.error("Error: Could not load the image.")
        return None

    # Preprocessing and feature extraction
    image_resized = cv2.resize(image, (image_size, image_size))
    image_normalized = image_resized / 255.0
    image_uint8 = (image_normalized * 255).astype(np.uint8)
    
    # Feature extraction (same as original)
    flat_features = image_normalized.flatten()
    lbp = local_binary_pattern(image_uint8, 16, 2, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=int(lbp.max() + 1))
    lbp_hist_normalized = lbp_hist.astype("float") / lbp_hist.sum()
    moments = cv2.moments(image_normalized)
    hu_moments = cv2.HuMoments(moments).flatten()
    hog_features, _ = hog(image_uint8, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    
    return np.concatenate([flat_features, lbp_hist_normalized, hu_moments, hog_features])

def enhance_image(input_image):
    """Image enhancement for DenseNet"""
    output_image = remove(input_image, alpha_matting=True)
    black_background = Image.new('RGBA', input_image.size, (0, 0, 0, 255))
    composite_image = Image.alpha_composite(black_background, output_image)
    
    # Image processing
    composite_np = np.array(composite_image)
    bgr = cv2.cvtColor(composite_np, cv2.COLOR_RGBA2BGR)
    hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(20, 20))
    v = clahe.apply(v)
    hsv_img = cv2.merge([h, s, v])
    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    gamma_corrected = np.power(rgb / 255.0, 1.5)
    return Image.fromarray(np.uint8(gamma_corrected * 255))

def preprocess(image):
    """Preprocessing for DenseNet"""
    image = transform(image)
    return image.unsqueeze(0).numpy()

# Streamlit interface
st.title("Hybrid Disease Detection System")
uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    
    # First stage: Random Forest
    with st.spinner("Initial screening..."):
        features = extract_features(file_bytes)
        if features is not None:
            rf_prediction = rf_model.predict([features])[0]
            print(rf_model.classes_)  # Should show actual class names
            print(rf_model.n_features_in_)  # Should match len(features)
            st.success(f"Initial Screening Result: {rf_prediction}")

            # Second stage: Deep Learning if needed
            if "diseased" in rf_prediction.lower():
                st.subheader("Detailed Analysis")
                
                with st.spinner("Enhancing image..."):
                    image_pil = Image.open(io.BytesIO(file_bytes))
                    enhanced_img = enhance_image(image_pil)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image_pil, caption="Original Image", use_column_width=True)
                    with col2:
                        st.image(enhanced_img, caption="Enhanced Image", use_column_width=True)
                
                with st.spinner("Running deep analysis..."):
                    input_tensor = preprocess(enhanced_img)
                    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
                    ort_outs = ort_session.run(None, ort_inputs)
                    final_pred = classes[np.argmax(ort_outs[0])]
                    
                    st.success(f"Detailed Diagnosis: {final_pred}")
                    st.markdown("**Recommended Actions:** Consult with agricultural expert for targeted treatment.")