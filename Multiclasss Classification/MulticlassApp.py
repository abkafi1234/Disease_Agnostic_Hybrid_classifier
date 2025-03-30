# import streamlit as st
# import onnxruntime as ort
# from PIL import Image
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from rembg import remove

# import torchvision.transforms as transforms

# def enhance_image(image):
#   # Load the input image
#   input_image = Image.open(image)
#   plt.imshow(input_image)

#   # remove background
#   output_image = remove(input_image, alpha_matting=True)

#   # Create a black background of the same size as the input image
#   black_background = Image.new('RGB', input_image.size, (0, 0, 0))

#   if black_background.size != output_image.size:
#     # Resize one of the images to match the dimensions of the other
#     black_background = black_background.resize(output_image.size)

#   if black_background.mode != output_image.mode:
#     # Convert one of the images to the same mode as the other
#     black_background = black_background.convert(output_image.mode)

#   # Composite the foreground object onto the black background
#   composite_image = Image.alpha_composite(black_background.convert('RGBA'), output_image)

#   # Convert the PIL Image to a NumPy array
#   composite_np = np.array(composite_image)

#     # Convert to BGR (OpenCV format)
#   bgr = cv2.cvtColor(composite_np, cv2.COLOR_RGBA2BGR)

#   hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#   h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]

#   clahe = cv2.createCLAHE(clipLimit = 10.0, tileGridSize = (20,20))
#   v = clahe.apply(v)
#   hsv_img = np.dstack((h,s,v))
#   rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)


#   bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

#   gamma = 1.5

#   # Perform gamma correction
#   gamma_corrected = np.power(bgr / 255.0, gamma)
  
#   gamma_corrected = np.uint8(gamma_corrected * 255)
#   final_output = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)

#   return final_output

# # Load the ONNX model
# classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Powdery Mildew', 'Sooty Mould']
# onnx_model_path = 'densenet.onnx'
# ort_session = ort.InferenceSession(onnx_model_path)

# # Define the transformations for the images
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Function to preprocess the image
# def preprocess(image):
#     image = transform(image)
#     image = image.unsqueeze(0)  # Add batch dimension
#     return image.numpy()

# # Streamlit app
# st.title("Image Classification with DenseNet ONNX Model")
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")
#     st.write("Classifying...")

#     # Preprocess the image
#     input_image = preprocess(enhance_image(image))

#     # Run the model
#     ort_inputs = {ort_session.get_inputs()[0].name: input_image}
#     ort_outs = ort_session.run(None, ort_inputs)

#     # Get the predicted class
#     pred_class = np.argmax(ort_outs[0], axis=1)[0]
#     st.write(pred_class)
#     # Display the result
#     st.write(f"Predicted Class: {classes[pred_class]}")


import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np
import cv2
from rembg import remove
import torchvision.transforms as transforms

def enhance_image(input_image):
    # Remove background
    output_image = remove(input_image, alpha_matting=True)

    # Create a black background of the same size as the input image
    black_background = Image.new('RGBA', input_image.size, (0, 0, 0, 255))

    # Ensure both images have the same mode and size
    if black_background.size != output_image.size:
        black_background = black_background.resize(output_image.size)

    # Composite the foreground object onto the black background
    composite_image = Image.alpha_composite(black_background, output_image)

    # Convert the PIL Image to a NumPy array and apply enhancements
    composite_np = np.array(composite_image)
    bgr = cv2.cvtColor(composite_np, cv2.COLOR_RGBA2BGR)

    # HSV and CLAHE for brightness and contrast enhancement
    hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(20, 20))
    v = clahe.apply(v)
    hsv_img = np.dstack((h, s, v))
    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    # Gamma correction for further brightness enhancement
    gamma = 1.5
    gamma_corrected = np.power(rgb / 255.0, gamma)
    final_output = np.uint8(gamma_corrected * 255)

    return Image.fromarray(final_output)  # Return the processed image as a PIL Image
# Function to preprocess the image
def preprocess(image):
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.numpy()

# Define the transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ONNX model
classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Powdery Mildew', 'Sooty Mould']
onnx_model_path = 'densenet.onnx'
ort_session = ort.InferenceSession(onnx_model_path)





# Streamlit app
st.title("Image Classification with DenseNet ONNX Model")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Classifying...")

    # Enhance the image and display the processed version
    enhanced_image = enhance_image(image)
    st.image(enhanced_image, caption='Enhanced Image', use_container_width=True)

    # Preprocess the enhanced image
    input_image = preprocess(enhanced_image)

    # Run the model
    ort_inputs = {ort_session.get_inputs()[0].name: input_image}
    ort_outs = ort_session.run(None, ort_inputs)

    # Get the predicted class
    pred_class = np.argmax(ort_outs[0], axis=1)[0]

    # Display the prediction
    st.write(f"Predicted Class: {classes[pred_class]}")
