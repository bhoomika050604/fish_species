import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Fish Species Identification",
    page_icon="🐟",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="fish_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------ LOAD LABELS ------------------
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

IMG_SIZE = 224

# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype('float32')
    return img

# ------------------ UI DESIGN ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #4facfe, #00f2fe);
    color: white;
}
h1, h2, h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🐟 Fish Species Identification System")
st.write("Upload a fish image to identify its species using AI")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        img = preprocess_image(image)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img)

        # Run inference
        interpreter.invoke()

        # Get output
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Prediction: {labels[class_index]}")
        st.info(f"Confidence: {confidence:.2f}%")