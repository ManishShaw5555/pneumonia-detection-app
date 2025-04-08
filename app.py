import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import from_pretrained_keras

# Page config
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🩻 Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to detect pneumonia using a pre-trained model from Hugging Face.")

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model = from_pretrained_keras("ryefoxlime/PneumoniaDetection")
    return model

model = load_model()

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.convert("L")
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# File uploader
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing the image..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0][0]

            if prediction > 0.5:
                st.error(f"🛑 Pneumonia Detected (Confidence: {prediction:.2f})")
            else:
                st.success(f"✅ Normal (Confidence: {1 - prediction:.2f})")
