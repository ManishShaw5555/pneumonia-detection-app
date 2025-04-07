
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🩺 AI-Powered Pneumonia Detection")
st.write("Upload a chest X-ray image to check for signs of pneumonia.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L").resize((150, 150))
    st.image(img, caption="Uploaded X-ray", use_column_width=True)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal"
    st.subheader("Result:")
    st.success(result)
