import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import from_pretrained_keras

# Load model (cache to avoid reloading every time)
@st.cache_resource
def load_model():
    model = from_pretrained_keras("ryefoxlime/PneumoniaDetection")
    return model

model = load_model()

st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to detect pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        st.error(f"Prediction: Pneumonia ({prediction:.2f})")
    else:
        st.success(f"Prediction: Normal ({1 - prediction:.2f})")
