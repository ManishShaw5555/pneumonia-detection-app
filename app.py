import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from huggingface_hub import hf_hub_download
import os

st.title("Pneumonia Detection from Chest X-rays")

@st.cache_resource
def load_model():
    # Download model weights from Hugging Face
    model_path = hf_hub_download(repo_id="nateraw/resnet18-pneumonia", filename="pytorch_model.bin")
    
    # Load model definition (ResNet18)
    model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Image uploader
uploaded_file = st.file_uploader("Upload a Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred = torch.argmax(probs).item()

    label = "Pneumonia" if pred == 1 else "Normal"
    confidence = probs[pred].item()

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"### Confidence: **{confidence:.2f}**")
