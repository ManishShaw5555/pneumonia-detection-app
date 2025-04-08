import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import torchvision.models as models
import torch.nn as nn

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="nateraw/resnet18-pneumonia", filename="pytorch_model.bin")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

st.title("Pneumonia Detection from Chest X-rays")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
        label = "Pneumonia" if prediction == 1 else "Normal"
        st.subheader(f"Prediction: {label}")
