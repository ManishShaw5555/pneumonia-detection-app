import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Recreate the class used during training
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 output classes

    def forward(self, x):
        return self.model(x)

# Load model
@st.cache_resource
def load_model():
    model = PneumoniaCNN()
    model.load_state_dict(torch.load("pneumonia_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # RGB normalization
                         std=[0.229, 0.224, 0.225])
])

# Streamlit app UI
st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image and get a prediction.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        class_names = ["Normal", "Pneumonia"]
        prediction = class_names[predicted.item()]

    st.write(f"### Prediction: **{prediction}**")