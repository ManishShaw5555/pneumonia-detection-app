import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import json

# Load model (adjusted to how it was trained)
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)  # 2 classes: Normal and Pneumonia
    model.load_state_dict(torch.load("pneumonia_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your chatbot knowledge base (this can be adjusted based on your local PDF processing)
def load_encyclopedia():
    # Simulating a medicine encyclopedia (you should implement the actual loading and querying logic)
    return {
        "pneumonia": "Pneumonia is an infection that inflames the air sacs in one or both lungs...",
        "normal": "Normal refers to a healthy state where no infection or illness is detected."
    }

# Initialize encyclopedia
encyclopedia = load_encyclopedia()
# Function to handle chatbot queries
def chatbot_query(query):
    query = query.lower()
    response = encyclopedia.get(query, "Sorry, I don't have information on that topic.")
    return response
# Streamlit UI
st.title("Pneumonia Detection and Chatbot")
st.write("Upload a chest X-ray image, and the model will predict whether it indicates Pneumonia or not.")

# Tabs for chatbot and image detection
tab1, tab2 = st.tabs(["Pneumonia Detection", "Chatbot"])

# Pneumonia detection tab
with tab1:
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        img_tensor = transform(image).unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            class_names = ["Normal", "Pneumonia"]
            prediction = class_names[predicted.item()]

        st.write(f"### Prediction: {prediction}")

# Chatbot tab
with tab2:
    st.write("Ask me about pneumonia or general health-related topics.")

    user_query = st.text_input("You: ", "")
    
    if user_query:
        response = chatbot_query(user_query)
        st.write(f"Bot: {response}")
