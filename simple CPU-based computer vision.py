import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# Page config
st.set_page_config(page_title="AI Image Classifier", layout="wide")
st.title("🖼️ Computer Vision Image Classifier")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.info(f"Using device: {device}")

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model.to(device)

model = load_model()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load labels
@st.cache_data
def load_labels():
    try:
        # Try local file first
        with open("imagenet_labels.txt", "r") as f:
            labels = json.load(f)
    except:
        # Fallback to URL
        import urllib.request
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        labels = json.loads(urllib.request.urlopen(url).read().decode())
    return labels

labels = load_labels()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Get top-5 predictions
    top5_probs, top5_indices = torch.topk(probabilities, 5)
    top5_labels = [labels[i] for i in top5_indices]
    
    # Display results
    st.subheader("📊 Top-5 Predictions")
    results_df = pd.DataFrame({
        "Class": top5_labels,
        "Probability": [f"{prob:.4f}" for prob in top5_probs]
    })
    st.table(results_df)
    
    # Bar chart
    fig, ax = plt.subplots()
    ax.barh(top5_labels, top5_probs.numpy())
    ax.set_xlabel("Probability")
    st.pyplot(fig)
    
    # Discussion
    st.subheader("🧠 Discussion")
    st.markdown("""
    - **Model**: Pre-trained ResNet18 from torchvision
    - **Performance**: Accuracy depends on image similarity to ImageNet training data
    - **Limitations**: Model trained on 1000 ImageNet classes; may not recognize niche objects
    - **Reproducibility**: Code available on GitHub with live deployment
    """)
else:
    st.info("👆 Please upload an image file (JPG/PNG)")