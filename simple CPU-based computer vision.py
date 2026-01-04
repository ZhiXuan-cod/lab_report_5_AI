import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Page configuration
st.set_page_config(page_title="AI Image Classifier", layout="wide")

st.title("🖼️ Computer Vision Image Classifier")
st.markdown("Upload an image to classify using a pre-trained ResNet18 model (CPU-only).")

# Step 2: Import libraries (already done above)

# Step 3: Ensure CPU mode
device = torch.device("cpu")

# Step 4: Load pre-trained ResNet18 model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model.to(device)

model = load_model()

# Step 5: Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Step 6: File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 7: Preprocess and inference
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Step 8: Load labels and get top-5 predictions
    labels_path = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = pd.read_json(labels_path, typ="series")

    top5_probs, top5_indices = torch.topk(probabilities, 5)
    top5_labels = labels.iloc[top5_indices].tolist()

    st.subheader("📊 Top-5 Predictions")
    results_df = pd.DataFrame({
        "Class": top5_labels,
        "Probability": [f"{prob:.4f}" for prob in top5_probs]
    })
    st.table(results_df)

    # Step 9: Bar chart visualization
    fig, ax = plt.subplots()
    ax.barh(top5_labels, top5_probs.numpy())
    ax.set_xlabel("Probability")
    ax.set_title("Top-5 Class Probabilities")
    st.pyplot(fig)

    # Step 10: Discussion
    st.subheader("🧠 Discussion")
    st.markdown("""
    - The model successfully classifies common objects using pre-trained ResNet18.
    - The softmax output provides a confidence score for each class.
    - Performance depends on image quality, object clarity, and similarity to ImageNet training data.
    - The system is reproducible via GitHub and deployable on Streamlit Cloud.
    """)

else:
    st.info("👆 Please upload an image to get started.")
