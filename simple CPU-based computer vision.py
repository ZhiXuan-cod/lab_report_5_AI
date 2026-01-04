import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# Step 1: Configure Streamlit page
st.set_page_config(
    page_title="ResNet18 Image Classifier",
    page_icon="",
    layout="centered"
)

st.title("Computer Vision Image Classification")
st.write("Using **PyTorch ResNet18 (pretrained on ImageNet)**")

# Step 3: Force CPU usage
device = torch.device('cpu')

# Step 4: Load pre-trained ResNet18 model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Step 5: Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Step 6: Image upload UI
uploaded_file = st.file_uploader(
    "Upload an image (JPG/PNG)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Step 7: Preprocess and run inference
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_batch)
    
    # Step 8: Apply softmax and get top-5 predictions
    probabilities = F.softmax(outputs[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Load ImageNet class labels
    import requests
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(url).text.strip().split("\n")
    
    # Display predictions
    st.subheader("Top 5 Predictions")
    
    # Create DataFrame for table display
    results = []
    for i in range(5):
        results.append({
            "Class": labels[top5_catid[i]],
            "Probability": f"{top5_prob[i].item():.4f}"
        })
    
    df = pd.DataFrame(results)
    st.table(df)
    
    # Step 9: Create bar chart visualization
    st.subheader("Prediction Probabilities Chart")
    chart_data = pd.DataFrame({
        "Probability": [float(p) for p in top5_prob],
        "Class": [labels[idx] for idx in top5_catid]
    })
    st.bar_chart(chart_data.set_index("Class")["Probability"])
    
else:
    st.info("👆 Please upload an image to classify.")