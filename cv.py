# Step 1: Create a new Streamlit application using Python and configure an appropriate page title and layout.
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import torch.nn.functional as F

st.set_page_config(
    page_title="Computer Vision Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# Step 2: Import the required libraries including Streamlit, PyTorch, Torchvision, PIL, and Pandas.
# (Already done above in Step 1)

# Step 3: Configure the application to run only on CPU settings.
torch.set_default_tensor_type(torch.FloatTensor)  # Ensure CPU usage
device = torch.device("cpu")

st.title("Image Classification Web Application")
st.write("This app uses a pre-trained ResNet18 model to classify images from the ImageNet dataset.")

# Step 4: Load a pre-trained ResNet18 model from torchvision.models and set the model to evaluation mode.
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

# Step 5: Apply the recommended image preprocessing transformations associated with the ResNet18 pre-trained weights.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet class labels
@st.cache_data
def load_labels():
    import requests
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    labels = response.text.strip().split("\n")
    return labels

imagenet_labels = load_labels()

# Step 6: Design a user interface that allows users to upload an image file (e.g., JPG or PNG).
uploaded_file = st.file_uploader("Choose an image file (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Step 7: Convert the uploaded image into a tensor and perform model inference using PyTorch without gradient computation.
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Perform inference without gradient computation
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Step 8: Apply the softmax function to the model output and display the top-5 predicted classes along with their probabilities.
    probabilities = F.softmax(outputs[0], dim=0)
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    st.subheader("Top-5 Predictions:")
    
    # Display predictions in a table
    predictions_df = pd.DataFrame({
        "Class": [imagenet_labels[idx] for idx in top5_indices],
        "Probability": [f"{prob:.4f}" for prob in top5_prob]
    })
    
    st.table(predictions_df)
    
    # Step 9: Visualize the prediction probabilities using a bar chart in Streamlit.
    st.subheader("Probability Distribution (Bar Chart):")
    
    # Prepare data for bar chart
    chart_data = pd.DataFrame({
        "Class": [imagenet_labels[idx] for idx in top5_indices],
        "Probability": [prob.item() for prob in top5_prob]
    })
    
    # Set class names as index for better display
    chart_data.set_index("Class", inplace=True)
    
    # Display bar chart
    st.bar_chart(chart_data)

# Step 10: Run the Streamlit application and test the system using multiple images. 
# Discussion will be included in the PDF report.
st.divider()
st.write("**Testing Instructions:**")
st.write("1. Click 'Run' in your IDE or execute: `streamlit run app.py`")
st.write("2. Upload various test images (animals, objects, vehicles)")
st.write("3. Observe the top-5 predictions and their probabilities")
st.write("4. Note the confidence levels for different image types")

st.write("**Deployment Details:**")
st.write("- GitHub Repository: [Your Repository Link]")
st.write("- Live Deployment: [Your Streamlit Cloud URL]")