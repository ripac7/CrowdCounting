from matplotlib import pyplot as plt
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from io import BytesIO
from utils import test_transform
from models import CGDRCN1, MyFC
from matplotlib import cm as CM
from utils import single_evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
norm = 100

# Function to load the model
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

# Function to display the uploaded image
def display_uploaded_image(image):
    st.image(image, caption='Uploaded Image', use_column_width=True)

# Function to display the density map
def display_density_map(prediction):
    st.markdown("<h2 style='text-align: center;'>Density Map:</h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.imshow(prediction, cmap=plt.cm.jet)
    ax.axis('off')
    st.pyplot(fig)

# Header
st.markdown("<h1 style='text-align: center;'>People Counter</h1>", unsafe_allow_html=True)

# File uploader
st.markdown("<h2 style='text-align: center;'>Upload Image:</h2>", unsafe_allow_html=True)
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Model selection
selected_model = st.selectbox("Select Model", ["Model 1", "Model 2"])

if file is not None:
    # Display uploaded image
    image = Image.open(file)
    display_uploaded_image(image)
    click = False
    
    col1, col2, col3 = st.columns([2.35,1,2])

    with col2:
        if st.button('Predict', key='predict_button'):
            # Load the selected model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if selected_model == "Model 1":
                model_path = 'models\model1-50epochs-lr5.pt'
            else:
                model_path = 'models\model2-lr6-unfreezed.pt'
            model = load_model(model_path, device)
            click = True
        
    if (click):
        # Perform prediction
        out = single_evaluate(model, file, norm)
        predicted_count = int(out['count'])
        
        # Display prediction
        st.success(f"Predicted number of people: {predicted_count}")
        
        # Display density map
        display_density_map(out['prediction'])