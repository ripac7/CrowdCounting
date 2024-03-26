import os
import random
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from config import image_data_test_path, norm
from utils import single_evaluate
from models import CGDRCN1, MyFC
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
game_images_path = 'game_images/img'

# Define Session Variables 
if 'first' not in st.session_state:
    st.session_state.first = True

if 'playing' not in st.session_state:
    st.session_state.playing = False

if 'userguess' not in st.session_state:
    st.session_state.userguess = 0

if 'img_path' not in st.session_state:
    st.session_state.img_path = ''

if 'waiting' not in st.session_state:
    st.session_state.waiting=False


# Define Functions
def store_guess (userguess):
    st.session_state.userguess = userguess
    st.session_state.playing = True

def ripristina ():
    st.session_state.playing = False
    st.session_state.first = True
    st.session_state.waiting = False
    st.session_state.userguess = 0

def display_image(folder_path):
    files = os.listdir(folder_path)
    i = random.randint(0,len(files)-1)
    random_image_path = os.path.join(folder_path, files[i])
    st.image(random_image_path)
    return random_image_path

def get_gt(image_path):
    map_path = (image_path).replace("img", "den").replace(".jpg", ".csv")
    density_map_csv = pd.read_csv(map_path, header=None)
    gt = density_map_csv.values
    gt_heads = int(np.sum(gt))
    return gt_heads

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def wait():
    st.session_state.waiting = True

def display_uploaded_image(image):
    st.image(image, caption='Uploaded Image', use_column_width=True)

def display_density_map(prediction):
    st.markdown("<h2 style='text-align: center;'>Density Map:</h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.imshow(prediction, cmap=plt.cm.jet)
    ax.axis('off')
    st.pyplot(fig)

# Header
st.markdown("<h1 style='text-align: center;'>People Counter</h1>", unsafe_allow_html=True)

# Add a selectbox for selecting the mode
mode = st.selectbox("Select Mode", ["Predict", "Play a Game"])

if mode == 'Predict':
    # Existing prediction functionality
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
                    model_path = 'models/model1-50epochs-lr5.pt'
                else:
                    model_path = 'models/model2-lr6-unfreezed.pt'
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
    

elif mode == 'Play a Game': 

    if (st.session_state.first==True):
        st.session_state.first=False
        col1, col2, col3 = st.columns([2.35,1,2])
        with col2:
            st.button("Play")

    elif (st.session_state.first==False and st.session_state.playing==False and st.session_state.waiting==False):
        st.session_state.img_path = display_image(game_images_path)
        userguess = st.number_input("Enter your guess for the number of people:", min_value=0, step=1, on_change=wait())
        col1, col2, col3 = st.columns([2.35,1,2])
        with col2:
            st.button("Submit", on_click=store_guess, args=[userguess])
    
    elif (st.session_state.first==False and st.session_state.playing==False and st.session_state.waiting==True):
        st.image(st.session_state.img_path)
        userguess = st.number_input("Enter your guess for the number of people:", min_value=0, step=1, on_change=wait())
        col1, col2, col3 = st.columns([2.35,1,2])
        with col2:
            st.button("Submit", on_click=store_guess, args=[userguess])

    elif (st.session_state.first==False and st.session_state.playing==True):
        st.image(st.session_state.img_path)
        gt_heads = get_gt(st.session_state.img_path)
        model = load_model('models/model1-50epochs-lr5.pt', device)
        out = single_evaluate(model, st.session_state.img_path, norm)
        predicted_count = int(out['count'])
        
        # Compare the user's guess with the ground truth count
        user_error = abs(st.session_state.userguess - gt_heads)
        model_error = abs(predicted_count - gt_heads)
        
        # Determine the winner
        if user_error < model_error:
            st.success(f"Congratulations! You were closer to the ground truth count. Your guess: {st.session_state.userguess}, Ground Truth: {gt_heads}, Model Prediction: {predicted_count}")
        elif user_error > model_error:
            st.error(f"Sorry! The model was closer to the ground truth count. Your guess: {st.session_state.userguess}, Ground Truth: {gt_heads}, Model Prediction: {predicted_count}")
        else:
            st.info(f"It's a tie! Your guess: {st.session_state.userguess}, Ground Truth: {gt_heads}, Model Prediction: {predicted_count}")       
        
        col1, col2, col3 = st.columns([2.35,1,2])
        with col2:
            st.button("Play Again", on_click=ripristina)