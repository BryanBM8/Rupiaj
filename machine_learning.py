import streamlit as st
from utils import load_pickle, extract_features, labels, color_hist_equalize
from camera_input_live import camera_input_live
import numpy as np
import cv2
from streamlit import session_state as state

st.title("Machine Learning")

mode = st.segmented_control("Select Mode", ["Realtime Camera", "Upload Image"], default="Upload Image", selection_mode="single")
selection = st.selectbox("Choose model", [
    "K-Nearest Neighbour",
    "Support Vector Machine",
    "Random Forest"
])

if mode == "Realtime Camera":
  image = camera_input_live(500)
elif mode == "Upload Image":
  image = st.file_uploader("Upload an Image", type=['jpg', 'png'])

if image:
  image = np.asarray(bytearray(image.read()), dtype="uint8") 
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  # image = color_hist_equalize(image)
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (512, 512))
  features = extract_features(image)
  # st.write(features)
  # st.write(selection)
  model = state['models'][selection]
  pred = model.predict([features])
  st.write(labels[int(pred)])
  st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
