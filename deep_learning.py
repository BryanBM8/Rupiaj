import streamlit as st
import torch
from utils import load_pickle, color_hist_equalize, extract_features, labels, load_pt
from camera_input_live import camera_input_live
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from streamlit import session_state as state
from torchvision import transforms
from ultralytics import YOLO
import math

st.title("Deep Learning")

mode = st.segmented_control("Select Mode", ["Realtime Camera", "Upload Image"], default="Upload Image", selection_mode="single")
selection = st.selectbox("Choose model", ["ResNet50","YoLoV11"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if mode == "Realtime Camera":
  image = camera_input_live(1000)
elif mode == "Upload Image":
  image = st.file_uploader("Upload an Image", type=['jpg', 'png'])

if image:
  # st.image(image)
  image = np.asarray(bytearray(image.read()), dtype="uint8") 
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  image = color_hist_equalize(image)
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  model = state['models'][selection]
  if selection == "ResNet50":
      image = cv2.resize(image, (512, 512))
      transform = transforms.Compose([
          transforms.ToTensor(),
      ])
      
      inference_image = transform(image)
      inference_image = inference_image.to(device)
      model.eval()
      with torch.no_grad():
        # st.write(inference_image.unsqueeze(0).shape)
        pred = model(inference_image.unsqueeze(0))
        st.write(pred.data)
        _, pred = torch.max(pred.data, 1)
        st.write(pred)
        st.write(labels[int(pred)])

  elif (selection == "YoLoV11"):
      image = cv2.resize(image, (640, 640))
      results = model.predict(image)
      names = model.names
      # st.write(labels[int(pred)])
      for r in results:
          boxes = r.boxes
          # st.write(boxes)
          for box in boxes:
              x1, y1, x2, y2 = box.xyxy[0]
              x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
              cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
              confidence = math.ceil((box.conf[0]*100))/100
              cls = int(box.cls)
              org = [x1, y1-20]
              font = cv2.FONT_HERSHEY_SIMPLEX

              cv2.putText(image, names[cls], org, font, 1, (0, 0, 255), 2)
              st.write(f"{names[cls]}, confidence: {confidence}")
  st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  
  
