import pickle
import cv2
import numpy as np
import gzip

import torch
from torchvision import models
from ultralytics import YOLO

labels = ['1 RIBU ASLI', '1 RIBU PALSU', '10 RIBU ASLI', '10 RIBU PALSU', 
         '100 RIBU ASLI', '100 RIBU PALSU', '20 RIBU ASLI', '20 RIBU PALSU', 
         '5 RIBU ASLI', '5 RIBU PALSU', '50 RIBU ASLI', '50 RIBU PALSU']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# microtext
def extract_microtext(image_gray):
    edges = cv2.Canny(image_gray, 50, 150)
    return np.sum(edges) / edges.size

# warna
def extract_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_mean = np.mean(hsv[:, :, 0])
    sat_mean = np.mean(hsv[:, :, 1])
    val_mean = np.mean(hsv[:, :, 2])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    intensity_mean = np.mean(gray)

    red_mean = np.mean(image[:, :, 2])
    green_mean = np.mean(image[:, :, 1])
    blue_mean = np.mean(image[:, :, 0])

    return hue_mean, sat_mean, val_mean, intensity_mean, red_mean, green_mean, blue_mean

# benang pengaman
def detect_thread(image_gray):
    _, thresh = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    thread = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    return np.sum(thread) / thread.size


def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray=cv2.equalizeHist(gray)
    microtext_density = extract_microtext(gray)
    hue, saturation, value, intensity, red, green, blue = extract_color(image)
    thread_density = detect_thread(gray)
    return [microtext_density, hue, saturation, value, intensity, red, green, blue, thread_density]
    

class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Linear(in_features=2048, out_features=12)
        
    def forward(self, x):
        return self.resnet(x)
    
class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        self.vgg19.fc = torch.nn.Linear(in_features=2048, out_features=12)
        # self.resnet.compile()
        
    def forward(self, x):
        return self.vgg19(x)

def load_pt(model_name=None):
    if model_name == "ResNet50":
        model = ResNet50()
        # model = VGG19()
        model.load_state_dict(torch.load('resnet50-4.pth', map_location=torch.device(device)))
        model = model.to(device)
        model.eval()
    if model_name == "YoLoV11":
        model = YOLO("last-4.pt")
    return model