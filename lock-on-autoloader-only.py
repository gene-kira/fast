import cv2
import numpy as np
import random
from yolov8 import YOLOv8  # Assuming this is a custom or third-party YOLOv8 implementation
import torch

# Ensure the necessary libraries are installed
try:
    import torchvision.transforms as transforms
except ImportError:
    print("Installing torchvision...")
    !pip install torchvision

try:
    import sounddevice as sd
    import librosa
except ImportError:
    print("Installing sounddevice and librosa...")
    !pip install sounddevice librosa

# Import necessary libraries
import cv2
import numpy as np
import random
from yolov8 import YOLOv8  # Assuming this is a custom or third-party YOLOv8 implementation
import torch
import torchvision.transforms as transforms
import sounddevice as sd
import librosa
