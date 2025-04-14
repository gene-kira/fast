import time
import cv2
import pygetwindow as gw
from pypresence import Presence
import win32com.client
import win32con
import win32gui

# Function to check if Microsoft Teams is running
def is_teams_running():
    for window in gw.getWindowsWithTitle('Microsoft Teams'):
        if window.isActive:
            return True
    return False

# Function to block webcam access for all applications except Microsoft Teams
def block_webcam_access():
    cap = cv2.VideoCapture(0)
    while not is_teams_running():
        # Check if the webcam is being used by another application
        ret, frame = cap.read()
        if not ret:
            print("Webcam blocked from unauthorized access.")
        else:
            # Release the webcam to block it
            cap.release()
            cap = cv2.VideoCapture(0)
        time.sleep(1)

# Function to monitor and control webcam access
def monitor_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        if is_teams_running():
            print("Microsoft Teams is running. Webcam access granted.")
            # Release the webcam to allow Microsoft Teams to use it
            cap.release()
        else:
            print("Microsoft Teams is not running. Blocking all other applications from accessing the webcam.")
            block_webcam_access()
        time.sleep(1)

if __name__ == "__main__":
    monitor_webcam()
