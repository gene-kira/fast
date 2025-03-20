import subprocess
import sys
import datetime
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import random
import tkinter as tk
import webbrowser
import logging
import optuna
import tensorflow as tf
import win32com.client

# Auto-install required libraries
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_libraries = [
    "pywin32",
    "pyttsx3",
    "opencv-python",
    "numpy",
    "requests",
    "beautifulsoup4",
    "transformers",
    "tensorflow",
    "optuna"
]

for library in required_libraries:
    try:
        __import__(library)
    except ImportError:
        install(library)

# Load the conversational model
chatbot = pipeline('conversational', model='microsoft/DialoGPT-medium')

class HAL9000:
    def __init__(self):
        self.conversation_history = []
        self.current_user = None
        self.known_faces = {}
        self.code_snippets = []

    def get_time(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def search_internet(self, query):
        try:
            url = f"https://www.google.com/search?q={query}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('div', class_='BNeawe iBp4i AP7Wnd')
            return results[0].text if results else "No results found."
        except Exception as e:
            return f"An error occurred: {e}"

    def execute_command(self, command):
        try:
            exec(command)
            self.code_snippets.append((command, "Success"))
            return "Command executed successfully."
        except Exception as e:
            self.code_snippets.append((command, str(e)))
            return f"An error occurred while executing the command: {e}"

    def speak(self, text, confidence=1.0):
        if confidence < 1.0:
            text = self.add_uncertainty(text)
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    def add_uncertainty(self, text):
        uncertain_responses = [
            f"I'm not entirely sure, but {text}",
            f"Let me think... {text}",
            f"I believe {text}, but I could be wrong",
            f"Based on my current understanding, {text}",
            f"{text}, although there's a small chance I might be incorrect"
        ]
        return random.choice(uncertain_responses)

    def listen(self):
        recognizer = win32com.client.Dispatch("SAPI.SpSharedRecognizer")
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        phrase = None

        while True:
            try:
                print("Listening...")
                phrase = recognizer.AudioInput.InteractiveStream
                if phrase is not None and len(phrase) > 0:
                    user_input = phrase.Text
                    print(f"You said: {user_input}")
                    break
            except Exception as e:
                return "Sorry, I did not understand that."

        return user_input

    def recognize_face(self):
        import face_recognition
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                if len(self.known_faces) == 0:
                    self.known_faces.append((face_encoding, input("Please enter your name: ")))
                else:
                    matches = face_recognition.compare_faces([known_face[0] for known_face in self.known_faces], face_encoding)
                    if True in matches:
                        user = self.known_faces[matches.index(True)][1]
                        print(f"Recognized {user}")
                        break
                    else:
                        new_user = input("New user detected. Please enter your name: ")
                        self.known_faces.append((face_encoding, new_user))
                        print(f"Welcome {new_user}")

            cap.release()
            cv2.destroyAllWindows()

    def greet(self, user):
        return f"Hello {user}, how can I assist you today?"

    def process_command(self, command):
        if "search" in command:
            query = command.replace("search", "").strip()
            result = self.search_internet(query)
            self.speak(result)
        elif "execute" in command:
            code = command.replace("execute", "").strip()
            result = self.execute_command(code)
            self.speak(result)
        else:
            response = chatbot(command)[0]['generated_text']
            self.speak(response)

    def main_loop(self):
        while True:
            user_input = self.listen().lower()
            if "exit" in user_input or "quit" in user_input:
                self.speak("Goodbye!")
                break
            self.process_command(user_input)

def main():
    hal = HAL9000()
    hal.main_loop()

if __name__ == "__main__":
    main()
