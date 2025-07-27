import os
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import pyttsx3

# === CONFIGURATION ===
MONITORED_FOLDER = "external_files"
BADGE_PATH = "resources/badge.png"
SCAN_INTERVAL = 5
COUNTDOWN_TIME = 10
SELF_DESTRUCT_ENABLED = True

# === VOICE ENGINE SETUP ===
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# === GUI SETUP ===
class MagicBoxGuardian(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MagicBox Guardian - Classified Lab")
        self.geometry("600x400")
        self.configure(bg="#1a1a1a")
        self.badge_label = None
        self.countdown_label = None
        self.init_gui()
        self.monitor_thread = threading.Thread(target=self.monitor_files, daemon=True)
        self.monitor_thread.start()

    def init_gui(self):
        # Load badge image
        badge_img = Image.open(BADGE_PATH).resize((150, 150))
        self.badge_photo = ImageTk.PhotoImage(badge_img)
        self.badge_label = tk.Label(self, image=self.badge_photo, bg="#1a1a1a")
        self.badge_label.place(x=20, y=20)

        # Countdown text
        self.countdown_label = tk.Label(self, text="", font=("Consolas", 20), fg="#ff5555", bg="#1a1a1a")
        self.countdown_label.place(x=200, y=180)

        # Status message
        self.status_label = tk.Label(self, text="Monitoring...", font=("Consolas", 16), fg="#aaa", bg="#1a1a1a")
        self.status_label.place(x=200, y=140)

    def monitor_files(self):
        while True:
            time.sleep(SCAN_INTERVAL)
            if os.listdir(MONITORED_FOLDER):
                self.alert_sequence()

    def alert_sequence(self):
        self.status_label.config(text="THREAT DETECTED", fg="#ff0000")
        speak("Warning. Unauthorized access detected.")
        for i in range(COUNTDOWN_TIME, 0, -1):
            self.countdown_label.config(text=f"Self-destruct in {i}...")
            self.shake_window()
            time.sleep(1)
        self.countdown_label.config(text="Initiating self-destruct.")
        speak("Initiating self destruct.")
        if SELF_DESTRUCT_ENABLED:
            self.self_destruct()

    def shake_window(self):
        x, y = self.winfo_x(), self.winfo_y()
        for _ in range(3):
            self.geometry(f"+{x + 5}+{y}")
            time.sleep(0.05)
            self.geometry(f"+{x - 5}+{y}")
            time.sleep(0.05)
        self.geometry(f"+{x}+{y}")

    def self_destruct(self):
        for f in os.listdir(MONITORED_FOLDER):
            try:
                os.remove(os.path.join(MONITORED_FOLDER, f))
            except Exception as e:
                print(f"Error deleting file: {e}")
        self.status_label.config(text="Files deleted", fg="#00ff00")
        speak("All files have been purged.")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    if not os.path.exists(MONITORED_FOLDER):
        os.makedirs(MONITORED_FOLDER)
    app = MagicBoxGuardian()
    app.mainloop()

