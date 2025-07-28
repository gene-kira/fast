# MagicBox.py
import subprocess, sys, os, time

# === AUTO-INSTALL REQUIRED LIBRARIES ===
required = [
    "playsound==1.2.2",
    "speechrecognition",
    "pyttsx3",
    "face_recognition",
    "opencv-python",
    "winshell",
    "pypiwin32"
]
for pkg in required:
    try:
        __import__(pkg.split("==")[0])
    except ImportError:
        subprocess.call([sys.executable, "-m", "pip", "install", pkg])

# === IMPORTS AFTER AUTO-INSTALL ===
import tkinter as tk
from tkinter import messagebox, ttk
from playsound import playsound
import speech_recognition as sr
import pyttsx3
import face_recognition
import cv2
import winshell
from win32com.client import Dispatch

# === GUI & SECURITY CONFIG ===
ROLE = "Admin"  # Options: Admin, Spectre, Echo
REFERENCE_FACE = "user_face.jpg"  # Store your face here

ROLE_CONFIG = {
    "Admin": {"bg": "#001F3F", "badge": "🛡️", "voice": "Welcome, Commander."},
    "Spectre": {"bg": "#003B3B", "badge": "🌫️", "voice": "Spectre online."},
    "Echo": {"bg": "#1A1A1A", "badge": "📡", "voice": "Echo mode active."}
}

def verify_face(path=REFERENCE_FACE):
    try:
        known_image = face_recognition.load_image_file(path)
        known_encoding = face_recognition.face_encodings(known_image)[0]
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        cam.release()
        unknown_encodings = face_recognition.face_encodings(frame)
        if not unknown_encodings: return False
        return face_recognition.compare_faces([known_encoding], unknown_encodings[0])[0]
    except:
        return False

def is_device_trusted(device_id):
    return device_id == "USB_VID_1234_PID_5678"  # Replace with actual whitelist logic

def engage_lockdown():
    print("🔒 Lockdown Engaged")
    os.system("taskkill /F /IM OneDrive.exe")
    os.system("reg add HKCU\\Software\\Policies\\ClipboardBlock /v Enabled /t REG_DWORD /d 1 /f")

def disengage_lockdown():
    print("🔓 Lockdown Disengaged")
    os.system("start OneDrive.exe")

def unlock_with_countdown():
    for i in range(5, 0, -1):
        print(f"Unlocking in {i}...")
        time.sleep(1)
    print("✅ Unlock Complete")

def create_shortcut():
    path = os.path.join(winshell.desktop(), "MagicBox Guardian.lnk")
    target = os.path.abspath(__file__)
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(path)
    shortcut.Targetpath = target
    shortcut.WorkingDirectory = os.path.dirname(target)
    shortcut.IconLocation = target
    shortcut.save()

class MagicBoxGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"MagicBox Console - {ROLE}")
        self.geometry("600x520")
        self.configure(bg=ROLE_CONFIG[ROLE]["bg"])
        self.resizable(False, False)

        self.engine = pyttsx3.init()
        self.speak(ROLE_CONFIG[ROLE]["voice"])
        create_shortcut()

        # Header
        tk.Label(self, text=ROLE_CONFIG[ROLE]["badge"], font=("Arial", 32), bg=ROLE_CONFIG[ROLE]["bg"], fg="white").pack(pady=10)
        tk.Label(self, text=f"{ROLE} Guardian Protocol", font=("Arial", 20, "bold"), bg=ROLE_CONFIG[ROLE]["bg"], fg="#FEE715").pack()
        self.status = tk.Label(self, text="Status: Ready", font=("Arial", 14), bg=ROLE_CONFIG[ROLE]["bg"], fg="#0FF0FC")
        self.status.pack(pady=10)

        # Override Dropdown
        self.override_var = tk.StringVar(self)
        self.override_menu = ttk.Combobox(self, textvariable=self.override_var, values=["Emergency Access", "Admin Override", "Scheduled Maintenance"])
        self.override_menu.set("Select Override Reason")
        self.override_menu.pack(pady=5)

        # Buttons
        self.add_button("🧬 Scan My Face", self.run_biometric)
        self.add_button("🔎 Check USB Device", self.run_whitelist)
        self.add_button("🔒 Lock Everything", self.run_lockdown, "#C70039")
        self.add_button("🌀 Unlock Vault", self.run_unlock, "#900C3F")
        self.add_button("🎙️ Talk to Console", self.listen_command, "#222222")

        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    def add_button(self, label, command, bg="#0FF0FC"):
        tk.Button(self, text=label, command=command, font=("Arial", 14), bg=bg, fg="black").pack(fill='x', padx=60, pady=5)

    def speak(self, message):
        self.engine.say(message)
        self.engine.runAndWait()

    def run_biometric(self):
        self.status.config(text="Scanning face...")
        if verify_face():
            self.speak("Identity confirmed.")
            self.status.config(text="✅ Biometric success.")
        else:
            self.speak("Face not recognized.")
            self.status.config(text="❌ Biometric failed.")

    def run_whitelist(self):
        device_id = "USB_VID_1234_PID_5678"
        if is_device_trusted(device_id):
            self.speak("Device authorized.")
            self.status.config(text="✅ Trusted USB.")
        else:
            self.speak("Device not authorized.")
            self.status.config(text="❌ Untrusted USB.")

    def run_lockdown(self):
        engage_lockdown()
        self.speak("Vault lockdown engaged.")
        self.status.config(text="🔒 Lockdown active.")

    def run_unlock(self):
        unlock_with_countdown()
        disengage_lockdown()
        self.speak("Vault unlocked.")
        self.status.config(text="🔓 Lockdown lifted.")

    def listen_command(self):
        self.status.config(text="🎤 Listening...")
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                audio = r.listen(source)
            cmd = r.recognize_google(audio).lower()
            if "lock everything" in cmd:
                self.run_lockdown()
            elif "unlock vault" in cmd:
                self.run_unlock()
            else:
                self.speak("Command not recognized.")
                self.status.config(text="❓ Unknown command.")
        except:
            self.speak("Voice input failed.")
            self.status.config(text="⚠️ Microphone error.")

    def on_exit(self):
        if messagebox.askokcancel("Exit", "Shutdown Guardian Console?"):
            self.speak("Goodbye.")
            self.destroy()

# === LAUNCH CONSOLE ===
if __name__ == "__main__":
    MagicBoxGUI().mainloop()

