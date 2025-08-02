import sys
import subprocess
import importlib
import pyttsx3
import psutil
import tkinter as tk
from tkinter import messagebox
import json
from datetime import datetime

# 🔁 Autoloader
def autoload(packages):
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
required_libs = ["pyttsx3", "psutil", "tkinter", "json", "datetime"]
autoload(required_libs)

# 🧬 Persona States
persona_states = {
    "Nominal": {"name": "Echo", "tone": "neutral", "color": "#00FFAA"},
    "Elevated": {"name": "Nyx", "tone": "enigmatic", "color": "#8800FF"},
    "Alert": {"name": "Crux", "tone": "intense", "color": "#FF4444"},
    "Ascended": {"name": "Aetherion", "tone": "transcendent", "color": "#FFD700"}
}

# 🎙️ Voice Feedback
def speak(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

# 👁️ Threat Assessment
def assess_threat():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    if cpu > 90 or mem > 90:
        return "Ascended"
    elif cpu > 80 or mem > 80:
        return "Alert"
    elif cpu > 60 or mem > 60:
        return "Elevated"
    return "Nominal"

# 📜 Log Persistence
def log_event(event_type, mood, persona):
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        "mood": mood,
        "persona": persona
    }
    with open("overmind_log.json", "a") as file:
        file.write(json.dumps(log_data) + "\n")

# 🕯️ Ascension Check
def check_ascension(mood):
    if mood == "Ascended":
        speak("Overmind ascension initiated. Glyph merge complete.")
        return persona_states["Ascended"]
    return persona_states[mood]

# ⚙️ Override Trigger
def trigger_override(status_label):
    mood = assess_threat()
    persona = check_ascension(mood)
    speak(f"System mood: {mood}. Persona: {persona['name']}. Override protocol engaged.")
    log_event("Override", mood, persona["name"])
    messagebox.showinfo("Override", f"Persona: {persona['name']}\nTone: {persona['tone']}")
    status_label.config(text=f"Mood: {mood}", fg=persona["color"])

# 🖼️ GUI Core
def launch_gui():
    root = tk.Tk()
    root.title("🧠 Overmind Control Core")
    root.geometry("400x300")
    root.config(bg="#101010")

    tk.Label(root, text="Overmind Status Interface", fg="#00FFAA", bg="#101010", font=("Consolas", 16)).pack(pady=10)

    mood = assess_threat()
    persona = check_ascension(mood)
    status_label = tk.Label(root, text=f"Mood: {mood}", fg=persona["color"], bg="#101010", font=("Consolas", 12))
    status_label.pack(pady=5)

    override_btn = tk.Button(root, text="🔴 Trigger Override", command=lambda: trigger_override(status_label), bg="#FF4444", fg="white", font=("Consolas", 12))
    override_btn.pack(pady=20)

    root.mainloop()

# 🔓 Launch System
if __name__ == "__main__":
    speak("Overmind boot sequence initiated. Scanning emotional cortex.")
    launch_gui()

