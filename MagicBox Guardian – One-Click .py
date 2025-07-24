# MagicBox Guardian – One-Click Old Guy Friendly Edition

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox
from ttkthemes import ThemedTk
from collections import Counter, deque
import pyttsx3
import psutil
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 🧩 Autoloader for Required Libraries
def autoload_libraries():
    required = ["pyttsx3", "psutil", "matplotlib", "ttkthemes"]
    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

autoload_libraries()

# 🧠 Emotion Engine
class GuardianEmotionEngine:
    def __init__(self):
        self.signal_weights = {
            "intrusion_attempt": 10, "rogue_process_found": 8, "memory_leak_detected": 6,
            "high_cpu": 3, "high_memory": 2, "disk_io_spike": 1,
            "idle": -2, "probe_triggered": 3, "button_click": 1, "spam_clicking": -3,
            "signature_anomaly": 5, "probe_failure": 4, "checksum_mismatch": 7
        }
        self.mood_history = deque(maxlen=10)
        self.voice_profiles = {
            "normal": 0, "vigilant": 1, "alert": 2,
            "oracle": 3, "sarcasm": 4
        }
        self.engine = pyttsx3.init()
        self.current_mood = "normal"

    def receive_signals(self, signals):
        signal_count = Counter(signals)
        total_score = sum(self.signal_weights.get(sig, 0) * freq for sig, freq in signal_count.items())
        mood = self.evaluate_mood(total_score)
        self.mood_history.append(mood)
        self.current_mood = mood
        self.apply_voice_profile(mood)
        self.speak_mood_dialogue(mood)
        return mood

    def evaluate_mood(self, score):
        if score <= 0: return "sarcasm"
        elif score <= 5: return "normal"
        elif score <= 10: return "vigilant"
        elif score <= 15: return "oracle"
        else: return "alert"

    def apply_voice_profile(self, mood):
        index = self.voice_profiles.get(mood, 0)
        voices = self.engine.getProperty('voices')
        if index < len(voices):
            self.engine.setProperty('voice', voices[index].id)

    def speak_mood_dialogue(self, mood):
        mood_dialogues = {
            "normal": "Systems green. MagicBox Guardian standing by.",
            "vigilant": "Something's off. I'm watching everything.",
            "alert": "Warning! Danger detected! Locking it down.",
            "oracle": "I sense patterns deep in the data...",
            "sarcasm": "Well well... you again. At least push the right button this time."
        }
        message = mood_dialogues.get(mood, "Mood undefined.")
        self.engine.say(message)
        self.engine.runAndWait()

# 💬 Instantiate Guardian
guardian_ai = GuardianEmotionEngine()

# 🖼️ GUI Launcher
def launch_gui():
    root = ThemedTk(theme="arc")
    root.title("🛡️ MagicBox Guardian Control Panel")
    root.geometry("600x500")

    status_label = tk.Label(root, text="Mood: NORMAL", font=("Arial", 16))
    status_label.pack(pady=20)

    guardian_active = [False]

    # Toggle button
    def toggle_guardian():
        guardian_active[0] = not guardian_active[0]
        mood = guardian_ai.receive_signals(["button_click"])
        status_label.config(text=f"Mood: {mood.upper()}")
        messagebox.showinfo("MagicBox Guardian", f"Guardian is now {'ON' if guardian_active[0] else 'OFF'}")

    tk.Button(root, text="🟢 Toggle Guardian", font=("Arial", 14), width=25, command=toggle_guardian).pack(pady=10)

    # Probe chart
    def run_probe():
        fig = Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        ax.bar(["CPU", "Memory"], [cpu, mem], color=["green", "blue"])
        ax.set_title("System Probes")

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

        guardian_ai.receive_signals([
            "high_cpu" if cpu > 75 else "probe_triggered",
            "high_memory" if mem > 70 else "probe_triggered"
        ])

    tk.Button(root, text="📊 Run System Probe", font=("Arial", 14), width=25, command=run_probe).pack(pady=10)

    # Simulate threat
    def simulate_threat():
        guardian_ai.receive_signals(["intrusion_attempt", "rogue_process_found", "signature_anomaly"])
        status_label.config(text=f"Mood: {guardian_ai.current_mood.upper()}")
        messagebox.showwarning("Security Alert", "Threat detected! Systems are responding.")

    tk.Button(root, text="🚨 Simulate Threat", font=("Arial", 14), width=25, command=simulate_threat).pack(pady=10)

    # Exit
    tk.Button(root, text="❌ Exit Program", font=("Arial", 14), width=25, command=root.destroy).pack(pady=20)

    root.mainloop()

# 🚀 Launch
if __name__ == "__main__":
    launch_gui()

