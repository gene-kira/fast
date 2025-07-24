# MagicBox Guardian - All-in-One Friendly Sentinel System

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

# 🧩 Autoload Required Libraries
def autoload_libraries():
    required = ["pyttsx3", "psutil", "matplotlib", "ttkthemes"]
    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

autoload_libraries()

# 🧠 Guardian Emotion Engine
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
        self.voice_enabled = True

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
        if not self.voice_enabled:
            return
        mood_dialogues = {
            "normal": "Systems green. MagicBox Guardian standing by.",
            "vigilant": "I’m watching closely. Something feels off.",
            "alert": "Warning! Intrusion detected. Defenses engaged.",
            "oracle": "Patterns form. Intelligence awakens.",
            "sarcasm": "Yup. Still here. Waiting for something exciting."
        }
        message = mood_dialogues.get(mood, "Mood undefined.")
        self.engine.say(message)
        self.engine.runAndWait()

    def toggle_voice(self):
        self.voice_enabled = not self.voice_enabled
        return self.voice_enabled

# 💬 Create Guardian
guardian_ai = GuardianEmotionEngine()

# 🌈 Mood Color Matrix
mood_colors = {
    "normal": "green",
    "vigilant": "orange",
    "alert": "red",
    "oracle": "purple",
    "sarcasm": "blue"
}

# 🚀 GUI Launcher
def launch_gui():
    root = ThemedTk(theme="arc")
    root.title("🛡️ MagicBox Guardian Console")
    root.geometry("620x560")

    # Title
    tk.Label(root, text="MagicBox Guardian", font=("Arial", 20, "bold")).pack(pady=10)
    status_label = tk.Label(root, text="Mood: NORMAL", font=("Arial", 16))
    status_label.pack(pady=10)

    # 🟢 Mood Orb
    canvas = tk.Canvas(root, width=80, height=80, highlightthickness=0)
    canvas.pack()
    status_orb = canvas.create_oval(10, 10, 70, 70, fill="green")

    def update_status_visual(mood):
        color = mood_colors.get(mood, "gray")
        canvas.itemconfig(status_orb, fill=color)
        if mood == "alert":
            pulse_alert()

    def pulse_alert():
        current = canvas.itemcget(status_orb, "fill")
        next_color = "red" if current != "darkred" else "darkred"
        canvas.itemconfig(status_orb, fill=next_color)
        canvas.after(500, pulse_alert)

    guardian_active = [False]

    # Toggle Guardian
    def toggle_guardian():
        guardian_active[0] = not guardian_active[0]
        mood = guardian_ai.receive_signals(["button_click"])
        status_label.config(text=f"Mood: {mood.upper()}")
        update_status_visual(mood)
        messagebox.showinfo("Guardian Status", f"Guardian is now {'ON' if guardian_active[0] else 'OFF'}")

    tk.Button(root, text="🟢 Toggle Guardian", font=("Arial", 14), width=30, command=toggle_guardian).pack(pady=10)

    # Voice Toggle
    def toggle_voice_button():
        is_on = guardian_ai.toggle_voice()
        state = "ON" if is_on else "OFF"
        messagebox.showinfo("Voice Control", f"Voice feedback is now {state}")
        voice_btn.config(text=f"🎙️ Voice: {state}")

    voice_btn = tk.Button(root, text="🎙️ Voice: ON", font=("Arial", 14), width=30, command=toggle_voice_button)
    voice_btn.pack(pady=10)

    # Probe Scan
    def run_probe():
        fig = Figure(figsize=(4.5, 2.5), dpi=100)
        ax = fig.add_subplot(111)
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        ax.bar(["CPU", "Memory"], [cpu, mem], color=["green", "blue"])
        ax.set_title("System Probe")

        canvas_graph = FigureCanvasTkAgg(fig, master=root)
        canvas_graph.draw()
        canvas_graph.get_tk_widget().pack(pady=5)

        guardian_ai.receive_signals([
            "high_cpu" if cpu > 75 else "probe_triggered",
            "high_memory" if mem > 70 else "probe_triggered"
        ])
        mood = guardian_ai.current_mood
        status_label.config(text=f"Mood: {mood.upper()}")
        update_status_visual(mood)

    tk.Button(root, text="📊 Run System Probe", font=("Arial", 14), width=30, command=run_probe).pack(pady=10)

    # Simulate Threat
    def simulate_threat():
        guardian_ai.receive_signals(["intrusion_attempt", "rogue_process_found", "signature_anomaly"])
        mood = guardian_ai.current_mood
        status_label.config(text=f"Mood: {mood.upper()}")
        update_status_visual(mood)
        messagebox.showwarning("Security Alert", "🚨 Threat detected! Containment engaged.")

    tk.Button(root, text="🚨 Simulate Threat", font=("Arial", 14), width=30, command=simulate_threat).pack(pady=10)

    # Exit
    tk.Button(root, text="❌ Exit Program", font=("Arial", 14), width=30, command=root.destroy).pack(pady=20)

    root.mainloop()

# 💥 Launch Program
if __name__ == "__main__":
    launch_gui()

