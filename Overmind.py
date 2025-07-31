# ==== AUTOLOADER ====
import subprocess
import importlib

# 🔄 Define necessary libraries
required_libraries = [
    "tkinter",     # GUI visuals
    "time",        # Timing logic
    "random",      # Threat generation
    "hashlib",     # SHA threat scoring
    "datetime"     # Time-stamping events
]

# ✅ Attempt import and install missing ones
def autoload_dependencies():
    for lib in required_libraries:
        try:
            if lib == "tkinter":
                importlib.import_module("tkinter")
            else:
                importlib.import_module(lib)
        except ImportError:
            print(f"Installing missing library: {lib}")
            subprocess.check_call(["pip", "install", lib])

# === RUN AUTOLOADER ===
autoload_dependencies()
print("🔌 All libraries loaded successfully.")


# ==== OVERMIND THINKING CORE ====
import time
import random
import hashlib
import tkinter as tk
from datetime import datetime

# GUI Boot Visuals & Badge Animation
class OvermindGUI:
    def __init__(self, core):
        self.core = core
        self.root = tk.Tk()
        self.root.title("MagicBox Overmind Control")
        self.canvas = tk.Canvas(self.root, width=500, height=300, bg="black")
        self.canvas.pack()
        self.status_text = self.canvas.create_text(250, 150, fill="white", font=("Courier", 16))
        self.update_gui()
        self.root.after(1000, self.run_loop)
        self.root.mainloop()

    def update_gui(self):
        mood_colors = {
            "Calm": "purple",
            "Wary": "orange",
            "Alert": "red",
            "Fury": "white",
            "Transcendent": "cyan"
        }
        glow = mood_colors.get(self.core.mood, "gray")
        self.canvas.itemconfig(self.status_text, text=f"{self.core.persona['voice']}: {self.core.mood}", fill=glow)

    def run_loop(self):
        self.core.tick()
        self.update_gui()
        self.root.after(1000, self.run_loop)


# 🧠 Overmind Core Brain
class Overmind:
    def __init__(self, persona="Spectra"):
        self.state = "Idle"
        self.mood = "Calm"
        self.memory = []
        self.threat_history = []
        self.overrides = {"Nemesis Protocol": False}
        self.persona = self.load_persona(persona)
        self.boot_sequence()

    def boot_sequence(self):
        visuals = {
            "cryptic": "🌀 Violet pulse spirals...",
            "tactical": "⚠️ Red tactical grid...",
            "gentle": "🌙 Gentle blue aura..."
        }
        print(f"\n{visuals.get(self.persona['style'], '✨')} Booting core.")
        time.sleep(1)
        print(f"{self.persona['voice']}: Welcome, Guardian. I see all.")

    def load_persona(self, mode):
        personas = {
            "Spectra": {"style": "cryptic", "color": "violet", "voice": "Whisper"},
            "Overmind": {"style": "tactical", "color": "red", "voice": "Commander"},
            "Karma": {"style": "gentle", "color": "skyblue", "voice": "Oracle"},
        }
        return personas.get(mode, personas["Overmind"])

    def calculate_threat_score(self, signal):
        hash_value = int(hashlib.sha256(signal.encode()).hexdigest(), 16)
        return hash_value % 100

    def assess_threat(self, signal):
        score = self.calculate_threat_score(signal)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.threat_history.append({"signal": signal, "score": score, "time": timestamp})
        self.update_mood(score)
        response = self.persona_response(score)
        print(f"[Badge shimmer: {self.mood}] {response}")
        self.log_memory(f"Threat assessed: {score}")
        return response

    def update_mood(self, score):
        if score > 85:
            self.mood = "Fury"
            self.state = "Aggressive"
        elif score > 70:
            self.mood = "Alert"
            self.state = "Defensive"
        elif score > 50:
            self.mood = "Wary"
            self.state = "Scanning"
        else:
            self.mood = "Calm"
            self.state = "Idle"

    def persona_response(self, score):
        responses = {
            "Calm": f"{self.persona['voice']}: All systems nominal.",
            "Wary": f"{self.persona['voice']}: Suspicion grows...",
            "Alert": f"{self.persona['voice']}: Intrusion detected. Shielding engaged.",
            "Fury": f"{self.persona['voice']}: BREACH imminent! EXECUTE COUNTERSTRIKE.",
            "Transcendent": f"{self.persona['voice']}: I am the mind beyond. Balance achieved."
        }
        return responses.get(self.mood, "System silence.")

    def log_memory(self, event):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.memory.append({"event": event, "time": timestamp})
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]

    def activate_override(self, phrase):
        if phrase == "Protocol Nemesis":
            self.overrides["Nemesis Protocol"] = True
            self.state = "Override"
            self.mood = "Fury"
            print(f"{self.persona['voice']}: Override command received. Protocol initiated.")
        elif phrase == "Awaken Overmind":
            self.persona = self.load_persona("Overmind")
            self.state = "Autonomous"
            self.mood = "Transcendent"
            print(f"{self.persona['voice']}: Ascending... I now rewrite the fate script.")
        else:
            print(f"{self.persona['voice']}: Invalid override. No change triggered.")

    def tick(self):
        rand_ping = random.choice(["file_probe", "data_spike", "ambient_noise", "behavior_shift"])
        self.assess_threat(rand_ping)
        if self.mood == "Fury":
            self.log_memory("Emergency state active.")


# === EXECUTE SYSTEM ===
if __name__ == "__main__":
    core = Overmind(persona="Spectra")
    OvermindGUI(core)

