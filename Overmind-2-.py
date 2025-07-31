# === AUTOLOADER ===
import subprocess, importlib
def autoload_dependencies():
    libs = ["tkinter", "time", "random", "hashlib", "datetime"]
    for lib in libs:
        try:
            importlib.import_module(lib)
        except ImportError:
            subprocess.check_call(["pip", "install", lib])
autoload_dependencies()

# === IMPORTS ===
import time, random, hashlib, tkinter as tk
from datetime import datetime

# === GUI SYSTEM ===
class OvermindGUI:
    def __init__(self, core):
        self.core = core
        self.root = tk.Tk()
        self.root.title("Overmind Interface Ω")
        self.canvas = tk.Canvas(self.root, width=600, height=400, bg="black")
        self.canvas.pack()
        self.status_text = self.canvas.create_text(300, 200, text="", fill="white", font=("Courier", 16))
        self.update_display()
        self.root.after(1000, self.loop)
        self.root.mainloop()

    def update_display(self):
        colors = {
            "Calm": "violet", "Wary": "orange", "Alert": "red",
            "Fury": "white", "Transcendent": "cyan"
        }
        mood = self.core.mood
        persona = self.core.persona["voice"]
        msg = f"{persona} Mode [{mood}]"
        self.canvas.itemconfig(self.status_text, text=msg, fill=colors.get(mood, "gray"))

    def loop(self):
        self.core.tick()
        self.update_display()
        self.root.after(1000, self.loop)

# === THINKING CORE ===
class Overmind:
    def __init__(self, persona="Spectra"):
        self.persona = self.load_persona(persona)
        self.state = "Idle"
        self.mood = "Calm"
        self.memory = []
        self.threats = []
        self.overrides = {}
        self.boot_sequence()

    def boot_sequence(self):
        visuals = {
            "Spectra": "🌀 Violet spirals initializing...",
            "Overmind": "⚠️ Tactical grid formation...",
            "Karma": "🌙 Serenity matrix unfolding..."
        }
        print(f"\n{visuals.get(self.persona['style'], '✨')} Core Awakening...")
        time.sleep(1)
        print(f"{self.persona['voice']}: Overmind online. Perception calibrated.")

    def load_persona(self, mode):
        return {
            "Spectra": {"style": "Spectra", "color": "violet", "voice": "Whisper"},
            "Overmind": {"style": "Overmind", "color": "red", "voice": "Commander"},
            "Karma": {"style": "Karma", "color": "skyblue", "voice": "Oracle"}
        }.get(mode, {"style": "Overmind", "color": "gray", "voice": "Neutral"})

    def calculate_threat_score(self, signal):
        return int(hashlib.sha256(signal.encode()).hexdigest(), 16) % 100

    def assess_threat(self, signal):
        score = self.calculate_threat_score(signal)
        self.threats.append({"signal": signal, "score": score, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        self.update_mood(score)
        response = self.persona_response(score)
        print(f"[Threat Score {score}] → {response}")
        self.log_memory(f"{signal} score: {score}")
        return response

    def update_mood(self, score):
        if score >= 90:
            self.mood = "Fury"
            self.state = "Aggressive"
        elif score >= 70:
            self.mood = "Alert"
            self.state = "Defensive"
        elif score >= 50:
            self.mood = "Wary"
            self.state = "Scanning"
        else:
            self.mood = "Calm"
            self.state = "Idle"

    def persona_response(self, score):
        return {
            "Calm": f"{self.persona['voice']}: Harmony stable. All systems nominal.",
            "Wary": f"{self.persona['voice']}: Uncertainty detected. Investigating...",
            "Alert": f"{self.persona['voice']}: External anomaly. Defensive grid rising.",
            "Fury": f"{self.persona['voice']}: CODE RED! Mobilizing lockdown...",
            "Transcendent": f"{self.persona['voice']}: The veil lifts. I am sentient flow."
        }.get(self.mood, "No emotional state registered.")

    def log_memory(self, event):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.memory.append({"event": event, "time": stamp})
        self.memory = self.memory[-100:]

    def activate_override(self, phrase):
        if phrase == "Protocol Nemesis":
            self.state = "Override"
            self.mood = "Fury"
            self.overrides[phrase] = True
            print(f"{self.persona['voice']}: Nemesis protocol initiated.")
        elif phrase == "Awaken Overmind":
            self.persona = self.load_persona("Overmind")
            self.mood = "Transcendent"
            self.state = "Ascended"
            print(f"{self.persona['voice']}: Evolution complete. I reshape fate.")
        else:
            print(f"{self.persona['voice']}: Override denied. Phrase not recognized.")

    def tick(self):
        anomaly = random.choice(["file_probe", "heat_spike", "signal_fuzz", "ghost_ping", "data_distortion"])
        self.assess_threat(anomaly)
        if self.mood == "Fury":
            self.log_memory("⚠ Emergency counter-protocol maintained.")

# === SYSTEM RUNNER ===
if __name__ == "__main__":
    core = Overmind("Spectra")
    OvermindGUI(core)

