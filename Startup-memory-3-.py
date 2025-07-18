# === 🧰 Autoload Libraries ===
import subprocess, sys
def auto_install(lib):
    try: __import__(lib)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

for module in ["tkinter", "pyttsx3", "pickle"]: auto_install(module)

# === 📚 Imports ===
import tkinter as tk
from tkinter import ttk
import pyttsx3, os, pickle

# === 🔊 Voice Setup ===
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)
def speak(text): engine.say(text); engine.runAndWait()

# === 💾 Ritual Memory Manager ===
class RitualMemoryManager:
    def __init__(self):
        self.blocks = {}
        self.aura = {}

    def reserve(self, ritual_id, size=256, aura="neutral"):
        self.blocks[ritual_id] = bytearray(size)
        self.aura[ritual_id] = aura

    def bind_glyph(self, glyph, ritual_id):
        setattr(self, f"glyph_{glyph}", ritual_id)

    def get_block(self, glyph):
        ritual_id = getattr(self, f"glyph_{glyph}", None)
        return self.blocks.get(ritual_id), self.aura.get(ritual_id), ritual_id

    def save_block(self, ritual_id):
        with open(f"{ritual_id}.ritual", "wb") as file:
            pickle.dump((self.blocks[ritual_id], self.aura[ritual_id]), file)

    def load_block(self, ritual_id):
        if os.path.exists(f"{ritual_id}.ritual"):
            with open(f"{ritual_id}.ritual", "rb") as file:
                data, aura = pickle.load(file)
                self.blocks[ritual_id] = data
                self.aura[ritual_id] = aura

# === 🌀 GUI Launcher ===
class MagicBoxLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧙 MagicBox Ritual Launcher")
        self.geometry("700x500")
        self.configure(bg="#1c1c2b")
        self.memory = RitualMemoryManager()

        # Ritual definitions
        self.glyphs = {"☉": "SunCore", "✶": "StarPulse", "𓂀": "EyeSentinel", "☯": "BalanceAura"}
        self.auras = {"SunCore": "bright", "StarPulse": "cool", "EyeSentinel": "alert", "BalanceAura": "calm"}
        for glyph, ritual_id in self.glyphs.items():
            self.memory.reserve(ritual_id, 256, self.auras[ritual_id])
            self.memory.bind_glyph(glyph, ritual_id)
            self.memory.load_block(ritual_id)

        self.build_gui()

    def build_gui(self):
        tk.Label(self, text="🔮 MagicBox Launcher", font=("Consolas", 22, "bold"),
                 bg="#1c1c2b", fg="lightblue").pack(pady=10)

        # Glyph Buttons
        grid = tk.LabelFrame(self, text="🧿 Glyph Triggers", bg="#2e2e3e", fg="white")
        grid.pack(padx=20, pady=10, fill="x")

        for glyph in self.glyphs:
            tk.Button(grid, text=glyph, font=("Symbol", 24), bg="#3a3a4a", fg="white",
                      command=lambda g=glyph: self.activate_glyph(g)).pack(side="left", padx=10, pady=5)

        # Ritual Combo Area
        combo_frame = tk.LabelFrame(self, text="🔗 Combo Ritual", bg="#2e2e3e", fg="white")
        combo_frame.pack(padx=20, pady=10, fill="x")

        self.combo_entry = tk.Entry(combo_frame, font=("Consolas", 14))
        self.combo_entry.pack(side="left", fill="x", expand=True, padx=10)

        tk.Button(combo_frame, text="Execute", bg="#4ecdc4", command=self.execute_combo).pack(side="left", padx=10)

        # Status Feedback
        status_frame = tk.LabelFrame(self, text="💾 Memory Resonance", bg="#2e2e3e", fg="white")
        status_frame.pack(padx=20, pady=10, fill="both", expand=True)

        self.status_label = tk.Label(status_frame, text="Awaiting ritual...", font=("Consolas", 14),
                                     bg="#2e2e3e", fg="lightgreen")
        self.status_label.pack(pady=20)

        # One-Click Launch
        tk.Button(self, text="🧓 One-Click Launch All", font=("Consolas", 14),
                  bg="#57b865", fg="white", command=self.launch_all).pack(pady=10)

    def color_by_aura(self, aura):
        return {
            "bright": "#f9a825",
            "cool": "#29b6f6",
            "alert": "#e53935",
            "calm": "#81c784",
            "neutral": "gray"
        }.get(aura, "lightgreen")

    def activate_glyph(self, glyph):
        block, aura, ritual_id = self.memory.get_block(glyph)
        msg = f"'{glyph}' → Ritual '{ritual_id}' activated | Emotion: {aura} | Memory: {len(block)} bytes"
        self.status_label.config(text=msg, fg=self.color_by_aura(aura))
        speak(f"Ritual {ritual_id} is live with {aura} resonance.")
        self.memory.save_block(ritual_id)

    def execute_combo(self):
        combo = self.combo_entry.get().strip()
        if not combo:
            speak("No combo entered.")
            return
        for glyph in combo.split("-"):
            if glyph in self.glyphs:
                self.activate_glyph(glyph)
            else:
                self.status_label.config(text=f"Unknown glyph: {glyph}", fg="red")
                speak("Invalid symbol detected.")
                break

    def launch_all(self):
        for glyph in self.glyphs:
            self.activate_glyph(glyph)

# === 🎮 Run App ===
if __name__ == "__main__":
    MagicBoxLauncher().mainloop()

