# ===🧙 LIBRARY AUTOINSTALL ===
import subprocess
import sys

def ensure_lib(lib):
    try:
        __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

for lib in ["tkinter", "pyttsx3", "pickle"]:
    ensure_lib(lib)

# ===📚 IMPORTS ===
import tkinter as tk
from tkinter import messagebox, ttk
import pyttsx3
import pickle
import os

# ===🧬 EMOTIONAL VOICE ENGINE SETUP ===
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ===🎴 Ritual Memory Manager ===
class RitualMemoryManager:
    def __init__(self):
        self.blocks = {}
        self.glyph_map = {}

    def reserve(self, ritual_id, size):
        self.blocks[ritual_id] = bytearray(size)

    def bind(self, glyph, ritual_id):
        self.glyph_map[glyph] = ritual_id

    def get_block(self, glyph):
        ritual_id = self.glyph_map.get(glyph)
        return self.blocks.get(ritual_id)

    def save(self, ritual_id, path):
        with open(path, "wb") as f:
            pickle.dump(self.blocks[ritual_id], f)

    def load(self, ritual_id, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.blocks[ritual_id] = pickle.load(f)

# ===🧭 MagicBox GUI ===
class MagicBoxGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧙 MagicBox Ritual Launcher")
        self.geometry("680x480")
        self.configure(bg="#1b1b2a")
        self.resonance_colors = {"active": "#4ecdc4", "idle": "#a9a9a9", "error": "#e63946"}
        self.memory = RitualMemoryManager()

        # Setup rituals
        self.glyphs = {"☉": "SunGate", "✶": "StarChime", "𓂀": "EyeWatch", "☯": "BalanceMode"}
        for g, r in self.glyphs.items():
            self.memory.reserve(r, 256)
            self.memory.bind(g, r)

        self.create_widgets()
        self.load_memory()

    def create_widgets(self):
        # Title
        tk.Label(self, text="MagicBox Ritual Launcher", font=("Consolas", 20, "bold"),
                 bg="#1b1b2a", fg="lightblue").pack(pady=10)

        # Glyph Grid
        grid = tk.LabelFrame(self, text="Glyphs", bg="#2c2c3b", fg="white")
        grid.pack(padx=20, pady=10, fill="x")

        for glyph in self.glyphs:
            btn = tk.Button(grid, text=glyph, font=("Symbol", 24), bg="#3a3a4a", fg="white",
                            command=lambda g=glyph: self.trigger_glyph(g))
            btn.pack(side="left", padx=10, pady=5)
            self.create_tooltip(btn, f"Trigger ritual: {self.glyphs[glyph]}")

        # Ritual Sequence Entry
        combo_frame = tk.LabelFrame(self, text="Combo Sequence", bg="#2c2c3b", fg="white")
        combo_frame.pack(padx=20, pady=10, fill="x")

        self.combo_entry = tk.Entry(combo_frame, font=("Consolas", 14))
        self.combo_entry.pack(side="left", padx=10, fill="x", expand=True)

        execute_btn = tk.Button(combo_frame, text="Execute", bg="#4ecdc4", fg="black",
                                command=self.execute_combo)
        execute_btn.pack(side="left", padx=10)

        # Status Display
        status = tk.LabelFrame(self, text="Memory Resonance", bg="#2c2c3b", fg="white")
        status.pack(padx=20, pady=10, fill="both", expand=True)

        self.status_label = tk.Label(status, text="Awaiting glyph...", bg="#2c2c3b",
                                     fg="lightgreen", font=("Consolas", 14))
        self.status_label.pack(pady=20)

        # One-click launcher
        launch_btn = tk.Button(self, text="🧓 One-Click Launch Rituals", font=("Consolas", 14),
                               bg="#5dbb63", fg="white", command=self.one_click_launch)
        launch_btn.pack(pady=10)

    def trigger_glyph(self, glyph):
        ritual_id = self.glyphs[glyph]
        block = self.memory.get_block(glyph)
        msg = f"Glyph '{glyph}' activated: ritual '{ritual_id}' | Memory size: {len(block)} bytes"
        self.status_label.config(text=msg, fg=self.resonance_colors["active"])
        speak(f"{ritual_id} ritual has begun.")
        self.memory.save(ritual_id, f"{ritual_id}.dat")

    def execute_combo(self):
        combo = self.combo_entry.get().strip()
        parts = combo.split("-")
        self.status_label.config(text=f"Combo ritual started: {combo}", fg=self.resonance_colors["active"])
        for glyph in parts:
            if glyph in self.glyphs:
                self.trigger_glyph(glyph)
            else:
                self.status_label.config(text=f"Unknown glyph in combo: {glyph}", fg=self.resonance_colors["error"])
                speak("Unrecognized glyph in combo.")
                break

    def one_click_launch(self):
        self.status_label.config(text="Launching all default rituals...", fg=self.resonance_colors["active"])
        speak("Launching MagicBox rituals now.")
        for glyph in self.glyphs:
            self.trigger_glyph(glyph)

    def load_memory(self):
        for ritual_id in self.glyphs.values():
            path = f"{ritual_id}.dat"
            self.memory.load(ritual_id, path)

    # Tooltip helper
    def create_tooltip(self, widget, text):
        def on_enter(e):
            self.status_label.config(text=text, fg=self.resonance_colors["idle"])
        def on_leave(e):
            self.status_label.config(text="Ready.", fg="lightgreen")
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

# ===🧠 MAIN ===
if __name__ == "__main__":
    app = MagicBoxGUI()
    app.mainloop()

