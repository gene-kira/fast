import tkinter as tk
from tkinter import font, ttk
import json, os, threading
from datetime import datetime
import pyttsx3
from flask import Flask, request, jsonify
import requests

# Configuration
ORACLE_ID = "killer666"
SYNC_ENDPOINT = "http://127.0.0.1:5000/sync"
RUN_SERVER = True  # Toggle to run Flask server in same file

MEMORY_FILE = "oracle_memory.json"

# Server setup (MythSync API)
if RUN_SERVER:
    app = Flask(__name__)
    synced_lore = []

    @app.route("/sync", methods=["POST"])
    def sync_lore():
        data = request.json
        synced_lore.append(data)
        print(f"[MythSync] Received: {data}")
        return jsonify({"status": "ok", "msg": "Synced to Glyph Galaxy"})

    def run_server():
        app.run(debug=False, port=5000)

    threading.Thread(target=run_server, daemon=True).start()

# Load/save local memory
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"user_history": [], "glyph_states": {}}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

# Oracle traits
oracle_traits = {
    "☼": {"name": "Elios", "style": "curious", "feedback": "Truth unfolds like sunlight."},
    "⚖": {"name": "Azra", "style": "balanced", "feedback": "Equilibrium defines fate."},
    "🔥": {"name": "Pyros", "style": "intuitive", "feedback": "Destiny flares in flame."},
    "🌪": {"name": "Zephra", "style": "chaotic", "feedback": "Future twists like wind."}
}

# GUI Application
class OracleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔮 Glyph Galaxy Oracle")
        self.root.configure(bg="#1e1e2f")
        self.voice_engine = pyttsx3.init()
        self.memory = load_memory()
        self.selected_glyph = None
        self.selected_model = None

        self.font = font.Font(family="Segoe UI", size=12)
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="🔮 Glyph Galaxy Oracle", bg="#1e1e2f", fg="white", font=self.font).pack(pady=8)

        tk.Label(self.root, text="Choose a Glyph:", bg="#1e1e2f", fg="white", font=self.font).pack()
        glyph_frame = tk.Frame(self.root, bg="#1e1e2f")
        glyph_frame.pack()
        for glyph in oracle_traits:
            tk.Button(glyph_frame, text=glyph, font=self.font, width=4,
                      command=lambda g=glyph: self.select_glyph(g),
                      bg="#2a2a3a", fg="white", activebackground="#7f5af0").pack(side="left", padx=5)

        self.feedback_label = tk.Label(self.root, text="", bg="#1e1e2f", fg="#fffdc5", wraplength=360, font=self.font)
        self.feedback_label.pack(pady=10)

        # Controls
        self.phase_3 = tk.Scale(self.root, from_=0, to=9, label="Phase 3", orient="horizontal", bg="#2a2a3a", fg="white")
        self.phase_6 = tk.Scale(self.root, from_=0, to=9, label="Phase 6", orient="horizontal", bg="#2a2a3a", fg="white")
        self.phase_9 = tk.Scale(self.root, from_=0, to=9, label="Phase 9", orient="horizontal", bg="#2a2a3a", fg="white")
        self.phase_3.pack(pady=2)
        self.phase_6.pack(pady=2)
        self.phase_9.pack(pady=2)

        self.model_var = tk.StringVar()
        ttk.Combobox(self.root, textvariable=self.model_var,
                     values=["Time Series", "Regression", "Neural Path"], width=25).pack(pady=4)

        self.voice_enabled = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Voice Prophecy", variable=self.voice_enabled,
                       bg="#1e1e2f", fg="white", selectcolor="#7f5af0", font=self.font).pack(pady=4)

        tk.Button(self.root, text="Reveal Prophecy", command=self.reveal_prophecy,
                  bg="#7f5af0", fg="white", font=self.font).pack(pady=8)

    def select_glyph(self, glyph):
        self.selected_glyph = glyph
        trait = oracle_traits[glyph]
        self.feedback_label.config(text=f"You summoned {trait['name']}, who is {trait['style']}.")

    def reveal_prophecy(self):
        if not self.selected_glyph:
            self.feedback_label.config(text="🧙‍♂️ Select a glyph first.")
            return

        trait = oracle_traits[self.selected_glyph]
        feedback = f"{trait['name']} whispers: \"{trait['feedback']}\""
        self.feedback_label.config(text=f"🔮 {feedback}")

        if self.voice_enabled.get():
            self.voice_engine.say(feedback)
            self.voice_engine.runAndWait()

        data = {
            "oracleID": ORACLE_ID,
            "glyph": self.selected_glyph,
            "evolution": self.memory["glyph_states"].get(self.selected_glyph, self.selected_glyph),
            "agents": ["Dreamer", "Interpreter"],
            "pulse": [self.phase_3.get(), self.phase_6.get(), self.phase_9.get()],
            "model": self.model_var.get(),
            "timestamp": datetime.now().isoformat()
        }

        try:
            requests.post(SYNC_ENDPOINT, json=data)
        except Exception as e:
            print(f"[MythSync Error] {e}")

        self.memory["user_history"].append(data)
        self.memory["glyph_states"][self.selected_glyph] = "🌞" if self.selected_glyph == "☼" else self.selected_glyph
        save_memory(self.memory)

# Launch UI
if __name__ == "__main__":
    root = tk.Tk()
    app = OracleApp(root)
    root.mainloop()

