# soul_mirror.py

# === Autoloader ===
import importlib
import sys

required_modules = [
    'tkinter', 'pyttsx3', 'sqlite3', 'datetime', 'threading',
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends.backend_tkagg'
]

for module in required_modules:
    try:
        importlib.import_module(module)
    except ImportError:
        print(f"Missing: {module}. Install with: pip install {module}")
        sys.exit(1)

# === Imports ===
import tkinter as tk
from tkinter import messagebox
import pyttsx3
import sqlite3
from datetime import datetime
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === Voice ===
class VoiceNarrator:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 140)

    def speak(self, text):
        threading.Thread(target=self.engine.say, args=(text,)).start()
        self.engine.runAndWait()

# === Emotion Parsing ===
def parse_emotion(text):
    text = text.lower()
    if any(w in text for w in ['happy', 'excited', 'confident']):
        return 'Positive'
    if any(w in text for w in ['angry', 'tired', 'frustrated']):
        return 'Negative'
    if any(w in text for w in ['confused', 'unsure', 'anxious']):
        return 'Neutral'
    return 'Unknown'

# === Snapshot Object ===
class StateSnapshot:
    def __init__(self, desc, score, delta, tags, emotion):
        self.desc = desc
        self.score = score
        self.delta = delta
        self.tags = tags
        self.emotion = emotion

# === Memory ===
class MemoryManager:
    def __init__(self):
        self.conn = sqlite3.connect("soul_memory.db")
        self.cursor = self.conn.cursor()
        self.setup()

    def setup(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            state_desc TEXT,
            verdict_score REAL,
            trust_delta REAL,
            glyph_tags TEXT,
            emotion_valence TEXT
        )''')
        self.conn.commit()

    def save(self, snapshot):
        now = datetime.now().isoformat()
        self.cursor.execute('''INSERT INTO states 
            (timestamp, state_desc, verdict_score, trust_delta, glyph_tags, emotion_valence)
            VALUES (?, ?, ?, ?, ?, ?)''',
            (now, snapshot.desc, snapshot.score, snapshot.delta, ",".join(snapshot.tags), snapshot.emotion))
        self.conn.commit()

    def fetch_all(self):
        self.cursor.execute("SELECT * FROM states ORDER BY id ASC")
        return self.cursor.fetchall()

# === GUI ===
class SoulMirrorGUI:
    def __init__(self, root, memory, voice):
        self.root = root
        self.memory = memory
        self.voice = voice
        self.root.title("🧓 Soul Mirror")
        self.build_ui()

    def build_ui(self):
        tk.Label(self.root, text="Describe Your Thought:", font=('Arial', 14)).pack()
        self.entry = tk.Entry(self.root, width=60, font=('Arial', 14))
        self.entry.pack(pady=5)

        tk.Label(self.root, text="Trust Level:", font=('Arial', 12)).pack()
        self.trust_var = tk.DoubleVar()
        self.trust_slider = tk.Scale(self.root, variable=self.trust_var, from_=-1, to=1, resolution=0.1,
                                     orient=tk.HORIZONTAL, length=300)
        self.trust_slider.pack()

        tk.Label(self.root, text="Verdict Strength:", font=('Arial', 12)).pack()
        self.verdict_var = tk.DoubleVar()
        self.verdict_slider = tk.Scale(self.root, variable=self.verdict_var, from_=0, to=1, resolution=0.1,
                                       orient=tk.HORIZONTAL, length=300)
        self.verdict_slider.pack()

        tk.Label(self.root, text="Glyph Tags (comma-separated):", font=('Arial', 12)).pack()
        self.tag_entry = tk.Entry(self.root, width=60, font=('Arial', 14))
        self.tag_entry.pack(pady=5)

        tk.Button(self.root, text="📝 Save Memory", font=('Arial', 12), command=self.save_snapshot).pack(pady=5)
        tk.Button(self.root, text="🔊 Play Last Memory", font=('Arial', 12), command=self.play_last).pack(pady=5)
        tk.Button(self.root, text="🔄 Refresh Timeline", font=('Arial', 12), command=self.refresh).pack(pady=5)

        self.timeline = tk.Listbox(self.root, width=100, font=('Arial', 12))
        self.timeline.pack(pady=5)

        self.figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack(pady=5)

    def save_snapshot(self):
        desc = self.entry.get().strip()
        score = self.verdict_var.get()
        delta = self.trust_var.get()
        tags = [t.strip() for t in self.tag_entry.get().split(',') if t.strip()]
        emotion = parse_emotion(desc)

        if not desc:
            messagebox.showwarning("Hold On", "Please enter a thought before saving.")
            return

        snap = StateSnapshot(desc, score, delta, tags, emotion)
        self.memory.save(snap)
        self.entry.delete(0, tk.END)
        self.tag_entry.delete(0, tk.END)
        self.refresh()
        self.voice.speak("Memory saved successfully.")

    def refresh(self):
        self.timeline.delete(0, tk.END)
        self.ax.clear()

        states = self.memory.fetch_all()
        trust_vals = []

        for s in states:
            trust_vals.append(s[4])
            line = f"{s[1][-8:]} | {s[2]} | Mood: {s[6]} | Trust: {s[4]} | Tags: {s[5]}"
            self.timeline.insert(tk.END, line)

        if trust_vals:
            self.ax.plot(trust_vals, marker='o', color='darkcyan')
            self.ax.set_title("Trust Curve")
            self.ax.grid(True)
            self.canvas.draw()

    def play_last(self):
        states = self.memory.fetch_all()
        if states:
            last = states[-1]
            msg = f"Last memory was: {last[2]}. Mood was {last[6]}."
            self.voice.speak(msg)
        else:
            self.voice.speak("No memories saved yet.")

# === Launch ===
if __name__ == "__main__":
    root = tk.Tk()
    voice = VoiceNarrator()
    memory = MemoryManager()
    app = SoulMirrorGUI(root, memory, voice)
    root.mainloop()

