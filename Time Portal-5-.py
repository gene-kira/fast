# ─── Autoloader ───
import subprocess
import sys
required = ["pyttsx3"]
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ─── Imports ───
import tkinter as tk
import threading
import pyttsx3
import time
import random
from datetime import datetime

# ─── Voice Narration ───
engine = pyttsx3.init()
def narrate(message, tone="neutral"):
    rate = {"chant": 130, "whisper": 90, "neutral": 110}.get(tone, 110)
    engine.setProperty("rate", rate)
    engine.say(message)
    engine.runAndWait()

# ─── Verdict Profile ───
class VerdictProfile:
    def __init__(self, url, trust_score, emotion, glyph, voice_tone, decay_rate):
        self.url = url
        self.trust_score = trust_score
        self.emotion = emotion
        self.glyph = glyph
        self.voice_tone = voice_tone
        self.decay_rate = decay_rate
        self.label = None

    def apply_cast(self, glyph, effect):
        self.glyph = glyph
        self.emotion = effect
        self.trust_score = min(self.trust_score + 0.1, 1.0)
        narrate(f"{effect} cast on {self.url}", self.voice_tone)
        log_cast(f"{glyph} → {effect} | {self.url}")
        update_label(self)

    def decay_trust(self):
        while True:
            time.sleep(self.decay_rate)
            self.trust_score = max(self.trust_score - 0.02, 0)
            update_label(self)

# ─── Ritual Log ───
log_history = []
def log_cast(text):
    time_stamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{time_stamp}] {text}"
    log_history.insert(0, entry)
    if len(log_history) > 12:
        log_history.pop()
    log_box.delete(0, tk.END)
    for e in log_history:
        log_box.insert(tk.END, e)

def update_label(v):
    if v.label:
        v.label.config(text=f"{v.url} | Trust: {round(v.trust_score, 2)}")

# ─── Glyph Casting Engine ───
class GlyphCaster:
    def __init__(self, verdicts, canvas):
        self.verdicts = verdicts
        self.canvas = canvas

    def cast(self, glyph="⚗", effect="Protect"):
        x, y = 200, 250
        glyph_id = self.canvas.create_text(x, y, text=glyph, font=("Courier New", 72), fill="gold")

        def animate():
            for i in range(30):
                self.canvas.move(glyph_id, random.choice([-1, 0, 1]), -2)
                self.canvas.itemconfig(glyph_id, font=("Courier New", 72 + (i % 5)))
                time.sleep(0.04)
            self.canvas.itemconfig(glyph_id, text=f"⟐ {effect} ⟐", font=("Courier", 18), fill="white")
            time.sleep(1)
            self.canvas.delete(glyph_id)

        threading.Thread(target=animate).start()
        for v in self.verdicts:
            v.apply_cast(glyph, effect)

# ─── GUI Setup ───
def start_portal():
    verdicts = [
        VerdictProfile("https://example.com", 0.82, "Sanctum", "⚶", "chant", 15),
        VerdictProfile("https://trackers.net", 0.34, "Surveillance", "𐍃", "whisper", 8)
    ]

    root = tk.Tk()
    root.title("🌀 Glyph Portal")
    root.configure(bg="black")

    canvas = tk.Canvas(root, width=400, height=300, bg="black", highlightthickness=0)
    canvas.pack()

    for v in verdicts:
        lbl = tk.Label(root, text="", fg="white", bg="black", font=("Courier", 11))
        lbl.pack()
        v.label = lbl

    global log_box
    log_box = tk.Listbox(root, height=6, width=50, bg="black", fg="lime", font=("Courier", 10))
    log_box.pack(pady=5)

    temple = GlyphCaster(verdicts, canvas)

    button = tk.Button(root, text="🔮 Cast Glyph", command=lambda: temple.cast(),
                       font=("Courier", 16), bg="gold", fg="black", height=2, width=20)
    button.pack(pady=10)

    for v in verdicts:
        threading.Thread(target=v.decay_trust, daemon=True).start()

    root.mainloop()

start_portal()

