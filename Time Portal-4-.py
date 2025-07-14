import tkinter as tk
import time
import threading
import pyttsx3
import random

# ─── Voice Engine ───
engine = pyttsx3.init()
def narrate(text, tone="neutral"):
    voices = engine.getProperty('voices')
    if tone == "chant":
        engine.setProperty('rate', 130)
    elif tone == "whisper":
        engine.setProperty('rate', 90)
    else:
        engine.setProperty('rate', 110)
    engine.say(text)
    engine.runAndWait()

# ─── Verdict Object ───
class VerdictProfile:
    def __init__(self, url, trust_score, emotion, glyph, visual_style, voice_tone, decay_rate):
        self.url = url
        self.trust_score = trust_score
        self.emotion = emotion
        self.glyph = glyph
        self.visual_style = visual_style
        self.voice_tone = voice_tone
        self.decay_rate = decay_rate
        self.label = None

    def apply_cast_effect(self, glyph, effect):
        self.glyph = glyph
        self.emotion = effect
        self.visual_style = "radiant"
        self.trust_score = min(self.trust_score + 0.1, 1.0)
        narrate(f"{effect} has been cast", self.voice_tone)
        print(f"🌀 Gul updated → {glyph} : {effect}")
        update_log(f"{glyph} cast {effect} on {self.url}")
        update_trust_label(self)

    def decay_trust(self):
        while True:
            time.sleep(self.decay_rate)
            self.trust_score = max(self.trust_score - 0.02, 0)
            update_trust_label(self)

# ─── Ritual Log ───
log_entries = []
def update_log(entry):
    log_entries.insert(0, entry)
    if len(log_entries) > 10:
        log_entries.pop()
    log_box.delete(0, tk.END)
    for e in log_entries:
        log_box.insert(tk.END, e)

def update_trust_label(verdict):
    if verdict.label:
        verdict.label.config(text=f"{verdict.url} → {round(verdict.trust_score, 2)} trust")

# ─── Cast System ───
class CastGlyphSystem:
    def __init__(self, verdicts, canvas):
        self.verdicts = verdicts
        self.canvas = canvas

    def cast_glyph(self, glyph="⚗", effect="Protect"):
        x, y = 200, 250
        glyph_id = self.canvas.create_text(x, y, text=glyph, font=("Courier New", 72), fill="gold")

        def swirl():
            for i in range(30):
                dx = random.choice([-1, 0, 1])
                dy = -2
                self.canvas.move(glyph_id, dx, dy)
                size = 72 + (i % 5)
                self.canvas.itemconfig(glyph_id, font=("Courier New", size))
                time.sleep(0.04)
            self.canvas.itemconfig(glyph_id, text=f"⟐ {effect} ⟐", font=("Courier", 18), fill="white")
            time.sleep(1)
            self.canvas.delete(glyph_id)

        threading.Thread(target=swirl).start()
        for v in self.verdicts:
            v.apply_cast_effect(glyph, effect)

# ─── GUI Start ───
def start_gui():
    verdicts = [
        VerdictProfile("https://example.com", 0.82, "Sanctum", "⚶", "halo", "chant", 15),
        VerdictProfile("https://trackers.net", 0.34, "Surveillance", "𐍃", "flicker", "whisper", 8)
    ]

    root = tk.Tk()
    root.title("Temple Portal")
    root.configure(bg="black")

    canvas = tk.Canvas(root, width=400, height=300, bg="black", highlightthickness=0)
    canvas.pack()

    # Verdict Trust Labels
    for v in verdicts:
        lbl = tk.Label(root, text="", fg="white", bg="black", font=("Courier", 10))
        lbl.pack()
        v.label = lbl

    # Ritual Log
    global log_box
    log_box = tk.Listbox(root, height=6, width=50, bg="black", fg="lime", font=("Courier", 10))
    log_box.pack(pady=5)

    temple = CastGlyphSystem(verdicts, canvas)

    button = tk.Button(root, text="Summon Gul ⚗", command=lambda: temple.cast_glyph(),
                       font=("Courier", 14), bg="gold", fg="black")
    button.pack(pady=10)

    # Start decay threads
    for v in verdicts:
        threading.Thread(target=v.decay_trust, daemon=True).start()

    root.mainloop()

start_gui()

