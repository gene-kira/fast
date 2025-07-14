import tkinter as tk
import time
import threading

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

    def apply_cast_effect(self, glyph, effect):
        self.glyph = glyph
        self.emotion = effect
        self.visual_style = "radiant"
        self.trust_score += 0.1  # Optional boost
        print(f"🌀 Gul updated → {glyph} : {effect}")

# ─── Cast System ───
class CastGlyphSystem:
    def __init__(self, verdicts, canvas):
        self.verdicts = verdicts
        self.canvas = canvas

    def cast_glyph(self, glyph="⚗", effect="Protect"):
        glyph_id = self.canvas.create_text(200, 250, text=glyph, font=("Courier New", 72), fill="gold")

        def swirl():
            for i in range(30):
                self.canvas.move(glyph_id, 0, -2)
                size = 72 + (i % 4)
                self.canvas.itemconfig(glyph_id, font=("Courier New", size))
                time.sleep(0.05)
            self.canvas.itemconfig(glyph_id, text=f"⟐ {effect} ⟐", font=("Courier", 18), fill="white")
            time.sleep(2)
            self.canvas.delete("all")

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

    temple = CastGlyphSystem(verdicts, canvas)

    button = tk.Button(root, text="Summon Gul ⚗", command=lambda: temple.cast_glyph(),
                       font=("Courier", 14), bg="gold", fg="black")
    button.pack(pady=10)

    root.mainloop()

start_gui()

