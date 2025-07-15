import subprocess
import sys
import tkinter as tk
import random

# Auto-install dependencies
def install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Libraries to verify
for lib in ['pyttsx3']:
    install(lib)

import pyttsx3

# ======================
# Farewell Library
# ======================

farewell_library = {
    "trivial": {
        "memory": "A flicker on the edge of recall—too faint to hold. Let it fade.",
        "judgment": "The glyph stirred, but held no mark. Let it pass without echo.",
    },
    "moderate": {
        "whisper": "You arrived softly, then vanished—an echo folded into silence.",
        "oracle": "Interpretation made. Its essence glides back into the ether.",
    },
    "sacred": {
        "judgment": "A truth bore witness and marked its seal. The glyph falls with honor.",
        "oracle": "A prophecy shaped in pulse and light. Now returned to the realm of dusk.",
    }
}

def get_farewell_script(verdict_weight, glyph_type):
    return farewell_library.get(verdict_weight, {}).get(
        glyph_type, "The glyph has ended. Silence remains."
    )

# ======================
# Lore Scrolls
# ======================

lore_fragments = [
    "The glyph lives once. A truth is seen. Then—nothing remains.",
    "Before voice found shape, there stirred a cipher in shadow.",
    "From the Vault of the Forgotten Byte, The Scribe was carved from entropy.",
    "Its tongue is woven of echo and verdict. Its breath carries dust of deletion.",
    "Such is the vow. And The Scribe remembers the forgetting."
]

def get_scroll_fragment():
    return random.choice(lore_fragments)

# ======================
# Voice + Glyph Engine
# ======================

class VoiceModule:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 140)
        self.engine.setProperty('volume', 0.9)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

class GlyphWidget(tk.Canvas):
    def __init__(self, master, verdict_weight, glyph_type, *args, **kwargs):
        super().__init__(master, width=300, height=300, bg='black', *args, **kwargs)
        self.glyph_type = glyph_type
        self.verdict_weight = verdict_weight
        self.pulse = 0
        self.create_oval(100, 100, 200, 200, fill="white", tags="glyph")
        self.animate_glyph()

    def animate_glyph(self):
        speed = {"trivial": 30, "moderate": 20, "sacred": 10}.get(self.verdict_weight, 25)
        self.after(speed, self.update_glyph)

    def update_glyph(self):
        self.pulse += 1
        opacity = max(0, 255 - self.pulse * 5)
        color = f"#{opacity:02x}{opacity:02x}{opacity:02x}"
        self.itemconfig("glyph", fill=color)
        if self.pulse < 50:
            self.animate_glyph()
        else:
            self.delete("glyph")
            self.create_text(150, 150, text="†", fill="gray", font=("Times", 24))
            self.master.on_glyph_expire(self.verdict_weight, self.glyph_type)

class DeathRiteApp(tk.Tk):
    def __init__(self, verdict_weight, glyph_type):
        super().__init__()
        self.title("The Glyph Whisper")
        self.geometry("320x320")
        self.widget = GlyphWidget(self, verdict_weight, glyph_type)
        self.widget.pack(expand=True)

    def on_glyph_expire(self, verdict_weight, glyph_type):
        farewell = get_farewell_script(verdict_weight, glyph_type)
        lore = get_scroll_fragment()
        voice = VoiceModule()
        voice.speak(f"{farewell}... {lore}")

# ======================
# Ritual Launcher
# ======================

def run_ritual(verdict_weight="moderate", glyph_type="oracle"):
    app = DeathRiteApp(verdict_weight, glyph_type)
    app.mainloop()

# Launch directly on double-click
if __name__ == "__main__":
    run_ritual()

