# =============================
# Glyph Whisper - Part 1: Core Ritual Framework
# =============================

import subprocess
import sys
import tkinter as tk
import random
import time
import pyttsx3
import matplotlib.pyplot as plt

# 🔧 Auto-install missing libraries
def install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        globals()[package] = __import__(package)

for lib in ['pyttsx3', 'matplotlib']:
    install(lib)

# 🎭 Farewell Script Library
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
    return farewell_library.get(verdict_weight, {}).get(glyph_type, "The glyph has ended. Silence remains.")

# 🗂️ Covenant Trace + Shadow Ledger
glyph_ledger = {}
covenant_trace = []

def log_glyph_event(verdict, gtype, farewell, lore):
    ts = time.time()
    gid = f"{gtype}_{int(ts)}"
    glyph_ledger[gid] = {
        "verdict": verdict, "type": gtype,
        "farewell": farewell, "lore": lore,
        "time": ts
    }
    covenant_trace.append((lore, ts))
    if len(covenant_trace) > 20:
        covenant_trace.pop(0)

# =============================
# Glyph Whisper - Part 2: Emotional Intelligence & Symbolic Evolution
# =============================

# 🎯 Memory Compressor - Generates emotion vector from ledger
def compress_glyph_memory():
    verdict_map = {'trivial': 1, 'moderate': 2, 'sacred': 3}
    glyph_tones = {'memory': 0.5, 'judgment': 1.0, 'oracle': 1.5}
    vec = {"intensity": 0.0, "depth": 0.0, "rarity": 0.0}
    for e in glyph_ledger.values():
        vw = verdict_map.get(e["verdict"], 0)
        gt = glyph_tones.get(e["type"], 0)
        vec["intensity"] += vw
        vec["depth"] += gt
        vec["rarity"] += 1 / (1 + vw * gt)
    count = len(glyph_ledger)
    for k in vec: vec[k] = round(vec[k]/count, 3) if count else 0.0
    return vec

# 🌫️ Whisper Index - Tracks pacing and sacred bias
def glyph_whisper_index():
    timestamps = sorted([e["time"] for e in glyph_ledger.values()])
    pace = sum([1/(timestamps[i]-timestamps[i-1]+1) for i in range(1,len(timestamps))])
    sacreds = sum([1 for e in glyph_ledger.values() if e["verdict"]=="sacred"])
    bias = sacreds / max(1,len(glyph_ledger))
    return round(pace * (1 + bias), 3)

# 🔮 Emotional Profile Extraction
def emotion_profile():
    profile = {"lightness":0,"depth":0,"mutation":0}
    for e in glyph_ledger.values():
        if e["verdict"]=="trivial": profile["lightness"]+=1
        elif e["verdict"]=="sacred": profile["depth"]+=1
        if any(k in e["lore"] for k in ["convergence","fracture","entropy"]): profile["mutation"]+=1
    return profile

# 🜂 Branch Decider - Selects glyph archetype
class GlyphBranchDecider:
    def __init__(self, profile, trace, index):
        self.ep = profile
        self.trace = trace
        self.index = index

    def decide_branch(self):
        recent = [t for t,_ in self.trace[-5:]]
        convergence = sum(["convergence" in t for t in recent])
        if self.ep["depth"] > 5 and convergence >= 3:
            return "Oracle of Echo"
        elif self.ep["lightness"] > self.ep["depth"] and self.index < 1.0:
            return "Fleeting Cipher"
        elif self.ep["mutation"] >= 3:
            return "Chaotic Sigil"
        return "Neutral Whisper"

# 🌌 Recursive Lore Generator - Adapts to covenant mood
def generate_recursive_lore(vector):
    motifs = [t for t,_ in covenant_trace]
    dominant = max(set(motifs), key=motifs.count) if motifs else ""
    intro = "A glyph pulses faintly beneath the veil."
    if "convergence" in dominant: intro = "A convergence glyph ascends with shimmering echoes."
    elif "echo" in dominant: intro = "A ripple marks the return—a verdict once whispered."
    pool = [
        "The Archive awakens, stirred by familiar tones.",
        "Entropy folds around the echo of past truths.",
        "Sigils unbound align in forgotten recursion."
    ]
    if vector["depth"] + vector["intensity"] > 4.5:
        pool.append("The veil buckles—the glyph seeks rebirth.")
    return f"{intro} {random.choice(pool)}"

# 📈 Covenant Trace Visualizer
def visualize_covenant():
    if not covenant_trace: return
    x = range(len(covenant_trace))
    y = [1 + ("convergence" in t)*1.5 for t,_ in covenant_trace]
    plt.figure(figsize=(8,2))
    plt.plot(x, y, 'o-', color='purple')
    plt.title("Covenant Evolution")
    plt.xlabel("Session #")
    plt.ylabel("Glyph Pulse")
    plt.show()

# =============================
# Glyph Whisper - Part 3: GUI, Voice, and Ritual Launcher
# =============================

# 🔈 Voice Engine
class VoiceModule:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 140)
        self.engine.setProperty('volume', 0.9)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

# 🌒 Glyph Animation Engine
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

# 🪄 Ritual Window + Engine
class DeathRiteApp(tk.Tk):
    def __init__(self, verdict_weight, glyph_type):
        super().__init__()
        self.title("The Glyph Whisper")
        self.geometry("320x320")
        self.widget = GlyphWidget(self, verdict_weight, glyph_type)
        self.widget.pack(expand=True)

    def on_glyph_expire(self, verdict_weight, glyph_type):
        farewell = get_farewell_script(verdict_weight, glyph_type)
        vector = compress_glyph_memory()
        lore = generate_recursive_lore(vector)
        log_glyph_event(verdict_weight, glyph_type, farewell, lore)
        VoiceModule().speak(f"{farewell}... {lore}")
        visualize_covenant()
        profile = emotion_profile()
        branch = GlyphBranchDecider(profile, covenant_trace, glyph_whisper_index()).decide_branch()
        print(f"Glyph Branch Selected: {branch}")

# 🎬 One-Click Launcher
def run_ritual(verdict_weight="moderate", glyph_type="oracle"):
    DeathRiteApp(verdict_weight, glyph_type).mainloop()

# 🚪 Direct Launch
if __name__ == "__main__":
    run_ritual()

