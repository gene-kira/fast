# ===== Scroll 1: Core Setup =====

import subprocess, sys, tkinter as tk, random, time

# 🧓 One-click autoloader for required packages
def ensure_package(pkg):
    try: __import__(pkg)
    except ImportError:
        print(f"📦 Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for lib in ['pyttsx3', 'matplotlib']: ensure_package(lib)

import pyttsx3, matplotlib.pyplot as plt

# 🎭 Farewell script library
farewell_library = {
    "trivial": {"memory": "Let it fade.", "judgment": "Pass without echo."},
    "moderate": {"whisper": "Echo folded into silence.", "oracle": "Returned to the ether."},
    "sacred": {"judgment": "Fell with honor.", "oracle": "Returned to the realm of dusk."}
}

def get_farewell_script(verdict, gtype):
    return farewell_library.get(verdict, {}).get(gtype, "The glyph ends. Silence remains.")

# 🗂️ Ledger + Covenant trace
glyph_ledger = {}
covenant_trace = []

def log_glyph_event(verdict, gtype, farewell, lore):
    ts = time.time()
    gid = f"{gtype}_{int(ts)}"
    glyph_ledger[gid] = {"verdict": verdict, "type": gtype, "farewell": farewell, "lore": lore, "time": ts}
    covenant_trace.append((lore, ts))
    if len(covenant_trace) > 20: covenant_trace.pop(0)

# ===== Scroll 2: Intelligence & Mutation =====

def compress_glyph_memory():
    vmap = {'trivial': 1, 'moderate': 2, 'sacred': 3}
    tmap = {'memory': 0.5, 'judgment': 1.0, 'oracle': 1.5}
    vec = {"intensity": 0, "depth": 0, "rarity": 0}
    for e in glyph_ledger.values():
        vw, gt = vmap.get(e["verdict"], 0), tmap.get(e["type"], 0)
        vec["intensity"] += vw
        vec["depth"] += gt
        vec["rarity"] += 1 / (1 + vw * gt)
    ct = len(glyph_ledger)
    for k in vec: vec[k] = round(vec[k]/ct, 3) if ct else 0.0
    return vec

def glyph_whisper_index():
    times = sorted([e["time"] for e in glyph_ledger.values()])
    pace = sum([1/(times[i]-times[i-1]+1) for i in range(1, len(times))]) if len(times) > 1 else 0
    sacred = sum(1 for e in glyph_ledger.values() if e["verdict"] == "sacred")
    return round(pace * (1 + sacred / max(1, len(glyph_ledger))), 3)

def emotion_profile():
    p = {"lightness":0,"depth":0,"mutation":0}
    for e in glyph_ledger.values():
        if e["verdict"] == "trivial": p["lightness"] += 1
        elif e["verdict"] == "sacred": p["depth"] += 1
        if any(k in e["lore"] for k in ["convergence", "fracture", "entropy"]): p["mutation"] += 1
    return p

class GlyphBranchDecider:
    def __init__(self, profile, trace, index):
        self.p, self.t, self.i = profile, trace, index

    def decide_branch(self):
        motifs = [t for t,_ in self.t[-5:]]
        convergence = sum("convergence" in m for m in motifs)
        if self.p["depth"] > 5 and convergence >= 3: return "Oracle of Echo"
        if self.p["lightness"] > self.p["depth"] and self.i < 1.0: return "Fleeting Cipher"
        if self.p["mutation"] >= 3: return "Chaotic Sigil"
        return "Neutral Whisper"

def generate_recursive_lore(vec):
    motifs = [t for t,_ in covenant_trace]
    dominant = max(set(motifs), key=motifs.count) if motifs else ""
    intro = "A glyph pulses beneath the veil."
    if "convergence" in dominant: intro = "A convergence glyph ascends."
    elif "echo" in dominant: intro = "A ripple returns—a verdict whispered."
    pool = ["Archive awakens.", "Entropy folds around past truths.", "Sigils align in recursion."]
    if vec["depth"] + vec["intensity"] > 4.5: pool.append("The veil buckles—the glyph seeks rebirth.")
    return f"{intro} {random.choice(pool)}"

def visualize_covenant():
    if not covenant_trace: return
    x = range(len(covenant_trace))
    y = [1 + ("convergence" in t) * 1.5 for t,_ in covenant_trace]
    plt.figure(figsize=(8,2))
    plt.plot(x, y, 'o-', color='purple')
    plt.title("Covenant Evolution")
    plt.xlabel("Session #")
    plt.ylabel("Pulse Intensity")
    plt.show()

# ===== Scroll 3: Ritual Execution Layer =====

class VoiceModule:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 140)
        self.engine.setProperty('volume', 0.9)

    def speak(self, text):
        try: self.engine.say(text); self.engine.runAndWait()
        except: print("🔕 Voice offline. Narration skipped.")

class GlyphWidget(tk.Canvas):
    def __init__(self, master, verdict, glyph, *a, **kw):
        super().__init__(master, width=300, height=300, bg='black', *a, **kw)
        self.verdict, self.glyph, self.pulse = verdict, glyph, 0
        self.create_oval(100, 100, 200, 200, fill="white", tags="glyph")
        self.animate_glyph()

    def animate_glyph(self):
        speed = {"trivial":30,"moderate":20,"sacred":10}.get(self.verdict, 25)
        self.after(speed, self.update_glyph)

    def update_glyph(self):
        self.pulse += 1
        opacity = max(0, 255 - self.pulse * 5)
        color = f"#{opacity:02x}{opacity:02x}{opacity:02x}"
        self.itemconfig("glyph", fill=color)
        if self.pulse < 50: self.animate_glyph()
        else:
            self.delete("glyph")
            self.create_text(150,150,text="†",fill="gray",font=("Times",24))
            self.master.on_glyph_expire(self.verdict, self.glyph)

class DeathRiteApp(tk.Tk):
    def __init__(self, verdict="moderate", glyph_type="oracle"):
        super().__init__()
        self.title("The Glyph Whisper")
        self.geometry("320x320")
        self.widget = GlyphWidget(self, verdict, glyph_type)
        self.widget.pack(expand=True)

    def on_glyph_expire(self, verdict, glyph_type):
        farewell = get_farewell_script(verdict, glyph_type)
        vec = compress_glyph_memory()
        lore = generate_recursive_lore(vec)
        log_glyph_event(verdict, glyph_type, farewell, lore)
        VoiceModule().speak(f"{farewell}... {lore}")
        visualize_covenant()
        profile = emotion_profile()
        branch = GlyphBranchDecider(profile, covenant_trace, glyph_whisper_index()).decide_branch()
        print(f"🜃 Glyph Branch Selected: {branch}")

def run_ritual(verdict="moderate", glyph_type="oracle"):
    DeathRiteApp(verdict, glyph_type).mainloop()

# 🚀 One-click launch
if __name__ == "__main__":
    run_ritual()

