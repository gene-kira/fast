# === AUTOLOADER: installs all required packages if missing ===
import importlib
import subprocess
import sys

REQUIRED_LIBRARIES = [
    "numpy"
]

def ensure_dependencies():
    for lib in REQUIRED_LIBRARIES:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"📦 Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

ensure_dependencies()

# === IMPORTS (AFTER LOADING) ===
import numpy as np
from collections import deque, Counter

# === SYMBOLIC COMPONENTS ===

class GlyphMemory:
    def __init__(self, max_len=250):
        self.buffer = deque(maxlen=max_len)

    def log(self, glyph):
        self.buffer.append(glyph)

    def recent(self, n=5):
        return list(self.buffer)[-n:]

class GlyphEssence:
    def __init__(self):
        self.cache = []

    def update(self, glyph):
        self.cache.append(tuple(np.round(glyph, 2)))
        if len(self.cache) > 500:
            self.cache.pop(0)

    def extract(self):
        freq = Counter(self.cache)
        common = [np.array(g) for g, _ in freq.most_common(3)]
        return np.mean(common, axis=0) if common else None

class EmpathyWaveDetector:
    def __init__(self):
        self.valence = deque(maxlen=100)

    def push(self, glyph):
        self.valence.append(np.tanh(np.mean(glyph)))

    def detect(self):
        if len(self.valence) < 10:
            return None
        avg = np.mean(self.valence)
        return "grief" if avg < -0.85 else "awe" if avg > 0.85 else None

class SwarmPersona:
    def __init__(self, name="meta-node"):
        self.memory = deque(maxlen=100)
        self.name = name

    def observe(self, vec):
        self.memory.append(vec)

    def reflect(self):
        if not self.memory: return None
        avg = np.mean(np.stack(self.memory), axis=0)
        tone = np.tanh(np.mean(avg))
        mood = "visionary" if tone > 0.6 else "stoic" if tone < -0.6 else "curious"
        return {"persona": mood, "signature": avg}

# === SYMBOLIC ENGINE ===

def generate_glyph(text, dim=16):
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.uniform(-1, 1, dim)

def ritual_transform(vec):
    return vec * np.sin(np.sum(vec))

def glyph_fuse(stack):
    return np.tanh(np.mean(np.stack(stack), axis=0))

# === MAIN LOOP ===

if __name__ == "__main__":
    memory = GlyphMemory()
    essence = GlyphEssence()
    empath = EmpathyWaveDetector()
    persona = SwarmPersona()

    print("🔁 Real-Time Symbolic Processing Started.\n(Type symbols, enter 'exit' to quit)\n")

    while True:
        user_input = input("⚡ Glyph input: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("🛑 Stopping symbolic engine.")
            break

        glyph = generate_glyph(user_input)
        transformed = ritual_transform(glyph)

        # Update systems
        memory.log(glyph)
        essence.update(glyph)
        empath.push(glyph)
        persona.observe(transformed)

        # Reflect state
        recent = memory.recent()
        fused = glyph_fuse(recent)
        mood = empath.detect()
        identity = persona.reflect()

        print(f"\n🧠 Essence: {np.round(essence.extract(), 2)}")
        print(f"🌊 Mood: {mood}")
        print(f"👁 Persona: {identity['persona']}")
        print(f"🔮 Fused Glyph: {np.round(fused, 2)}\n")

