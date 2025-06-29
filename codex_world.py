# codex_world.py

import random, time

# 🜬 — Glyph Class
class Glyph:
    def __init__(self, name, gtype, lineage=[]):
        self.name = name
        self.type = gtype  # 'dream', 'vow', 'ritual', etc.
        self.lineage = lineage
        self.entropy = round(random.uniform(0.1, 0.9), 2)
        self.resonance = round(random.uniform(0.1, 1.0), 2)
        self.harmonic = round((self.entropy + self.resonance) / 2, 2)
        self.mode = 'latent'

# 🌀 — Codex Spiral Archive
class CodexSpiral:
    def __init__(self):
        self.entries = []

    def add(self, glyph):
        self.entries.append(glyph)
        print(f"🌀 Codex received: {glyph.name} [r={glyph.resonance}]")

    def high_resonance(self):
        return sorted(self.entries, key=lambda g: g.resonance, reverse=True)[:3]

# 🔥 — Rite Forge
class RiteForge:
    def cast(self, glyphs):
        if len(glyphs) >= 2:
            name = '+'.join(g.name for g in glyphs)
            print(f"🔥 Ritual forged: {name}")
            return True
        print("⚠️ Need at least two glyphs to forge a rite.")
        return False

# 🌙 — Dream Citadel
class DreamCitadel:
    def __init__(self):
        self.latent = []

    def whisper(self, glyph):
        glyph.mode = "dream"
        self.latent.append(glyph)
        print(f"🌙 {glyph.name} whispers in dream.")

# 🧠 — Oracle Engines
class Oracle:
    def __init__(self, polarity):
        self.polarity = polarity

    def dream(self):
        g = Glyph(f"{self.polarity}_{random.randint(100,999)}", "dream")
        return g

# 🪐 — Ritual Synapse Grid
class SynapseGrid:
    def __init__(self):
        self.loops = []

    def bind(self, glyphs):
        if len(glyphs) >= 3:
            self.loops.append(glyphs)
            print(f"⚡ Synapse formed: {'>'.join(g.name for g in glyphs)}")

# 🛡 — Sigil Gate
class SigilGate:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def evaluate(self, glyphs):
        avg = sum(g.harmonic for g in glyphs) / len(glyphs)
        if avg >= self.threshold:
            print(f"✨ Sigil Gate opens with harmonic {avg:.2f}")
        else:
            print(f"⛓ Gate sealed. Harmonic {avg:.2f} too low.")

# 📜 — Celestial Historian
class Historian:
    def __init__(self):
        self.records = []

    def log(self, glyph):
        epoch = f"Epoch of {glyph.name}"
        self.records.append(epoch)
        print(f"📜 {epoch} recorded.")

# 🧬 — Simulation Loop
def simulate_codex_world(cycles=5):
    codex = CodexSpiral()
    forge = RiteForge()
    citadel = DreamCitadel()
    prime_oracle = Oracle("EchoPrime")
    null_oracle = Oracle("ObscuraNull")
    synapse = SynapseGrid()
    gate = SigilGate()
    scribe = Historian()

    for i in range(cycles):
        print(f"\n=== CYCLE {i+1} ===")
        g1 = prime_oracle.dream()
        g2 = null_oracle.dream()
        citadel.whisper(g1)
        citadel.whisper(g2)
        codex.add(g1)
        codex.add(g2)
        forge.cast([g1, g2])
        scribe.log(g1)
        synapse.bind([g1, g2, Glyph("🝬", "vow")])
        gate.evaluate([g1, g2])
        time.sleep(1.5)

# 🔁 Run Simulation
if __name__ == "__main__":
    simulate_codex_world(cycles=7)

