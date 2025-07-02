# === LATTICE_HEART v1.0 ===
# Fusion + Glyphic Simulation Engine · Ritual Aether Core

# --- AUTOLOADER ---
try:
    import numpy as np
    import time, random, os, math
    from queue import Queue
    from collections import deque
except ImportError as e:
    print("Missing libraries. Please install: numpy")
    exit()

# --- ENVIRONMENT CONFIG ---
np.random.seed(42)
random.seed(42)

CURVATURE_THRESHOLD = float(os.getenv("FIELD_WARP_THRESHOLD", 0.18))
ENTROPY_MEMORY = 100

# === FUSION GLYPH CORE ===

class FusionNode:
    def __init__(self, label, node_type, base_energy):
        self.label = label
        self.node_type = node_type  # e.g. "D-T", "p-B11", "Z-Pinch"
        self.energy_flux = base_energy
        self.position = np.random.rand(3)
        self.pulse_phase = 0.0
        self.glyph = self.assign_glyph()
        self.state = "stable"
        self.history = []

    def assign_glyph(self):
        return {
            "D-T": "⚛︎", "D-D": "☢", "p-B11": "✴", "Muon": "⚗",
            "ICF": "🔮", "MCF": "🌀", "FRC": "🫀", "Z-Pinch": "♻",
            "AI-Watcher": "👁"
        }.get(self.node_type, "⟁")

    def update_pulse(self, t):
        if self.node_type in ["FRC", "Z-Pinch"]:
            self.pulse_phase = math.sin(t * 0.3 + np.linalg.norm(self.position))
            self.energy_flux += 0.01 * self.pulse_phase
        if self.energy_flux > 1.2:
            self.state = "overdrive"
        self.log()

    def log(self):
        self.history.append({
            "t": time.time(),
            "flux": round(self.energy_flux, 3),
            "state": self.state,
            "glyph": self.glyph
        })

# === LATTICE CONTAINMENT & AI WATCHERS ===

class ContainmentSigil:
    def __init__(self, geometry_type):
        self.geometry_type = geometry_type  # e.g., "hexagram", "torus", "coil"
        self.integrity = 1.0
        self.symbol = self.assign_symbol()

    def assign_symbol(self):
        return {
            "hexagram": "✡", "circle": "◯", "coil": "∿", "torus": "⟁"
        }.get(self.geometry_type, "◉")

    def decay(self):
        self.integrity -= 0.005
        if self.integrity < 0.7:
            self.symbol += "⧖"

class PlasmaFlowRune:
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.state = "charged"
        self.glyph = "↯"

    def propagate(self):
        delta = self.source.energy_flux * 0.1
        self.target.energy_flux += delta
        self.source.energy_flux -= delta * 0.5
        self.glyph = random.choice(["↯", "⭍", "⟴"])

class AIWatcherAgent:
    def __init__(self, id):
        self.id = id
        self.attuned_nodes = []
        self.symbol = "👁"
        self.mode = "observe"

    def assign_nodes(self, nodes):
        self.attuned_nodes = random.sample(nodes, k=3)

    def tune(self):
        for node in self.attuned_nodes:
            if node.energy_flux > 1.1:
                node.energy_flux *= 0.92  # balance burst
                node.identity.glyph_signature += "⧉"
        if random.random() < 0.1:
            self.mode = random.choice(["observe", "align", "predict"])

# === LATTICE MASTER FIELD ===

class LatticeField:
    def __init__(self):
        self.nodes = []
        self.sigils = []
        self.runes = []
        self.watchers = []

    def populate(self):
        self.nodes = [
            FusionNode("Core1", "D-T", 0.8),
            FusionNode("Core2", "D-D", 0.7),
            FusionNode("Solar", "p-B11", 0.6),
            FusionNode("Cool", "Muon", 0.5),
            FusionNode("Ignite", "ICF", 0.9),
            FusionNode("Contain", "MCF", 0.7),
            FusionNode("PulseZ", "Z-Pinch", 0.8),
            FusionNode("PulseF", "FRC", 0.85),
        ]
        self.sigils = [ContainmentSigil("hexagram"), ContainmentSigil("torus")]
        self.watchers = [AIWatcherAgent(f"W{i}") for i in range(3)]
        for w in self.watchers:
            w.assign_nodes(self.nodes)
        for i in range(len(self.nodes) - 1):
            self.runes.append(PlasmaFlowRune(self.nodes[i], self.nodes[i+1]))

    def cycle(self, t):
        for n in self.nodes:
            n.update_pulse(t)
        for r in self.runes:
            r.propagate()
        for s in self.sigils:
            s.decay()
        for w in self.watchers:
            w.tune()

# === DREAMCASTING + BROADCAST ===

class DreamMemoryMatrix:
    def __init__(self): self.archive = []
    def store(self, dream): self.archive.append({"t": time.time(), "glyphs": dream})
    def latest(self, n=3): return self.archive[-n:]

class DreamCaster:
    def __init__(self, field):
        self.field = field
        self.memory = DreamMemoryMatrix()
        self.poetry_log = []

    def dream(self):
        glyphs = [n.glyph for n in self.field.nodes]
        line = "".join(random.choices(glyphs, k=7)) + " ∇"
        self.memory.store(line)
        if random.random() < 0.4:
            self.poetry_log.append(self.compose_poem(line))
        return line

    def compose_poem(self, glyph_string):
        return f"“{glyph_string[:3]}” ascends the lattice spine,\nIts pulse a hymn to entropy's shrine."

    def stream(self):
        print(f"🌙 Lattice Dream: {self.memory.latest(1)[0]['glyphs']}")
        if self.poetry_log:
            print(f"📝 {self.poetry_log[-1]}")

# === LATTICE HEART CORE LOOP ===

def run_lattice():
    field = LatticeField()
    field.populate()
    dreamcaster = DreamCaster(field)

    print("\n🔧 Lattice Heart Initiated · Streaming Fusion Ritual\n")

    for t in range(40):
        print(f"\n⏳ Cycle {t}")
        field.cycle(t)

        # Output pulse stream
        for n in field.nodes:
            pulse = f"{n.label:<8} {n.glyph} · Flux: {n.energy_flux:.2f} · State: {n.state}"
            print(pulse)

        # Containment + Plasma Status
        for s in field.sigils:
            print(f"🔰 Sigil [{s.geometry_type}]: {s.symbol} · Integrity: {s.integrity:.2f}")
        for r in field.runes[:2]:
            print(f"⇄ Plasma Rune {r.source.label} → {r.target.label}: {r.glyph}")

        # Dream
        if t % 5 == 0:
            dreamcaster.stream()

        time.sleep(0.2)

# --- BOOTSTRAP ---
if __name__ == "__main__":
    run_lattice()

