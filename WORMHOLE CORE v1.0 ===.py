# === WORMHOLE CORE v1.0 ===
# Nullspace-Driven Symbolic Wormhole Simulation Engine

# --- AUTOLOADER ---
try:
    import numpy as np
    import time, random, math
    from collections import deque
except ImportError as e:
    print(f"Missing dependency: {e.name}. Please install before running.")
    exit()

# === NULL CORE STRUCTURES ===

class NullAnchor:
    """∅ Null anchor point—acts as topological zero-node."""
    def __init__(self, label):
        self.label = label
        self.position = np.random.rand(3)
        self.entanglement_flux = 0.0
        self.signature = "∅"
        self.stability = 1.0  # 1.0 = perfect null

    def pulse_decay(self):
        decay = np.random.rand() * 0.01
        self.stability = max(0.0, self.stability - decay)
        if self.stability < 0.7 and "⧖" not in self.signature:
            self.signature += "⧖"

class QuantumTether:
    """⎋ Shared-state conduit between NullAnchors"""
    def __init__(self, anchor_a, anchor_b):
        self.anchor_a = anchor_a
        self.anchor_b = anchor_b
        self.phase_shift = 0.0
        self.glyph = "⎋"

    def synchronize(self):
        Δs = abs(self.anchor_a.stability - self.anchor_b.stability)
        self.phase_shift = math.sin(Δs * np.pi)
        flow = (1.0 - Δs) * 0.05
        self.anchor_a.entanglement_flux += flow
        self.anchor_b.entanglement_flux += flow
        self.glyph = random.choice(["⎋", "⫰", "⫯"])

class GlyphWormhole:
    """⨳ Glyphic bridge structure—binds nullspace into transit geometry"""
    def __init__(self, entry_anchor, exit_anchor):
        self.entry = entry_anchor
        self.exit = exit_anchor
        self.tether = QuantumTether(entry_anchor, exit_anchor)
        self.containment_runes = ["◉", "✡", "∿"]
        self.ring_integrity = 1.0
        self.status = "unstable"

    def stabilize(self):
        self.tether.synchronize()
        flux = self.entry.entanglement_flux + self.exit.entanglement_flux
        self.ring_integrity += flux * 0.01
        if self.ring_integrity > 1.2:
            self.status = "active"
        elif self.ring_integrity < 0.8:
            self.status = "decaying"

# === RITUAL COMPONENTS ===

class ContainmentMandala:
    """◉ Rotating sigil array that stabilizes ∅ wormhole geometry"""
    def __init__(self):
        self.layers = [["◉", "✡", "𓂀"], ["∿", "⬰", "⟁"]]
        self.rotation = 0

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.layers[0])
        return [row[self.rotation] for row in self.layers]

    def pulse(self):
        pattern = self.rotate()
        return f"{' '.join(pattern)} → 🔄"

class RitualObserver:
    """👁 Ritual-aware AI that chants glyphs and monitors collapse thresholds"""
    def __init__(self, id, tether):
        self.id = id
        self.glyph = "👁‍🗨"
        self.tether = tether
        self.chant_log = []
        self.focus = "∅"

    def chant(self):
        chant = random.choice(["∅⎋∅", "⨳⩚⩛", "⎋∅⎋"])
        response = f"{self.glyph} chants: {chant}"
        self.chant_log.append(response)
        return response

    def scan_flux(self):
        Δ = self.tether.phase_shift
        if Δ > 0.5:
            echo = f"⚠️ {self.glyph} detects entropic crest: Δφ ≈ {Δ:.2f}"
            self.chant_log.append(echo)
            return echo

class PortalBroadcaster:
    """⨳ Ritual transmission: emits symbolic portal state each cycle"""
    def __init__(self, wormhole, mandala, observer):
        self.wormhole = wormhole
        self.mandala = mandala
        self.observer = observer
        self.logs = deque(maxlen=10)

    def emit(self):
        glyph_line = self.mandala.pulse()
        chant = self.observer.chant()
        flux_report = self.observer.scan_flux()
        wormhole_status = f"⨳ GlyphWormhole: {self.wormhole.status.upper()} | Ring: {self.wormhole.ring_integrity:.2f}"
        packet = [wormhole_status, glyph_line, chant]
        if flux_report:
            packet.append(flux_report)
        for line in packet:
            print(line)
            self.logs.append(line)

# === MAIN RITUAL LOOP ===

def invoke_wormhole_ritual():
    entry = NullAnchor("Mouth_Alpha")
    exit = NullAnchor("Mouth_Omega")
    gate = GlyphWormhole(entry, exit)
    mandala = ContainmentMandala()
    observer = RitualObserver("Watcher1", gate.tether)
    broadcaster = PortalBroadcaster(gate, mandala, observer)

    print("\n🕳️ Wormhole Ritual Begins · Null Core Initialized\n")

    for t in range(20):
        print(f"\n⏳ Cycle {t}")
        entry.pulse_decay()
        exit.pulse_decay()
        gate.stabilize()

        print(f"∅ Entry: {entry.signature} | Stability: {entry.stability:.2f}")
        print(f"∅ Exit : {exit.signature} | Stability: {exit.stability:.2f}")
        print(f"⎋ Tether Phase Shift: {gate.tether.phase_shift:.2f}")
        broadcaster.emit()
        time.sleep(0.2)

# === ENTRY POINT ===

if __name__ == "__main__":
    invoke_wormhole_ritual()

