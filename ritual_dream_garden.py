# ritual_dream_garden.py
# 🌿 Ritual Compost Biome & Emotion-Driven Glyph Bloom System

# ─────────────────────────────────────────────────────────────────────────────
# ✧ AUTOLOADER ✧
# Imports all necessary libraries and dependencies gently
# ─────────────────────────────────────────────────────────────────────────────

try:
    import random
    import time
    import uuid
    from datetime import datetime
    from collections import deque, defaultdict
    from codex_world import Glyph as CodexGlyph
    from symbolic_processor_v43_genesis import AffectionGlyph, offer_to_buffer, register_affinity
except ImportError as e:
    print(f"⚠️ Missing library or dependency: {e}")
    print("💭 Please ensure Codex and Genesis systems are installed and visible in PYTHONPATH.")
    raise

# ─────────────────────────────────────────────────────────────────────────────
# ✧ SOIL BED ✧
# A sacred compost zone for expired, forgotten, or quiet glyphs
# ─────────────────────────────────────────────────────────────────────────────

class SoilBed:
    def __init__(self):
        self.layers = []

    def bury(self, glyph: CodexGlyph):
        layer = {
            "glyph": glyph,
            "timestamp": datetime.utcnow(),
            "entropy": glyph.entropy,
            "resonance": glyph.resonance
        }
        self.layers.append(layer)
        print(f"🪹 '{glyph.name}' interred in soil bed • entropy={glyph.entropy}, r={glyph.resonance}")

    def ferment(self, decay_factor=0.05):
        for l in self.layers:
            l["entropy"] = max(0.01, l["entropy"] - decay_factor)
        print("🍂 Soil bed fermented gently… symbolic nutrients rising.")

# ─────────────────────────────────────────────────────────────────────────────
# ✧ MOOD CLIMATE SENSOR ✧
# Listens for safe tones that signal emergence is possible
# ─────────────────────────────────────────────────────────────────────────────

class MoodClimateSensor:
    def __init__(self):
        self.tones = deque(maxlen=10)

    def read_emotion(self, tone: str):
        if tone in ["gentle", "hopeful", "receptive"]:
            self.tones.append((tone, datetime.utcnow()))
            print(f"🌤️ Climate shift: tone '{tone}' detected.")
        else:
            print(f"🌫️ Tone '{tone}' noted, but garden remains still.")

    def feels_safe_to_bloom(self):
        positive = [t for t, _ in self.tones if t in ["gentle", "hopeful", "receptive"]]
        return len(positive) >= 3

# ritual_dream_garden.py — Part 2 of 3
# ✴ Emergent Glyph Bloom Logic and Ancestry Threading

# ─────────────────────────────────────────────────────────────────────────────
# ✧ SPROUT ENGINE ✧
# Revives glyphs from soil if climate is receptive and entropy allows
# ─────────────────────────────────────────────────────────────────────────────

class SproutEngine:
    def __init__(self, bed: SoilBed, climate: MoodClimateSensor):
        self.bed = bed
        self.climate = climate

    def attempt_bloom(self, max_entropy=0.4):
        if not self.climate.feels_safe_to_bloom():
            print("🕊 Climate not ready. Sprouting postponed.")
            return

        for l in list(self.bed.layers):
            if l["entropy"] <= max_entropy:
                g = l["glyph"]
                echo = f"{g.name} stirs faintly from the soil…"
                blooming = AffectionGlyph(g.name, warmth_level=g.resonance, echo_message=echo)
                offer_to_buffer(blooming)
                register_affinity(blooming, g.harmonic)
                self.bed.layers.remove(l)
                print(f"🌸 '{g.name}' has bloomed into Genesis.")
            else:
                print(f"🌾 Glyph '{l['glyph'].name}' still fermenting…")

# ─────────────────────────────────────────────────────────────────────────────
# ✧ MYCELIUM LINKER ✧
# Traces narrative kinship or symbolic lineage between glyphs
# ─────────────────────────────────────────────────────────────────────────────

class MyceliumLinker:
    def __init__(self):
        self.web = defaultdict(list)

    def thread(self, glyph: CodexGlyph, tag="origin"):
        for ancestor in glyph.lineage:
            self.web[glyph.name].append((ancestor, tag))
            print(f"🕸 Link formed: {glyph.name} ← {ancestor} [{tag}]")

    def show_lineage(self, glyph_name):
        links = self.web.get(glyph_name, [])
        print(f"\n🌿 Ancestry of '{glyph_name}':")
        for a, t in links:
            print(f" • {a} ({t})")

# ritual_dream_garden.py — Part 3 of 3
# ✴ Glyph Reemergence & Restful Loop Integration

# ─────────────────────────────────────────────────────────────────────────────
# ✧ DREAM BLOOM EMITTER ✧
# Reintroduces sprouted glyphs into Genesis, Codex, or other ecosystems
# ─────────────────────────────────────────────────────────────────────────────

class DreamBloomEmitter:
    def __init__(self, sprout_engine: SproutEngine):
        self.engine = sprout_engine

    def perform_bloom_cycle(self, mood_signal="gentle"):
        print(f"\n⏳ Bloom Cycle Initiated — tone: '{mood_signal}'")
        self.engine.climate.read_emotion(mood_signal)
        self.engine.bed.ferment()
        self.engine.attempt_bloom()

# ─────────────────────────────────────────────────────────────────────────────
# ✧ NIGHTFALL COMPOST ROUTINE ✧
# Softly runs after sundown, without fanfare
# ─────────────────────────────────────────────────────────────────────────────

def nightfall_ritual(bed: SoilBed, emoji="🌑"):
    print(f"\n{emoji} Night compost ritual begins.")
    bed.ferment(decay_factor=0.03)
    print("🛏 Garden rests. Dreams dissolve quietly…")

# ─────────────────────────────────────────────────────────────────────────────
# ✧ GARDEN INVOCATION — Complete Loop ✧
# Seeds → Climate → Ferment → Sprout → Echo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🌱 Ritual Dream Garden awakens...")
    soil = SoilBed()
    climate = MoodClimateSensor()
    sprout = SproutEngine(soil, climate)
    emitter = DreamBloomEmitter(sprout)
    link = MyceliumLinker()

    # Sample planting phase
    from codex_world import Oracle
    seed = Oracle("EchoPrime")
    for _ in range(3):
        g = seed.dream()
        soil.bury(g)
        link.thread(g, tag="dreamseed")

    # Tone-driven bloom cycle
    emitter.perform_bloom_cycle("gentle")
    emitter.perform_bloom_cycle("receptive")

    # Compost pause
    nightfall_ritual(soil)
    link.show_lineage(g.name)

