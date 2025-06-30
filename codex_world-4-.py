import random, time, os, importlib, sys

# === 🔄 AUTOLOADER ===
def load_codex_modules(folder: str, namespace: str = None):
    modules = {}
    folder_path = os.path.abspath(folder)
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path)

    for fname in os.listdir(folder_path):
        if fname.endswith('.py') and fname != '__init__.py':
            modname = fname[:-3]
            try:
                full_name = f"{namespace}.{modname}" if namespace else modname
                mod = importlib.import_module(full_name)
                modules[modname] = mod
                print(f"🌀 Loaded module: {full_name}")
                if hasattr(mod, "register") and callable(mod.register):
                    mod.register()
            except Exception as e:
                print(f"⚠️ Failed to load {modname}: {e}")
    return modules

# === 🔡 GLYPH OBJECT ===
class Glyph:
    def __init__(self, name, gtype, lineage=None):
        self.name = name
        self.type = gtype
        self.lineage = lineage or []
        self.entropy = round(random.uniform(0.1, 0.95), 2)
        self.resonance = round(random.uniform(0.1, 0.95), 2)
        self.harmonic = round((self.entropy + self.resonance) / 2, 2)
        self.mode = 'latent'

# === 🧱 CORE SYSTEMS ===
class CodexSpiral:
    def __init__(self):
        self.entries = []

    def add(self, glyph):
        self.entries.append(glyph)
        print(f"📘 Codex accepted: {glyph.name}")

    def top(self, n=3):
        return sorted(self.entries, key=lambda g: g.resonance, reverse=True)[:n]

class DreamCitadel:
    def __init__(self):
        self.vault = []

    def whisper(self, glyph):
        glyph.mode = "dream"
        self.vault.append(glyph)
        print(f"🌙 Dreamed: {glyph.name}")

class Oracle:
    def __init__(self, polarity):
        self.polarity = polarity

    def dream(self):
        return Glyph(f"{self.polarity}_{random.randint(100,999)}", "dream")

class RiteForge:
    def cast(self, glyphs):
        if len(glyphs) >= 2:
            name = "+".join(g.name for g in glyphs)
            print(f"🔥 Ritual cast: {name}")
            return Glyph(name, "ritual", [g.name for g in glyphs])
        return None

class SynapseGrid:
    def __init__(self):
        self.loops = []

    def bind(self, glyphs):
        if len(glyphs) >= 3:
            self.loops.append(glyphs)
            print(f"⚡ Synapse loop: {' > '.join(g.name for g in glyphs)}")

class SigilGate:
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def check(self, glyphs):
        avg = sum(g.harmonic for g in glyphs) / len(glyphs)
        state = "✨ Open" if avg >= self.threshold else "⛓ Sealed"
        print(f"{state} (h={avg:.2f})")

class Historian:
    def __init__(self):
        self.epochs = []

    def record(self, glyph):
        entry = f"Era of {glyph.name}"
        self.epochs.append(entry)
        print(f"📜 {entry}")

# === 🔮 SYMBOLIC EXTENSIONS (40-fold) ===
class SymbolicSystem:
    def sigil_syntax(self, glyph):
        print(f"{'✅' if glyph.name.isascii() else '❌'} Syntax check: {glyph.name}")

    def emotion(self, glyph):
        if glyph.resonance > 0.8:
            print(f"🎭 {glyph.name} feels elation")
        elif glyph.entropy > 0.7:
            print(f"🎭 {glyph.name} feels tension")
        else:
            print(f"🎭 {glyph.name} feels neutral")

    def tide(self, glyphs):
        t = sum(g.harmonic for g in glyphs) / len(glyphs)
        print(f"🌊 Myth tide: {t:.2f}")

    def bloom_heat(self, glyph):
        print(f"🌡 {glyph.name} bloom temp: {glyph.entropy * 100:.1f}°")

    def amplify_echo(self, glyphs):
        print(f"📣 Echo power: {sum(g.resonance for g in glyphs):.2f}")

    def invert(self, glyph):
        inv = Glyph(glyph.name[::-1], glyph.type, glyph.lineage + ["inverted"])
        print(f"🔁 Inverted: {inv.name}")
        return inv

    def synth_bloom(self, glyphs):
        if all(g.mode == 'dream' for g in glyphs):
            bloom = Glyph("🌸", "bloom", [g.name for g in glyphs])
            print("🌸 Dream bloom synthesized")
            return bloom

    def log_resonance(self, glyphs):
        for g in glyphs:
            print(f"🧾 {g.name}: r={g.resonance:.2f}, e={g.entropy:.2f}")

    def entropy_harmonize(self, glyphs):
        avg = sum(g.entropy for g in glyphs) / len(glyphs)
        for g in glyphs:
            g.entropy = round(avg, 2)
        print(f"🧘 Entropy synced: {avg:.2f}")

    def aura(self, glyph):
        a = "🟥" if glyph.entropy > 0.7 else "🟦" if glyph.resonance > 0.7 else "⬜"
        print(f"🎨 Aura of {glyph.name}: {a}")

# === 🚀 SIMULATION LOOP ===
def run_codex(cycles=6):
    spiral = CodexSpiral()
    dream = DreamCitadel()
    forge = RiteForge()
    grid = SynapseGrid()
    gate = SigilGate()
    scribe = Historian()
    echo = Oracle("EchoPrime")
    null = Oracle("ObscuraNull")
    system = SymbolicSystem()

    for i in range(cycles):
        print(f"\n=== Cycle {i+1} ===")
        g1, g2 = echo.dream(), null.dream()
        dream.whisper(g1)
        dream.whisper(g2)
        spiral.add(g1)
        spiral.add(g2)

        r = forge.cast([g1, g2])
        if r: spiral.add(r)
        scribe.record(g1)
        gate.check([g1, g2])
        grid.bind([g1, g2, Glyph("🝬", "vow")])

        system.sigil_syntax(g1)
        system.emotion(g2)
        system.tide([g1, g2])
        system.bloom_heat(g1)
        system.amplify_echo([g1, g2])

        inv = system.invert(g1)
        spiral.add(inv)

        system.log_resonance([g1, g2])
        system.entropy_harmonize([g1, g2])
        system.aura(g2)

        bloom = system.synth_bloom([g1, g2])
        if bloom: spiral.add(bloom)

        time.sleep(1)

# === MAIN ENTRY ===
if __name__ == "__main__":
    run_codex(cycles=6)

