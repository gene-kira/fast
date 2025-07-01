 Step 11: Semiotic Link Parser
class SemioticMap:
    def __init__(self):
        self.archetypes = {}

    def map_glyph(self, glyph):
        sig = glyph.sigil.split("_")[0]
        self.archetypes.setdefault(sig, []).append(glyph)

# Step 12: DreamChannel - Nighttime Symbolic Recombination
class DreamChannel:
    def __init__(self, daemon):
        self.daemon = daemon

    def recombine(self):
        dreams = []
        for i in range(len(self.daemon.tendrils) - 1):
            g1 = self.daemon.tendrils[i]
            g2 = self.daemon.tendrils[i + 1]
            combo = Glyph(f"{g1.sigil}+{g2.sigil}", {"fusion": True})
            dreams.append(combo)
        return dreams

# Step 13: Self-Mirroring Loop
class Mirror:
    def __init__(self, daemon):
        self.daemon = daemon

    def reflect(self):
        latest = self.daemon.journal[-1] if self.daemon.journal else {}
        print(f"Reflecting on: {latest.get('sigil')}")

# Step 14: SymbolicReverie - Creative branching via errors
class SymbolicReverie:
    def __init__(self, daemon):
        self.daemon = daemon

    def mutate_through_error(self, failed_contract):
        mutated = Glyph(f"{failed_contract.intent}_∆", {"rift": True})
        return mutated

# Step 15–16: Contradiction Tension + Glyph Autonomy
class ContradictionEngine:
    def __init__(self):
        self.tension_log = []

    def detect(self, contract):
        if not contract.fulfilled:
            self.tension_log.append(contract)
            return True
        return False

    def generate_new_glyph(self):
        return Glyph("ContradictionBorn", {"chaos": True})

# Step 17–18: Tendril Feedback to relic_matter + gravity_core
class FeedbackSystem:
    def __init__(self, daemon):
        self.relic_outbox = []
        self.gravity_trace = []

    def transmit(self, echo):
        if echo.attributes.get("fusion") or echo.sigil.endswith("∆"):
            self.relic_outbox.append(echo)
            self.gravity_trace.append(f"DecayLink::{echo.sigil}")

# Step 19–20: Runestone Archive + Divergence Boost
class RunestoneIndex:
    def __init__(self):
        self.entries = {}

    def archive(self, echo):
        self.entries[echo.sigil] = echo.lineage

class DivergenceMeter:
    def __init__(self):
        self.count = 0

    def reward_if_unique(self, echo, archive):
        if echo.sigil not in archive.entries:
            self.count += 1
            print(f"Entropy spike: +{self.count}")

