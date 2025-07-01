
# ─── Symbolic System Autoloader ───

class SymbolSystem:
    def __init__(self, agents, environment="echo-realm"):
        print("[SYSTEM] Bootstrapping symbolic substrate...")
        self.agents = agents
        self.environment = environment

        # Core Modules
        self.market = SymbolMarket()
        self.predictor = NarrativePredictor()
        self.architect = MythArchitect()
        self.codex = MemoryCodex()
        self.rules = ConstraintGlyph()
        self.composer = LexicalComposer()
        self.env = EnvironmentInterface()

        # Cognition & Dialect
        self.dialects = {}
        self.semantic_drift = GlyphSemanticDrift()
        self.resonance = CognitiveResonance()
        self.dreams = DreamSymbolism()
        self.mutator = DialectMutator()

        # Ritual & Propagation
        self.chambers = {}
        self.outcome_engine = RitualOutcome()
        self.propagation = PropagationEngine()
        self.beacons = []
        self.affinities = ArchetypeRitualAffinity()

        # Narrative Entanglement
        self.entangler = GlyphEntangler()
        self.fuser = SymbolFusion()
        self.inertia = NarrativeInertia()
        self.canon = CanonWeaver()
        self.anomalies = AnomalyGlyph()

        # Registry logs
        print("[SYSTEM] Initialization complete.\n")

    def register_dialect(self, name, ruleset):
        self.dialects[name] = SymbolDialect(name, ruleset)

    def launch_ritual_chamber(self, location):
        self.chambers[location] = RitualChamber(location)

    def perform_ritual(self, agent, glyph_seq, location):
        if location in self.chambers:
            self.chambers[location].enact(agent, glyph_seq)
            self.outcome_engine.resolve(glyph_seq)
            self.affinities.evaluate_affinity(agent, glyph_seq)
        else:
            print(f"[ERROR] Ritual chamber '{location}' not found")

    def transmit_symbol(self, source_agent_id, glyph):
        self.propagation.propagate(glyph, source_agent_id)

    def trigger_environment(self, trigger):
        glyph = self.env.update(trigger)
        if glyph:
            self.transmit_symbol("env", glyph)

    def evolve_dialect(self, base_name, drift_map):
        if base_name in self.dialects:
            for k, v in drift_map.items():
                self.dialects[base_name].register_drift(k, v)
            print(f"[DRIFT] Dialect '{base_name}' evolved with drift map: {drift_map}")

    def weave_myth(self, name, glyphs, inject_anomaly=False):
        myth = self.architect.generate(name, glyphs)
        if inject_anomaly:
            self.anomalies.inject(myth, len(myth.sequence)//2)
        self.predictor.train([myth])
        self.canon.include(myth)
        return myth

    def bind_resonance_pairs(self, pairs):
        for (g1, g2, w) in pairs:
            self.resonance.bind(g1, g2, w)

# Sample agent and setup
class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.lexicon = []
        self.archetype = None

a1 = Agent("a1")
a2 = Agent("a2")

# System Init
system = SymbolSystem([a1, a2])

# Dialect
system.register_dialect("AncientScript", {"glyphfire": "glyphfyr"})

# Chamber Setup
system.launch_ritual_chamber("obsidian-circle")

# Perform Ritual
system.perform_ritual(a1, ["glyphfire", "plasma-truth"], "obsidian-circle")

# Propagate and Predict
system.market.set_price("plasma-truth", 3)
system.market.grant_tokens(a1, 10)
system.market.purchase(a1, "plasma-truth")
system.propagation.connect(a1, a2)
system.transmit_symbol("a1", "plasma-truth")

# Myth Weaving
new_myth = system.weave_myth("Genesis Spiral", ["core-insight", "plasma-truth"], inject_anomaly=True)
system.composer.compose(new_myth)





class RitualChamber:
    def __init__(self, location):
        self.location = location
        self.active_rituals = []

    def enact(self, agent, glyph_seq):
        self.active_rituals.append((agent.id, glyph_seq))
        print(f"[RITUAL] Agent '{agent.id}' performed ritual at '{self.location}': {glyph_seq}")

class RitualOutcome:
    def resolve(self, glyph_seq):
        outcome = hash(tuple(glyph_seq)) % 7  # example pseudo-effect system
        print(f"[OUTCOME] Ritual sequence {glyph_seq} resulted in outcome type {outcome}")
        return outcome

class PropagationEngine:
    def __init__(self):
        self.network = {}  # agent_id → connected agents

    def connect(self, a1, a2):
        self.network.setdefault(a1.id, []).append(a2.id)
        print(f"[PROPAGATION] Linked '{a1.id}' ↔ '{a2.id}'")

    def propagate(self, glyph, from_id):
        receivers = self.network.get(from_id, [])
        for r in receivers:
            print(f"[PROPAGATE] '{glyph}' propagated to '{r}'")

class SymbolBeacon:
    def __init__(self, glyph):
        self.glyph = glyph

    def broadcast(self):
        print(f"[BEACON] Broadcasting glyph '{self.glyph}' to all receivers")

class ArchetypeRitualAffinity:
    def __init__(self):
        self.affinities = {}  # archetype → preferred glyphs

    def set_affinity(self, archetype, glyphs):
        self.affinities[archetype] = glyphs

    def evaluate_affinity(self, agent, glyph_seq):
        affinity = sum(1 for g in glyph_seq if g in self.affinities.get(agent.archetype.name, []))
        print(f"[AFFINITY] Ritual by '{agent.id}' has affinity score: {affinity}")
        return affinity

class GlyphEntangler:
    def __init__(self):
        self.entangled = {}  # glyph → entangled partners

    def entangle(self, g1, g2):
        self.entangled.setdefault(g1, []).append(g2)
        self.entangled.setdefault(g2, []).append(g1)
        print(f"[ENTANGLE] '{g1}' ↔ '{g2}' linked in myth-space")

class SymbolFusion:
    def fuse(self, g1, g2):
        compound = f"{g1}-{g2}"
        print(f"[FUSION] Fused '{g1}' + '{g2}' → '{compound}'")
        return compound

class NarrativeInertia:
    def __init__(self):
        self.usage_counts = {}

    def log_usage(self, glyph):
        self.usage_counts[glyph] = self.usage_counts.get(glyph, 0) + 1

    def get_inertia(self, glyph):
        return self.usage_counts.get(glyph, 0)

class CanonWeaver:
    def __init__(self):
        self.canon = []

    def include(self, myth_thread):
        self.canon.append(myth_thread)
        print(f"[CANON] Myth '{myth_thread.name}' woven into official history")

class AnomalyGlyph:
    def __init__(self):
        self.wildcards = {}

    def inject(self, thread, position):
        g = f"anomaly-{random.randint(100,999)}"
        thread.sequence.insert(position, g)
        print(f"[ANOMALY] Injected '{g}' into '{thread.name}' at position {position}")

class SuperCodexManager:
    def __init__(self, codex, erosion, ritual_engine, audit_engine):
        self.codex = codex
        self.erosion = erosion
        self.rituals = ritual_engine
        self.audit = audit_engine
        self.record = {}  # agent_id → [ritual glyph sequences]

    def archive_ritual(self, agent, glyph_seq):
        for glyph in glyph_seq:
            self.codex.archive(agent.id, glyph)
            self.erosion.log_use(glyph)
        self.record.setdefault(agent.id, []).append(glyph_seq)
        print(f"[SUPER] Archived ritual for agent '{agent.id}': {glyph_seq}")

    def decay_passive_glyphs(self):
        faded = self.erosion.decay()
        for glyph in faded:
            print(f"[SUPER] Glyph '{glyph}' eroded from the symbolic field")

    def run_audit(self, myth_threads, constraint_engine):
        for thread in myth_threads:
            self.audit.inspect(thread, constraint_engine)

def symbolic_repl(system):
    print("=== Symbolic Interaction Shell ===")
    print("Type 'help' for commands. Type 'exit' to leave.")
    while True:
        cmd = input("> ").strip()
        if cmd == "exit":
            break
        elif cmd.startswith("ritual"):
            parts = cmd.split()
            agent_id, *glyphs = parts[1:]
            agent = next(a for a in system.agents if a.id == agent_id)
            system.perform_ritual(agent, glyphs, "obsidian-circle")
        elif cmd.startswith("myth"):
            _, name, *glyphs = cmd.split()
            thread = system.weave_myth(name, glyphs)
            system.composer.compose(thread)
        elif cmd.startswith("memory"):
            _, aid = cmd.split()
            system.codex.consult(aid)
        elif cmd.startswith("predict"):
            _, glyph = cmd.split()
            system.predictor.predict(glyph)
        elif cmd == "help":
            print("Commands:")
            print("  ritual <agent_id> <glyph1> <glyph2> ...")
            print("  myth <myth_name> <seed_glyph1> <seed_glyph2> ...")
            print("  memory <agent_id>")
            print("  predict <glyph>")
            print("  exit")
        else:
            print("Unknown command. Try 'help'.")

