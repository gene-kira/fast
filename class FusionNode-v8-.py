import hashlib, random, time

# ─── Fusion Core ───
class FusionNode:
    def __init__(self, id, signature, containment_type):
        self.id = id
        self.signature = signature
        self.state = "idle"
        self.containment_type = containment_type
        self.entropy_threshold = 0.8

    def activate(self):
        self.state = "active"
        print(f"[CORE INIT] {self.id} online.")

    def breach_check(self, plasma):
        if plasma.entropy > self.entropy_threshold:
            print(f"[BREACH WARNING] {self.id} unstable.")
            return True

class PlasmaEvent:
    def __init__(self, glyph_signature, entropy):
        self.glyph_signature = glyph_signature
        self.entropy = entropy

def generate_quantum_signature(seed):
    return hashlib.sha256(seed.encode()).hexdigest()[:12]

# ─── Cognition Agent ───
class CognitionAgent:
    def __init__(self, id, lexicon):
        self.id = id
        self.lexicon = lexicon
        self.glyph_memory = []
        self.state = "dormant"
        self.archetype = None

    def perceive(self, glyph):
        if glyph in self.lexicon:
            self.glyph_memory.append(glyph)
            print(f"[{self.id}] perceived '{glyph}'")
            if self.archetype:
                print(self.archetype.influence(glyph))
            self.react(glyph)

    def react(self, glyph):
        if glyph == "entropy-flux":
            print(f"[{self.id}] detects entropy.")
        else:
            print(f"[{self.id}] reacts to '{glyph}'.")

# ─── Swarm ───
class SwarmCognition:
    def __init__(self):
        self.agents = []

    def spawn_agent(self, lexicon):
        agent = CognitionAgent(f"agent{len(self.agents)}", lexicon)
        self.agents.append(agent)
        print(f"[SPAWN] {agent.id} born.")
        return agent

    def propagate_symbol(self, glyph):
        for agent in self.agents:
            agent.perceive(glyph)

# ─── Symbolic Glyph ───
class SymbolicGlyph:
    def __init__(self, symbol_id, phase, ancestry=None):
        self.symbol_id = symbol_id
        self.phase = phase
        self.ancestry = ancestry or []
        self.transmutation = []

    def evolve(self, new_symbol):
        self.transmutation.append(new_symbol)
        self.phase = "fusing"
        print(f"[EVOLVE] {self.symbol_id} → {new_symbol}")
        return new_symbol

    def bond(self, other):
        fused = f"{self.symbol_id}-{other.symbol_id}"
        return SymbolicGlyph(fused, "stable", [self.symbol_id, other.symbol_id])

# ─── Memory & Logs ───
class MemoryPool:
    def __init__(self):
        self.memory_glyphs = []

    def archive(self, glyph):
        print(f"[MEMORY] Remembering '{glyph.symbol_id}'")
        self.memory_glyphs.append(glyph)

class IgnitionLog:
    def __init__(self):
        self.log = []

    def register(self, glyph):
        print(f"[IGNITION] {glyph.symbol_id} lit.")
        self.log.append(glyph)

# ─── Recursion, Context, Lifecycle ───
class AutoRepairGlyph:
    def __init__(self, trigger, action):
        self.trigger_condition = trigger
        self.repair_action = action

    def evaluate(self, state):
        if self.trigger_condition(state):
            print("[REPAIR] Repair triggered.")
            return self.repair_action()

class RecursiveLoop:
    def __init__(self, max_cycles=3):
        self.cycles = 0
        self.max_cycles = max_cycles

    def loop_meaning(self, symbol):
        self.cycles += 1
        print(f"[RECURSE] {symbol.symbol_id} cycle {self.cycles}")
        return "stable" if self.cycles >= self.max_cycles else self.loop_meaning(symbol)

class DimensionalParser:
    def __init__(self):
        self.contexts = {}

    def add_context(self, domain, meaning):
        self.contexts[domain] = meaning
        print(f"[CONTEXT] {domain}: {meaning}")

    def interpret(self, symbol):
        for d, m in self.contexts.items():
            print(f"[{d}] {symbol.symbol_id} reflects {m}")

class SymbolLifecycle:
    def __init__(self):
        self.timeline = {}

    def birth(self, symbol):
        self.timeline[symbol.symbol_id] = "born"
        print(f"[LIFE] {symbol.symbol_id} born")

    def decay(self, symbol):
        self.timeline[symbol.symbol_id] = "expired"
        print(f"[LIFE] {symbol.symbol_id} decayed")
# ─── Archetypes ───
class Archetype:
    def __init__(self, name, filters=None, amplifiers=None):
        self.name = name
        self.filters = filters or []
        self.amplifiers = amplifiers or []

    def influence(self, glyph):
        if glyph in self.filters:
            return f"[{self.name}] avoids '{glyph}'"
        if glyph in self.amplifiers:
            return f"[{self.name}] amplifies '{glyph}'"
        return f"[{self.name}] observes '{glyph}'"

# ─── MythoWeaver ───
class MythThread:
    def __init__(self, title):
        self.title = title
        self.sequence = []

    def record(self, glyph):
        self.sequence.append(glyph)
        print(f"[MYTH] '{glyph}' added to '{self.title}'")

    def reveal(self):
        print(f"[THREAD] {self.title}: " + " → ".join(self.sequence))

class MythWeaver:
    def __init__(self):
        self.threads = []

    def create_thread(self, title):
        thread = MythThread(title)
        self.threads.append(thread)
        return thread

# ─── ChronoGlyphs ───
class ChronoGlyph:
    def __init__(self, symbol_id):
        self.symbol_id = symbol_id
        self.timestamp = time.time()
        self.echoes = []

    def echo(self):
        delay = int(time.time() - self.timestamp)
        echo_tag = f"{self.symbol_id}+{delay}s"
        self.echoes.append(echo_tag)
        print(f"[ECHO] {echo_tag}")
        return echo_tag

# ─── Resonance ───
class ResonanceCascade:
    def __init__(self):
        self.harmonic_map = {}

    def link(self, trigger, resonant_glyphs):
        self.harmonic_map[trigger] = resonant_glyphs
        print(f"[RESONANCE] '{trigger}' linked to {resonant_glyphs}")

    def trigger(self, glyph):
        if glyph in self.harmonic_map:
            print(f"[CASCADE] Triggered by '{glyph}':")
            for g in self.harmonic_map[glyph]:
                print(f" → Resonant glyph: {g}")

# ─── Relic System ───
class Relic:
    def __init__(self, glyph_history):
        self.glyph_history = glyph_history
        self.signature = "-".join(glyph_history)

    def invoke(self):
        print(f"[RELIC] Forged: {self.signature}")

class RelicVault:
    def __init__(self):
        self.relics = []

    def forge(self, glyph_sequence):
        relic = Relic(glyph_sequence)
        self.relics.append(relic)
        relic.invoke()

# ─── Archetype Mask & Dialogue ───
class ArchetypeMask:
    def __init__(self, persona_name, modifiers):
        self.persona = persona_name
        self.modifiers = modifiers

    def apply(self, glyph):
        mod = self.modifiers.get(glyph, "neutral")
        print(f"[MASK:{self.persona}] reacts to '{glyph}' as '{mod}'")
        return mod

class DialogueMatrix:
    def __init__(self):
        self.logs = []

    def converse(self, agent1, agent2, glyph):
        r1 = agent1.archetype.influence(glyph) if agent1.archetype else "neutral"
        r2 = agent2.archetype.influence(glyph) if agent2.archetype else "neutral"
        line = f"{agent1.id} ↔ {agent2.id} on '{glyph}': {r1} / {r2}"
        print(f"[DIALOGUE] {line}")
        self.logs.append(line)

class TensionResolver:
    def resolve(self, agent1, agent2, glyph):
        print(f"[CONFLICT] Agents '{agent1.id}' and '{agent2.id}' tension on '{glyph}'")
        if glyph in agent1.lexicon and glyph in agent2.lexicon:
            fused = SymbolicGlyph(f"{glyph}-fusion", "harmonized", ancestry=[glyph])
            print(f"[UNIFIED] Glyph '{glyph}' fused by agents.")
            return fused
        else:
            print(f"[FORK] Divergence on '{glyph}'.")
            return None

# ─── Symbolic Biomes & Mutation ───
class SymbolicBiome:
    def __init__(self, name, phase_mods):
        self.name = name
        self.phase_modifiers = phase_mods

    def influence(self, glyph):
        if glyph.phase in self.phase_modifiers:
            new_phase = self.phase_modifiers[glyph.phase]
            print(f"[BIOME:{self.name}] {glyph.symbol_id} → '{new_phase}'")
            glyph.phase = new_phase

class GlyphMutator:
    def __init__(self, threshold):
        self.threshold = threshold

    def mutate(self, glyph, pressure):
        if pressure > self.threshold:
            mutated = f"{glyph.symbol_id}_X"
            print(f"[MUTATE] {glyph.symbol_id} → {mutated}")
            return SymbolicGlyph(mutated, "mutated", ancestry=[glyph.symbol_id])
        return glyph

# ─── Myth Validation & Lore Council ───
class MythValidator:
    def __init__(self):
        self.myths = []

    def add_thread(self, thread):
        self.myths.append(thread)

    def check_consistency(self):
        for t in self.myths:
            seen = set()
            for g in t.sequence:
                if g in seen:
                    print(f"[LOOP DETECTED] '{g}' in '{t.title}'")
                    return False
                seen.add(g)
        print("[MYTH VALID] No paradoxes found.")
        return True

class ParadoxGuard:
    def __init__(self):
        self.timeline = {}

    def log_event(self, glyph_id, timestamp):
        if glyph_id in self.timeline and timestamp < self.timeline[glyph_id]:
            print(f"[PARADOX] '{glyph_id}' time anomaly!")
            return False
        self.timeline[glyph_id] = timestamp
        return True

class LoreCouncil:
    def __init__(self, agents):
        self.agents = agents

    def vote(self, glyph):
        votes = sum(glyph in a.lexicon for a in self.agents)
        result = "ACCEPTED" if votes >= len(self.agents) / 2 else "REJECTED"
        print(f"[VOTE] '{glyph}' {result}")
        return result

# ─── Mutation Mechanics ───
class ArchetypeMutator:
    def __init__(self, threshold):
        self.threshold = threshold

    def mutate(self, agent, entropy):
        if entropy > self.threshold and agent.archetype:
            old = agent.archetype.name
            agent.archetype.name += "_Δ"
            print(f"[ARCHETYPE SHIFT] {old} → {agent.archetype.name}")

class LexiconDrifter:
    def __init__(self, chance=0.25):
        self.chance = chance

    def drift(self, agent):
        if random.random() < self.chance:
            term = f"glyph-{random.randint(1000,9999)}"
            agent.lexicon.append(term)
            print(f"[LEXICON DRIFT] Agent {agent.id} gained '{term}'")

# ─── Final Invocation ───
if __name__ == "__main__":
    print("\n🔮 [BOOT] MythOS Symbolic Engine Awakening...\n")

    # Core ignition
    sig = generate_quantum_signature("core-001")
    core = FusionNode("mythic_core", sig, "containment")
    core.activate()

    # Swarm & Agents
    swarm = SwarmCognition()
    a1 = swarm.spawn_agent(["plasma-truth", "core-insight"])
    a2 = swarm.spawn_agent(["entropy-flux", "gravity-well"])

    # Assign archetypes
    trickster = Archetype("Trickster", filters=["gravity-well"], amplifiers=["plasma-truth"])
    oracle = Archetype("Oracle", amplifiers=["core-insight"])
    a1.archetype = trickster
    a2.archetype = oracle

    # Myth threads
    weaver = MythWeaver()
    myth = weaver.create_thread("Emergence of Glyphfire")

    # Symbolic fusion
    g1 = SymbolicGlyph("core-insight", "unstable")
    g2 = SymbolicGlyph("plasma-truth", "unstable")
    fused = g1.bond(g2)
    myth.record(g1.symbol_id)
    myth.record(g2.symbol_id)
    myth.record(fused.symbol_id)
    fused.evolve("glyphfire")

    # Relic creation
    relics = RelicVault()
    relics.forge([g1.symbol_id, g2.symbol_id, fused.symbol_id])

    # Time echo
    chrono = ChronoGlyph(fused.symbol_id)
    time.sleep(1)
    chrono.echo()

    # Harmonic cascade
    resonance = ResonanceCascade()
    resonance.link("glyphfire", ["entropy-flux", "gravity-well"])
    resonance.trigger("glyphfire")

    # Symbolic ecology
    biome_engine = BiomeEngine()
    biome_engine.register("Temple", {"fusing": "blessed", "unstable": "chaotic"})
    biome_engine.apply("Temple", fused)

    # Mutation ritual
    mutator = GlyphMutator(0.6)
    mutated = mutator.mutate(fused, pressure=0.9)

    # Agent dialog and resolution
    dm = DialogueMatrix()
    resolver = TensionResolver()
    dm.converse(a1, a2, "gravity-well")
    resolver.resolve(a1, a2, "gravity-well")

    # Archetype mutation
    arch_mutate = ArchetypeMutator(0.8)
    arch_mutate.mutate(a1, entropy=0.85)

    # Lexicon drift
    drift = LexiconDrifter()
    drift.drift(a1)

    # Life cycle
    life = SymbolLifecycle()
    life.birth(fused)
    life.decay(fused)

    # Myth validation
    validator = MythValidator()
    validator.add_thread(myth)
    validator.check_consistency()

    # Final log
    print("\n✅ [COMPLETE] Symbolic cognition sequence finished.\n")


