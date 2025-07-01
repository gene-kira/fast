# === Autoloader: Core & Optional Imports ===

import sys
import os
import random
import time
import datetime
from collections import defaultdict

# Optional: extendable for GUI, plotting, or async support
try:
    import tkinter as tk  # For GUI experiments
except ImportError:
    tk = None

try:
    import matplotlib.pyplot as plt  # For graphing symbolic data
except ImportError:
    plt = None

try:
    import asyncio  # For future async ritual tasks
except ImportError:
    asyncio = None

# symbolic_simulator.py – Part 1

import random
import time
import datetime
from collections import defaultdict

# ─── Agent & Archetype ───
class Archetype:
    def __init__(self, name, amplifiers=[]):
        self.name = name
        self.amplifiers = amplifiers

class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.lexicon = []
        self.archetype = None

# ─── Myth Thread ───
class MythThread:
    def __init__(self, name, sequence=None):
        self.name = name
        self.sequence = sequence or []

    def record(self, glyph):
        self.sequence.append(glyph)

# ─── Market Economy ───
class SymbolMarket:
    def __init__(self):
        self.glyph_prices = {}
        self.agent_wealth = {}

    def set_price(self, glyph, value):
        self.glyph_prices[glyph] = value
        print(f"[MARKET] '{glyph}' valued at {value} tokens")

    def grant_tokens(self, agent, amount):
        self.agent_wealth[agent.id] = self.agent_wealth.get(agent.id, 0) + amount
        print(f"[GRANT] Agent '{agent.id}' received {amount} tokens")

    def purchase(self, agent, glyph):
        price = self.glyph_prices.get(glyph, 1)
        if self.agent_wealth.get(agent.id, 0) >= price:
            agent.lexicon.append(glyph)
            self.agent_wealth[agent.id] -= price
            print(f"[PURCHASE] Agent '{agent.id}' bought '{glyph}'")
        else:
            print(f"[DENIED] Agent '{agent.id}' lacks funds for '{glyph}'")

# ─── Prediction & Threading ───
class NarrativePredictor:
    def __init__(self):
        self.patterns = {}

    def train(self, threads):
        for thread in threads:
            for i in range(len(thread.sequence) - 1):
                prefix = thread.sequence[i]
                next_glyph = thread.sequence[i + 1]
                self.patterns.setdefault(prefix, []).append(next_glyph)

    def predict(self, glyph):
        options = self.patterns.get(glyph, [])
        if options:
            prediction = max(set(options), key=options.count)
            print(f"[PREDICT] After '{glyph}', likely: '{prediction}'")
            return prediction
        print(f"[PREDICT] No clear prediction after '{glyph}'")
        return None

# ─── Myth Architect ───
class MythArchitect:
    def __init__(self):
        self.threads = []

    def generate(self, name, seed_glyphs, length=5):
        thread = MythThread(name)
        for g in seed_glyphs:
            thread.record(g)
        while len(thread.sequence) < length:
            thread.record(f"auto-{random.randint(1000,9999)}")
        self.threads.append(thread)
        print(f"[ARCHITECT] Myth '{name}' composed: {thread.sequence}")
        return thread

# symbolic_simulator.py – Part 2

# ─── Symbolic Cognition & Drift ───
class SymbolDialect:
    def __init__(self, name, ruleset):
        self.name = name
        self.ruleset = ruleset
        self.drift_map = {}

    def transform(self, glyph):
        return self.ruleset.get(glyph, glyph)

    def register_drift(self, original, drifted):
        self.drift_map[original] = drifted

class GlyphSemanticDrift:
    def __init__(self):
        self.meaning_map = {}

    def evolve(self, glyph, new_meaning):
        self.meaning_map.setdefault(glyph, []).append(new_meaning)

    def get_current_meaning(self, glyph):
        meanings = self.meaning_map.get(glyph, [])
        return meanings[-1] if meanings else "undefined"

class DialectMutator:
    def hybridize(self, d1, d2):
        merged = dict(d1.ruleset)
        for k, v in d2.ruleset.items():
            if k not in merged:
                merged[k] = v
        return SymbolDialect(f"{d1.name}_{d2.name}_Hybrid", merged)

# ─── Lexicon Memories & Codices ───
class MemoryCodex:
    def __init__(self):
        self.codex = {}

    def archive(self, agent_id, glyph):
        self.codex.setdefault(agent_id, []).append(glyph)

    def consult(self, agent_id):
        return self.codex.get(agent_id, [])

# ─── DreamState Symbolism ───
class DreamSymbolism:
    def __init__(self):
        self.dream_glyphs = {}

    def inject(self, agent_id, glyph):
        self.dream_glyphs.setdefault(agent_id, []).append(glyph)

    def awaken(self, agent_id):
        return self.dream_glyphs.pop(agent_id, [])

# ─── Ritual Chamber & Outcome ───
class RitualChamber:
    def __init__(self, location):
        self.location = location
        self.active_rituals = []

    def enact(self, agent, glyph_seq):
        self.active_rituals.append((agent.id, glyph_seq))
        print(f"[RITUAL] Agent '{agent.id}' performed at '{self.location}': {glyph_seq}")

class RitualOutcome:
    def resolve(self, glyph_seq):
        outcome = hash(tuple(glyph_seq)) % 7
        print(f"[OUTCOME] Ritual outcome: {outcome}")
        return outcome

# ─── Archetype Affinity ───
class ArchetypeRitualAffinity:
    def __init__(self):
        self.affinities = {}

    def set_affinity(self, archetype, glyphs):
        self.affinities[archetype] = glyphs

    def evaluate_affinity(self, agent, glyph_seq):
        count = sum(1 for g in glyph_seq if g in self.affinities.get(agent.archetype.name, []))
        print(f"[AFFINITY] Score for '{agent.id}': {count}")
        return count

# ─── Propagation Engine ───
class PropagationEngine:
    def __init__(self):
        self.network = {}

    def connect(self, a1, a2):
        self.network.setdefault(a1.id, []).append(a2.id)

    def propagate(self, glyph, from_id):
        receivers = self.network.get(from_id, [])
        for r in receivers:
            print(f"[PROPAGATE] '{glyph}' reached '{r}'")

# symbolic_simulator.py – Part 3

# ─── Narrative Entanglement ───
class GlyphEntangler:
    def __init__(self):
        self.entangled = {}

    def entangle(self, g1, g2):
        self.entangled.setdefault(g1, []).append(g2)
        self.entangled.setdefault(g2, []).append(g1)

class SymbolFusion:
    def fuse(self, g1, g2):
        return f"{g1}-{g2}"

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

class AnomalyGlyph:
    def __init__(self):
        self.wildcards = {}

    def inject(self, thread, position):
        anomaly = f"anomaly-{random.randint(100,999)}"
        thread.sequence.insert(position, anomaly)
        print(f"[ANOMALY] Injected '{anomaly}' into '{thread.name}'")

# ─── Societal Symbolics ───
class Faction:
    def __init__(self, name):
        self.name = name
        self.members = []

class FactionSchism:
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_conflict(self, faction):
        counts = {}
        for member in faction.members:
            for g in member.lexicon:
                counts[g] = counts.get(g, 0) + 1
        divergent = [g for g, c in counts.items() if c < self.threshold]
        if divergent:
            print(f"[SCHISM] Conflict in '{faction.name}': {divergent}")
        return divergent

class DoctrinalRelics:
    def __init__(self):
        self.relics = {}

    def sanctify(self, glyph, faction):
        self.relics[glyph] = faction.name

class CivicRituals:
    def __init__(self):
        self.events = {}

    def register_event(self, name, glyphs):
        self.events[name] = glyphs

    def enact(self, name):
        glyphs = self.events.get(name, [])
        print(f"[CIVIC] Ritual '{name}' emits glyphs: {glyphs}")
        return glyphs

class SymbolicLaw:
    def __init__(self):
        self.restrictions = set()

    def outlaw(self, glyph):
        self.restrictions.add(glyph)
        print(f"[LAW] Glyph '{glyph}' is now forbidden")

    def is_legal(self, glyph):
        return glyph not in self.restrictions

class InscriptionTower:
    def __init__(self):
        self.log = set()

    def inscribe(self, glyph):
        self.log.add(glyph)
        print(f"[TOWER] Glyph '{glyph}' permanently etched")

# symbolic_simulator.py – Part 4

# ─── Constraint Grammar & Curses ───
class ForbiddenGrammar:
    def __init__(self):
        self.syntax_rules = []

    def restrict(self, seq):
        self.syntax_rules.append(seq)

    def validate(self, glyph_seq):
        for rule in self.syntax_rules:
            if all(r in glyph_seq for r in rule):
                print(f"[VIOLATION] Forbidden sequence detected: {rule}")
                return False
        return True

class CursedGlyph:
    def __init__(self):
        self.cursed = set()

    def curse(self, glyph):
        self.cursed.add(glyph)
        print(f"[CURSE] Glyph '{glyph}' now carries unstable charge")

class LexBreak:
    def sabotage(self, agent, glyph):
        if glyph in agent.lexicon:
            agent.lexicon.remove(glyph)
            print(f"[SABOTAGE] '{glyph}' erased from '{agent.id}'")

# ─── Emergence, Synthesis, and Mirrors ───
class SymbolLab:
    def synthesize(self, g1, g2):
        return f"lab-{g1[:3]}{g2[-3:]}"

class AgentReverie:
    def dream(self, agent_id):
        glyph = f"dream-{random.randint(1000,9999)}"
        print(f"[DREAM] Agent '{agent_id}' saw '{glyph}'")
        return glyph

class MythFork:
    def fork(self, myth, new_name):
        alt = MythThread(new_name)
        alt.sequence = myth.sequence[:len(myth.sequence)//2] + [f"fork-{random.randint(100,999)}"]
        return alt

class SyntacticMirror:
    def reflect(self, glyph_seq):
        mirrored = glyph_seq[::-1]
        print(f"[MIRROR] {glyph_seq} → {mirrored}")
        return mirrored

# ─── SuperCodexManager ───
class SuperCodexManager:
    def __init__(self, codex, erosion, ritual_engine, audit_engine):
        self.codex = codex
        self.erosion = erosion
        self.rituals = ritual_engine
        self.audit = audit_engine
        self.record = {}

    def archive_ritual(self, agent, glyph_seq):
        for glyph in glyph_seq:
            self.codex.archive(agent.id, glyph)
            self.erosion.log_use(glyph)
        self.record.setdefault(agent.id, []).append(glyph_seq)

    def decay_passive_glyphs(self):
        faded = self.erosion.decay()
        for g in faded:
            print(f"[DECAY] Glyph '{g}' faded from symbolic field")

    def run_audit(self, myth_threads, constraint_engine):
        for thread in myth_threads:
            if not constraint_engine.validate(thread.sequence):
                print(f"[AUDIT] Myth '{thread.name}' violated constraints")

# ─── Simulation Loop ───
def simulate_ritual_epoch(system, agents, location, turns=3):
    for epoch in range(turns):
        print(f"\n=== Epoch {epoch+1} ===")

        for agent in agents:
            new_glyph = f"dream-{random.randint(1000,9999)}"
            agent.lexicon.append(new_glyph)
            print(f"[EPOCH] '{agent.id}' dreams '{new_glyph}'")

        for agent in agents:
            if len(agent.lexicon) >= 2:
                sample = random.sample(agent.lexicon, 2)
                system.perform_ritual(agent, sample, location)
                system.codex.archive(agent.id, sample[0])
                system.codex.archive(agent.id, sample[1])
                system.erosion.log_use(sample[0])
                system.erosion.log_use(sample[1])

        if epoch % 2 == 0:
            faded = system.erosion.decay()
            for glyph in faded:
                print(f"[DECAY] Symbol '{glyph}' is forgotten")

# ─── Example System Boot ───
if __name__ == "__main__":
    # Setup
    a1, a2 = Agent("a1"), Agent("a2")
    agents = [a1, a2]
    chamber = RitualChamber("obsidian-circle")

    # System init
    class SymbolicSystem:
        def __init__(self, agents, chamber):
            self.agents = agents
            self.chamber = chamber
            self.codex = MemoryCodex()
            self.erosion = SymbolicErosion()
            self.predictor = NarrativePredictor()
            self.rituals = RitualOutcome()
            self.manager = SuperCodexManager(self.codex, self.erosion, self.rituals, MythAudit())

        def perform_ritual(self, agent, glyphs, location):
            self.chamber.enact(agent, glyphs)
            self.rituals.resolve(glyphs)
            self.manager.archive_ritual(agent, glyphs)

    system = SymbolicSystem(agents, chamber)

    # Simulate epochs
    simulate_ritual_epoch(system, agents, "obsidian-circle", turns=5)

class GlyphLogger:
    def __init__(self):
        self.ritual_logs = []
        self.glyph_stats = defaultdict(int)
        self.agent_usage = defaultdict(lambda: defaultdict(int))

    def log_ritual(self, agent_id, glyphs, epoch):
        self.ritual_logs.append({
            "agent": agent_id,
            "glyphs": glyphs,
            "epoch": epoch,
            "timestamp": datetime.datetime.now().isoformat()
        })
        for g in glyphs:
            self.glyph_stats[g] += 1
            self.agent_usage[agent_id][g] += 1
        print(f"[LOG] Epoch {epoch}: Agent '{agent_id}' used {glyphs}")

    def top_glyphs(self, n=5):
        return sorted(self.glyph_stats.items(), key=lambda x: -x[1])[:n]

    def agent_glyph_profile(self, agent_id):
        return dict(self.agent_usage.get(agent_id, {}))

    def export_log(self, path="glyph_log.json"):
        import json
        with open(path, "w") as f:
            json.dump({
                "logs": self.ritual_logs,
                "stats": dict(self.glyph_stats)
            }, f, indent=2)
        print(f"[EXPORT] Glyph logs saved to {path}")

import matplotlib.pyplot as plt
from collections import defaultdict

class GlyphDashboard:
    def __init__(self):
        self.epoch_logs = []
        self.glyph_history = defaultdict(list)
        self.current_epoch = 0

    def log_epoch(self, glyph_counts):
        self.current_epoch += 1
        self.epoch_logs.append((self.current_epoch, dict(glyph_counts)))
        for glyph, count in glyph_counts.items():
            self.glyph_history[glyph].append((self.current_epoch, count))

    def plot_top_glyphs(self, top_n=5):
        plt.figure(figsize=(10, 6))
        ranked = sorted(self.glyph_history.items(), key=lambda x: -sum(c for _, c in x[1]))[:top_n]
        for glyph, history in ranked:
            epochs, counts = zip(*history)
            plt.plot(epochs, counts, label=glyph)
        plt.title("Glyph Resonance Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Usage Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Before loop
dashboard = GlyphDashboard()

# Inside each epoch:
epoch_counts = defaultdict(int)
for agent in agents:
    for g in agent.lexicon:
        epoch_counts[g] += 1
dashboard.log_epoch(epoch_counts)

# After loop:
dashboard.plot_top_glyphs()

class SymbolEntropyScanner:
    def __init__(self):
        self.lineages = defaultdict(list)

    def trace(self, derived, source):
        self.lineages[derived].append(source)

    def compute_entropy(self, glyph):
        lineage = self.lineages.get(glyph, [])
        if not lineage:
            return 0.0
        unique = len(set(lineage))
        total = len(lineage)
        return round(unique / total, 3)

class GlyphMorphogenesis:
    def __init__(self):
        self.counter = 0

    def generate(self, base1, base2):
        self.counter += 1
        return f"morph-{base1[:2]}{base2[-2:]}-{self.counter}"

class DreamWave:
    def __init__(self):
        self.amplitude = defaultdict(int)

    def pulse(self, agent_id, glyph):
        self.amplitude[glyph] += 1
        print(f"[DREAMWAVE] '{glyph}' amplitude now {self.amplitude[glyph]}")

class AffinityLens:
    def __init__(self):
        self.affinity_map = defaultdict(lambda: defaultdict(int))

    def update(self, agent_id, glyph, points):
        self.affinity_map[agent_id][glyph] += points

    def top(self, agent_id, n=3):
        return sorted(self.affinity_map[agent_id].items(), key=lambda x: -x[1])[:n]

class EchoPhoneme:
    def echo(self, glyph):
        tone = sum(ord(c) for c in glyph) % 9
        print(f"[ECHO] Glyph '{glyph}' emits tone {tone}")
        return tone

class SymbolEntropyScanner:
    def __init__(self):
        self.lineages = defaultdict(list)

    def trace(self, derived, source):
        self.lineages[derived].append(source)

    def compute_entropy(self, glyph):
        lineage = self.lineages.get(glyph, [])
        if not lineage:
            return 0.0
        unique = len(set(lineage))
        total = len(lineage)
        return round(unique / total, 3)

class GlyphMorphogenesis:
    def __init__(self):
        self.counter = 0

    def generate(self, base1, base2):
        self.counter += 1
        return f"morph-{base1[:2]}{base2[-2:]}-{self.counter}"

class DreamWave:
    def __init__(self):
        self.amplitude = defaultdict(int)

    def pulse(self, agent_id, glyph):
        self.amplitude[glyph] += 1
        print(f"[DREAMWAVE] '{glyph}' amplitude now {self.amplitude[glyph]}")

class AffinityLens:
    def __init__(self):
        self.affinity_map = defaultdict(lambda: defaultdict(int))

    def update(self, agent_id, glyph, points):
        self.affinity_map[agent_id][glyph] += points

    def top(self, agent_id, n=3):
        return sorted(self.affinity_map[agent_id].items(), key=lambda x: -x[1])[:n]

class EchoPhoneme:
    def echo(self, glyph):
        tone = sum(ord(c) for c in glyph) % 9
        print(f"[ECHO] Glyph '{glyph}' emits tone {tone}")
        return tone

class EpochRuleStack:
    def __init__(self):
        self.rules = defaultdict(list)

    def inject_rule(self, epoch, fn):
        self.rules[epoch].append(fn)

    def execute(self, epoch):
        for fn in self.rules.get(epoch, []):
            fn()
        print(f"[EPOCH RULE] Executed {len(self.rules.get(epoch, []))} rules for epoch {epoch}")

class SymbolResonator:
    def __init__(self):
        self.links = defaultdict(list)

    def link(self, glyph_a, glyph_b):
        self.links[glyph_a].append(glyph_b)

    def resonate(self, input_glyph):
        echoes = self.links.get(input_glyph, [])
        print(f"[RESONANCE] '{input_glyph}' echoed → {echoes}")
        return echoes

class TemporalBiasField:
    def __init__(self):
        self.bias_map = {}

    def set_bias(self, epoch, field):
        self.bias_map[epoch] = field

    def apply_bias(self, epoch, ritual_outcome):
        modifier = self.bias_map.get(epoch, 0)
        result = (ritual_outcome + modifier) % 7
        print(f"[BIAS] Outcome modified by {modifier} → {result}")
        return result

class SynchroSignalEmitter:
    def __init__(self):
        self.schedule = {}

    def emit(self, epoch):
        return self.schedule.get(epoch, None)

    def bind(self, epoch, glyph):
        self.schedule[epoch] = glyph

class RitualCascadeEngine:
    def __init__(self, propagation_engine):
        self.network = propagation_engine

    def cascade(self, initiating_agent, glyph_seq):
        neighbors = self.network.propagate("::".join(glyph_seq), initiating_agent.id)
        for neighbor_id in neighbors:
            print(f"[CASCADE] Ritual symbol '{glyph_seq}' influenced '{neighbor_id}'")

class LexiconProfiler:
    def profile(self, agent):
        print(f"\n[PROFILE] Agent '{agent.id}' Lexicon:")
        for glyph in agent.lexicon:
            print(f"  - {glyph}")

class GlyphUsageHeatmap:
    def __init__(self):
        self.glyph_heat = defaultdict(int)

    def record_usage(self, glyph):
        self.glyph_heat[glyph] += 1

    def top_glyphs(self, n=5):
        return sorted(self.glyph_heat.items(), key=lambda x: -x[1])[:n]

class DecayVisualizer:
    def __init__(self, erosion_module):
        self.erosion = erosion_module

    def print_decay_state(self):
        print("[DECAY PROFILE]")
        for glyph, count in self.erosion.activity.items():
            print(f"  '{glyph}': {count} uses")

class AgentPersonaPrinter:
    def print_persona(self, agent, codex):
        memory = codex.consult(agent.id)
        print(f"\n[PERSONA] Agent '{agent.id}' recalls:")
        for g in memory:
            print(f"  • {g}")

class EpochMetricsRecorder:
    def __init__(self):
        self.history = []

    def log_epoch(self, epoch, data):
        self.history.append((epoch, dict(data)))

    def summarize(self):
        print("\n[EPOCH SUMMARY]")
        for epoch, state in self.history:
            print(f"  Epoch {epoch}: {state}")

class RitualLoopOrchestrator:
    def __init__(self, system, agents, logger, dashboard, rule_stack):
        self.system = system
        self.agents = agents
        self.logger = logger
        self.dashboard = dashboard
        self.rule_stack = rule_stack
        self.epoch = 0

    def tick(self):
        self.epoch += 1
        print(f"\n=== Epoch {self.epoch} ===")
        self.rule_stack.execute(self.epoch)

        epoch_glyph_counts = defaultdict(int)

        for agent in self.agents:
            glyph = f"dream-{random.randint(1000,9999)}"
            agent.lexicon.append(glyph)
            print(f"[EPOCH] Agent '{agent.id}' dreams '{glyph}'")
            epoch_glyph_counts[glyph] += 1
            self.logger.log_ritual(agent.id, [glyph], self.epoch)

        for agent in self.agents:
            if len(agent.lexicon) >= 2:
                ritual_seq = random.sample(agent.lexicon, 2)
                self.system.perform_ritual(agent, ritual_seq, "epoch-zone")
                self.logger.log_ritual(agent.id, ritual_seq, self.epoch)
                for g in ritual_seq:
                    epoch_glyph_counts[g] += 1

        self.dashboard.log_epoch(epoch_glyph_counts)

        if self.epoch % 2 == 0:
            self.system.manager.decay_passive_glyphs()

class SimulationStateExporter:
    def export(self, agents, codex, epoch, path="simulation_state.txt"):
        with open(path, "w") as f:
            f.write(f"=== SIMULATION STATE @ Epoch {epoch} ===\n")
            for agent in agents:
                f.write(f"\nAgent {agent.id}:\n  Lexicon: {agent.lexicon}\n")
                f.write(f"  Memory: {codex.consult(agent.id)}\n")
        print(f"[EXPORT] Simulation state saved to {path}")

class GlyphInjectionPortal:
    def __init__(self, system):
        self.system = system

    def inject(self, agent, glyph):
        agent.lexicon.append(glyph)
        print(f"[PORTAL] External glyph '{glyph}' injected to Agent '{agent.id}'")

class SignalOverride:
    def __init__(self):
        self.override_signals = {}

    def set_signal(self, epoch, action_fn):
        self.override_signals[epoch] = action_fn

    def check_and_execute(self, epoch):
        if epoch in self.override_signals:
            self.override_signals[epoch]()
            print(f"[OVERRIDE] Epoch {epoch} signal executed")

class SystemDiagnostics:
    def run_check(self, system, agents):
        print("\n[DIAGNOSTICS]")
        print(f"  Agents: {len(agents)}")
        print(f"  Codex Size: {len(system.codex.codex)}")
        print(f"  Erosion Tracked Glyphs: {len(system.erosion.activity)}")
        print(f"  Ritual Log Count: {len(getattr(system, 'manager').record)}")

