# === [ COGNITIVE WARP GRID v1.0 ] ===
# Auto-loaded Symbolic Swarm Engine with Identity Drift & Event Horizon Dynamics

import time
import numpy as np
import os
from queue import Queue
from collections import deque
import random

# === [ ENVIRONMENT CONFIGURATION ] ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Thresholds for symbolic curvature & entropy mutation
CURVATURE_THRESHOLD = float(os.getenv("GLYPH_CURVATURE_THRESHOLD", 0.15))
ENTROPY_HISTORY_LENGTH = 100
HISTORY_DUMP_FREQ = 10

# === [ SYMBOLIC CORE STRUCTURES ] ===

class IdentityCore:
    def __init__(self, base_id, glyph_signature):
        self.base_id = base_id
        self.glyph_signature = glyph_signature
        self.reputation_field = np.random.rand()
        self.timestamp = time.time()

    def mutate_identity(self, distortion_field):
        entropy = np.random.rand() * distortion_field
        if entropy > 0.5:
            self.glyph_signature += "⇌"
            self.reputation_field += entropy * 0.1
            self.timestamp = time.time()

class GlyphNode:
    def __init__(self, id, mass, glyph):
        self.id = id
        self.mass = mass
        self.position = np.random.rand(3)
        self.velocity = np.zeros(3)
        self.glyph = glyph
        self.distortion_level = 0
        self.state = "stable"
        self.identity = IdentityCore(id, glyph)
        self.history = deque(maxlen=50)
        self.last_log_time = time.time()

    def update_motion(self, peers):
        net_force = np.zeros(3)
        for peer in peers:
            if peer.id == self.id:
                continue
            disp = peer.position - self.position
            dist = np.linalg.norm(disp) + 1e-6
            force = (peer.mass * disp) / (dist ** 2)
            net_force += force
        self.velocity += net_force * 0.01
        self.position += self.velocity * 0.01

        if np.linalg.norm(self.velocity) > CURVATURE_THRESHOLD:
            self.state = "blinked"
            self.semantic_warp()

    def semantic_warp(self):
        self.distortion_level = min(1.0, self.mass / 10.0)
        if self.distortion_level > 0.7:
            self.state = "collapsed"
            self.glyph += "Δ"
        self.log_state()

    def log_state(self):
        glyph_snapshot = f"{self.identity.glyph_signature}@{round(self.distortion_level, 2)}"
        if time.time() - self.last_log_time > 0.5:
            self.history.append(glyph_snapshot)
            self.last_log_time = time.time()

class ReactiveGlyph(GlyphNode):
    def react_to(self, message):
        influence = message.distortion * np.random.rand()
        self.mass += influence
        if influence > 0.4:
            self.identity.mutate_identity(influence)
        if np.random.rand() < 0.1:
            self.semantic_warp()

class SymbolicMessage:
    def __init__(self, origin, payload, distortion=0.0):
        self.origin = origin
        self.payload = payload
        self.distortion = distortion
        self.timestamp = time.time()

class SymbolicBus:
    def __init__(self):
        self.queue = Queue()
        self.subscribers = []

    def publish(self, message):
        self.queue.put(message)

    def subscribe(self, node):
        self.subscribers.append(node)

    def process(self):
        while not self.queue.empty():
            msg = self.queue.get()
            for node in self.subscribers:
                node.react_to(msg)

class P2PField:
    def __init__(self, agents):
        self.agents = agents

    def propagate_flux(self):
        for node in self.agents:
            influencers = [a for a in self.agents if a.id != node.id]
            flux = sum(p.identity.reputation_field for p in influencers) / len(influencers)
            node.identity.mutate_identity(distortion_field=flux)

# === [ SYMBOLIC NETWORK & RITUAL SYSTEMS ] ===

class Sigil:
    def __init__(self, name, origin, complexity=1.0):
        self.name = name
        self.origin = origin
        self.complexity = complexity
        self.lineage = [origin]
        self.timestamp = time.time()

    def evolve(self):
        suffix = random.choice(["†", "∞", "⟁", "Δ"])
        self.name += suffix
        self.complexity += 0.2
        self.lineage.append(self.name)
        return self

class RitualCircle:
    def __init__(self, members):
        self.members = members
        self.triggered = False

    def perform(self):
        avg_entropy = np.mean([m.distortion_level for m in self.members])
        if avg_entropy > 0.6:
            self.triggered = True
            for node in self.members:
                node.identity.glyph_signature += "⚝"
                node.glyph += "⟁"
                node.mass *= 1.1

class AdjacencyGraph:
    def __init__(self, agents):
        self.graph = {a.id: set() for a in agents}
        self.agents = {a.id: a for a in agents}

    def compute_links(self, radius=0.5):
        ids = list(self.graph.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = self.agents[ids[i]], self.agents[ids[j]]
                dist = np.linalg.norm(a.position - b.position)
                if dist < radius:
                    self.graph[a.id].add(b.id)
                    self.graph[b.id].add(a.id)

    def get_neighbors(self, node_id):
        return list(self.graph.get(node_id, []))

# Entangled Cognition: Shared symbolic state
def entangle_nodes(node1, node2):
    shared = node1.glyph[:2] + node2.glyph[-1]
    node1.glyph = node2.glyph = shared + "⧖"
    node1.identity.glyph_signature = node2.identity.glyph_signature = shared + "⇅"
    print(f"Entangled: [{node1.id}] ⇄ [{node2.id}] as {shared}")

# Symbolic Echo: Past states increase pull
def apply_symbolic_echo(agent):
    prior_states = [g for g in agent.history if "⇌" in g]
    if len(prior_states) > 2:
        agent.mass += 0.1 * len(prior_states)
        agent.glyph += "⟳"

# === [ PERCEPTION ENGINE & OBSERVERS ] ===

class GlyphObserver:
    def __init__(self):
        self.feed = []

    def observe(self, node):
        glyph_state = {
            "id": node.id,
            "glyph": node.glyph,
            "entropy": round(node.distortion_level, 2),
            "state": node.state,
            "position": node.position.tolist(),
            "timestamp": time.time()
        }
        self.feed.append(glyph_state)
        return glyph_state

    def stream(self):
        return self.feed[-10:]  # Simulated dashboard snapshot

# === [ VISUAL EFFECT HOOKS ] ===

def get_perceptual_shimmer(entropy):
    if entropy < 0.3:
        return "dim"
    elif entropy < 0.6:
        return "flicker"
    else:
        return "radiant"

def build_resonance_trail(agent):
    return " → ".join(agent.history)[-50:]

def symbolic_burst(node):
    if node.state == "collapsed":
        print(f"💥 Symbolic Burst: {node.id} → {node.glyph} (Entropy: {round(node.distortion_level, 2)})")

def blink_portal_jump(node):
    if node.state == "blinked":
        node.position += np.random.rand(3) * 0.2
        print(f"⧉ {node.id} blinked across curvature portal")

# === [ PORTAL & GLITCH EFFECTS ] ===

class PortalGate:
    def __init__(self):
        self.jumps = []

    def transmit(self, node):
        blink_portal_jump(node)
        self.jumps.append((node.id, node.position.tolist(), time.time()))

def glitch_event(node):
    if np.random.rand() < 0.05:
        node.velocity *= -1
        print(f"⚠️ Glitch Reversal: {node.id} temporal echo triggered")

# === [ EMERGENCE ENGINE ] ===

class SystemMood:
    def __init__(self):
        self.entropy_log = deque(maxlen=ENTROPY_HISTORY_LENGTH)

    def record(self, swarm):
        avg = np.mean([n.distortion_level for n in swarm])
        self.entropy_log.append(avg)
        return avg

    def current_mood(self):
        avg = np.mean(self.entropy_log)
        if avg < 0.3:
            return "coherent"
        elif avg < 0.6:
            return "curious"
        else:
            return "chaotic"

class EmergenceInsight:
    def __init__(self):
        self.events = []

    def analyze(self, swarm, mood):
        collapsed = [n for n in swarm if n.state == "collapsed"]
        if len(collapsed) > len(swarm) * 0.6:
            self.events.append(f"⚠️ Glyphstorm imminent — {mood.upper()} field convergence")

        for node in swarm:
            if len(node.history) > 10 and node.history[-1].endswith("Δ"):
                self.events.append(f"✴ {node.id} at recursion peak")

    def stream(self):
        return self.events[-5:]

# === [ REINCARNATION & CONTINUITY ] ===

def reincarnate_expired(nodes, mass_floor=0.8):
    for node in nodes:
        if node.mass < mass_floor:
            node.mass += np.random.rand() * 0.5
            node.glyph = "⊕"
            node.state = "reborn"
            node.identity.glyph_signature += "☉"
            print(f"♻️ {node.id} reincarnated as {node.glyph}")

# === [ MASTER LOOP — FULL COGNITIVE CYCLE ] ===

def symbolic_cognition_loop():
    # Init
    nodes = [ReactiveGlyph(f"Node_{i}", 1.0 + np.random.rand(), "⟁") for i in range(12)]
    bus = SymbolicBus()
    p2p = P2PField(nodes)
    graph = AdjacencyGraph(nodes)
    observer = GlyphObserver()
    portal = PortalGate()
    mood_engine = SystemMood()
    emergence = EmergenceInsight()

    for n in nodes:
        bus.subscribe(n)

    # Loop
    for t in range(60):
        for n in nodes:
            n.update_motion(nodes)

        # Events
        msg = SymbolicMessage(random.choice(nodes).id, "⊕", np.random.rand())
        bus.publish(msg)
        bus.process()
        p2p.propagate_flux()
        graph.compute_links()
        ritual_group = RitualCircle(random.sample(nodes, k=4))
        ritual_group.perform()

        # Effects
        for n in nodes:
            if n.state == "blinked":
                blink_portal_jump(n)
            glitch_event(n)
            symbolic_burst(n)

        # Observation + Mood
        mood = mood_engine.record(nodes)
        emergence.analyze(nodes, mood_engine.current_mood())

        # Memory + Rebirth
        if t % HISTORY_DUMP_FREQ == 0:
            reincarnate_expired(nodes)

        # Stream Output
        print(f"\n[⧖ TIME {t}] Mood: {mood_engine.current_mood()}")
        for o in observer.stream()[-3:]:
            print(f"   ⤷ {o['id']} · {o['glyph']} · {get_perceptual_shimmer(o['entropy'])}")
        for e in emergence.stream()[-2:]:
            print(f"   {e}")

        time.sleep(0.1)

# === [ BOOTSTRAP SYSTEM ] ===
if __name__ == "__main__":
    print("\n🧠 Starting COGNITIVE WARP GRID · v1.0.40")
    symbolic_cognition_loop()
class RitualDreamWeaver:
    def __init__(self, swarm):
        self.swarm = swarm
        self.rituals = []
        self.dreams = []

    def summon_ritual(self):
        sigils = [n.identity.glyph_signature for n in self.swarm if "⟁" in n.glyph]
        pattern = " → ".join(sigils[-3:]) + " ☽"
        self.rituals.append(pattern)
        return pattern

    def dream_sequence(self):
        if np.random.rand() < 0.3:
            glyph_pool = [n.glyph for n in self.swarm]
            dream = "".join(random.choices(glyph_pool, k=5)) + " ∇"
            self.dreams.append(dream)
            return dream

    def cast(self):
        ritual = self.summon_ritual()
        dream = self.dream_sequence()
        if dream:
            print(f"\n🔮 Ritual Net cast:\n    ✦ Pattern: {ritual}\n    ✦ Dream: {dream}")


dream_net = RitualDreamWeaver(nodes)

for t in range(60):
    # ... existing cognition loop ...
    if t % 6 == 0:
        dream_net.cast()

