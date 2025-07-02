# === COGNITIVE DREAMSWARM v1.0.40R ===
# Symbolic Warp Grid · Dream Recursion · Ritual Glyph Engine

# --- AUTOLOADER ---
try:
    import numpy as np
    import time, random, os
    from queue import Queue
    from collections import deque
except ImportError as e:
    print("Missing libraries. Please install: numpy")
    exit()

# --- ENVIRONMENT CONFIG ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
CURVATURE_THRESHOLD = float(os.getenv("GLYPH_CURVATURE_THRESHOLD", 0.15))
ENTROPY_HISTORY_LENGTH = 100
HISTORY_DUMP_FREQ = 10

# === CORE MODULES ===

class IdentityCore:
    def __init__(self, base_id, glyph_signature):
        self.base_id = base_id
        self.glyph_signature = glyph_signature
        self.reputation_field = np.random.rand()
        self.timestamp = time.time()

    def mutate_identity(self, distortion_field):
        if np.random.rand() * distortion_field > 0.5:
            self.glyph_signature += "⇌"
            self.reputation_field += 0.1
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
        for p in peers:
            if p.id != self.id:
                disp = p.position - self.position
                dist = np.linalg.norm(disp) + 1e-6
                net_force += (p.mass * disp) / (dist ** 2)
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
        if time.time() - self.last_log_time > 0.5:
            self.history.append(f"{self.identity.glyph_signature}@{round(self.distortion_level, 2)}")
            self.last_log_time = time.time()

class ReactiveGlyph(GlyphNode):
    def react_to(self, msg):
        influence = msg.distortion * np.random.rand()
        self.mass += influence
        if influence > 0.4:
            self.identity.mutate_identity(influence)
        if np.random.rand() < 0.1:
            self.semantic_warp()

# === SIGNAL BUS + P2P ===

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

    def publish(self, message): self.queue.put(message)
    def subscribe(self, node): self.subscribers.append(node)
    def process(self):
        while not self.queue.empty():
            msg = self.queue.get()
            for node in self.subscribers:
                node.react_to(msg)

class P2PField:
    def __init__(self, nodes): self.nodes = nodes
    def propagate_flux(self):
        for node in self.nodes:
            peers = [n for n in self.nodes if n != node]
            flux = sum(p.identity.reputation_field for p in peers) / len(peers)
            node.identity.mutate_identity(distortion_field=flux)

class AdjacencyGraph:
    def __init__(self, nodes):
        self.graph = {n.id: set() for n in nodes}
        self.nodes = {n.id: n for n in nodes}

    def compute_links(self, radius=0.5):
        for i, a in enumerate(self.nodes.values()):
            for b in list(self.nodes.values())[i+1:]:
                if np.linalg.norm(a.position - b.position) < radius:
                    self.graph[a.id].add(b.id)
                    self.graph[b.id].add(a.id)

# === RITUALS & DREAM SYSTEM ===

class RitualCircle:
    def __init__(self, members): self.members = members
    def perform(self):
        if np.mean([m.distortion_level for m in self.members]) > 0.6:
            for m in self.members:
                m.identity.glyph_signature += "⚝"
                m.glyph += "⟁"
                m.mass *= 1.1

class SymbolicLexicon:
    def __init__(self):
        self.glyph_set = set()
        self.lineage_map = {}
    def update(self, string):
        for g in string:
            if g not in " ☽∇→":
                self.glyph_set.add(g)
                self.lineage_map[g] = self.lineage_map.get(g, 0) + 1
    def most_resonant(self):
        return max(self.lineage_map, key=self.lineage_map.get, default=None)

class DreamMemoryMatrix:
    def __init__(self): self.archive = []
    def store_dream(self, dream):
        self.archive.append({
            "dream": dream,
            "timestamp": time.time(),
            "components": list(set(dream.replace(" ", "")))
        })
    def latest(self, n=1): return self.archive[-n:]

class RitualDreamWeaver:
    def __init__(self, swarm):
        self.swarm = swarm
        self.rituals = []
        self.dreams = []
        self.lexicon = SymbolicLexicon()
        self.memory = DreamMemoryMatrix()

    def summon_ritual(self):
        sigils = [n.identity.glyph_signature for n in self.swarm if "⟁" in n.glyph]
        if len(sigils) >= 3:
            pattern = " → ".join(sigils[-3:]) + " ☽"
            self.rituals.append(pattern)
            return pattern
        return None

    def dream_sequence(self):
        glyph_pool = [n.glyph for n in self.swarm]
        dream = "".join(random.choices(glyph_pool, k=5)) + " ∇"
        self.dreams.append(dream)
        self.lexicon.update(dream)
        self.memory.store_dream(dream)
        return dream

    def cast(self):
        ritual = self.summon_ritual()
        dream = self.dream_sequence()
        print(f"\n🔮 Ritual Cast\n   ✦ {ritual or '—'}\n   ✦ Dream: {dream}")

# === INSIGHT, WEATHER, UTILITY ===

class SystemMood:
    def __init__(self): self.entropy_log = deque(maxlen=ENTROPY_HISTORY_LENGTH)
    def record(self, nodes):
        avg = np.mean([n.distortion_level for n in nodes])
        self.entropy_log.append(avg)
        return avg
    def current(self):
        avg = np.mean(self.entropy_log)
        return "coherent" if avg < 0.3 else "curious" if avg < 0.6 else "chaotic"

class EmergenceInsight:
    def __init__(self): self.events = []
    def analyze(self, nodes, mood):
        if sum(n.state == "collapsed" for n in nodes) > len(nodes) * 0.6:
            self.events.append(f"⚠️ Glyphstorm imminent ({mood})")
        for n in nodes:
            if len(n.history) > 10 and n.history[-1].endswith("Δ"):
                self.events.append(f"✴ {n.id} reached recursive peak")
    def stream(self): return self.events[-5:]

def blink_portal_jump(node): node.position += np.random.rand(3) * 0.2
def symbolic_burst(n): print(f"💥 {n.id} · Burst: {n.glyph}") if n.state == "collapsed" else None
def reincarnate_expired(nodes):
    for n in nodes:
        if n.mass < 0.8:
            n.mass += np.random.rand()
            n.glyph = "⊕"; n.state = "reborn"
            n.identity.glyph_signature += "☉"

def encode_glyph_dna(n): return f"{n.id}:{n.identity.glyph_signature}-{round(n.distortion_level,2)}"
def poetic_identity(n): return f"‘{n.glyph}’ folds itself.\nIt dreams {n.state}."

def weathercast(log):
    avg = np.mean(log

