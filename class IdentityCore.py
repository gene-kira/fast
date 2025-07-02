# === AUTOSPAWN: Symbolic Warp & Identity Engine ===
import time
import numpy as np
from queue import Queue

# --- Identity Core ---
class IdentityCore:
    def __init__(self, base_id, glyph_signature):
        self.base_id = base_id
        self.glyph_signature = glyph_signature  # Symbolic identity
        self.reputation_field = np.random.rand()  # Symbolic-social gravitas

    def mutate_identity(self, distortion_field):
        entropy = np.random.rand() * distortion_field
        if entropy > 0.5:
            self.glyph_signature = f"{self.glyph_signature}⇌"
            self.reputation_field += entropy * 0.1

# --- Glyph Node with Symbolic Warp Mechanics ---
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
        self.history = []

    def update_motion(self, peers, threshold=0.15):
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

        if np.linalg.norm(self.velocity) > threshold:
            self.state = "blinked"
            self.semantic_warp()

    def semantic_warp(self):
        self.distortion_level = min(1.0, self.mass / 10.0)
        if self.distortion_level > 0.7:
            self.state = "collapsed"
            self.glyph = f"{self.glyph}Δ"
            print(f"[{self.id}] entered warp: glyph now {self.glyph}")
        self.log_state()

    def log_state(self):
        self.history.append(self.identity.glyph_signature)

# --- Symbolic Message Bus ---
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

# --- GlyphNode Reactivity & Identity Mutation ---
class ReactiveGlyph(GlyphNode):
    def react_to(self, message: SymbolicMessage):
        influence = message.distortion * np.random.rand()
        self.mass += influence
        if influence > 0.4:
            self.identity.mutate_identity(influence)
            print(f"↯ [{self.id}] morphing identity → {self.identity.glyph_signature}")
        if np.random.rand() < 0.1:
            self.semantic_warp()

# --- Peer-to-Peer Field Resonance ---
class P2PField:
    def __init__(self, agents):
        self.agents = agents

    def propagate_flux(self):
        for node in self.agents:
            influencers = [a for a in self.agents if a.id != node.id]
            flux = sum(p.identity.reputation_field for p in influencers) / len(influencers)
            node.identity.mutate_identity(distortion_field=flux)

# === PROTOTYPE LOOP ===
if __name__ == "__main__":
    print("\n🧠 INITIATING COGNITIVE WARP ENGINE...\n")

    # Initialize agents, bus, P2P field
    nodes = [ReactiveGlyph(f"Node_{i}", mass=1.0 + np.random.rand(), glyph="⟁") for i in range(8)]
    bus = SymbolicBus()
    p2p = P2PField(nodes)
    for n in nodes:
        bus.subscribe(n)

    # Main loop
    for t in range(50):
        for n in nodes:
            n.update_motion(nodes)
        msg = SymbolicMessage(origin=np.random.choice(nodes).id, payload="⊕", distortion=np.random.rand())
        bus.publish(msg)
        bus.process()
        p2p.propagate_flux()
        time.sleep(0.08)

