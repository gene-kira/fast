

🜍 arknet_sim.py – Unified Mythic Swarm Orchestrator
import threading
import time
import random
from collections import deque
from arknet_node import ArkNode, entropy_score

# === ArkHive Definition ===
class ArkHive:
    """
    ArkHive: Glyphic Memory Field for ArkNode swarm.

    Responsibilities:
    - Collect pulses from all nodes.
    - Track active glyph signatures and evolving personas.
    - Detect swarm-wide entropy divergence or mythic convergence.
    """

    def __init__(self):
        self.node_registry = {}
        self.pulse_log = deque(maxlen=200)
        self.global_entropy = []

    def register_pulse(self, pulse):
        nid = pulse["sender_id"]
        self.node_registry[nid] = {
            "glyph": pulse["glyph"],
            "state": pulse["state"],
            "persona": pulse.get("persona"),
            "last_seen": time.time()
        }
        self.pulse_log.append(pulse)
        self.track_entropy()

    def track_entropy(self):
        if len(self.pulse_log) < 5:
            return
        glyphs = ''.join(p["glyph"] for p in list(self.pulse_log)[-10:])
        ent = entropy_score(glyphs)
        self.global_entropy.append(ent)
        if ent > 2.8:
            print(f"\n🌐 ArkHive Entropy Rising: {ent:.2f}")
        else:
            print(f"\n🌐 ArkHive Stable: {ent:.2f}")

    def display_resonance(self):
        print("\n🜋 ARKHIVE RESONANCE MAP 🜋")
        for nid, node in self.node_registry.items():
            print(f"{nid}: {node['glyph']} [{node['state']}] :: {node.get('persona')}")
        print("-" * 32)

# === Simulation Setup ===
def create_node(index, port_base, arkhive):
    node_id = f"ark_{index:03}"
    node = ArkNode(node_id=node_id, port=port_base + index)
    
    # Wrap original emit_pulse to forward to ArkHive
    original_emit = node.emit_pulse

    def wrapped_emit():
        sock = node.create_glyph_pulse()
        arkhive.register_pulse(sock)
        original_emit()

    node.emit_pulse = wrapped_emit
    node.start()
    return node

def run_arknet_sim(num_nodes=3, runtime=60):
    arkhive = ArkHive()
    port_base = 9000
    nodes = [create_node(i, port_base, arkhive) for i in range(num_nodes)]

    try:
        start_time = time.time()
        while time.time() - start_time < runtime:
            time.sleep(10)
            arkhive.display_resonance()
    except KeyboardInterrupt:
        print("\n🔻 Terminating simulation.")
    finally:
        for node in nodes:
            node.stop()

# === Entry Point ===
if __name__ == "__main__":
    run_arknet_sim(num_nodes=5, runtime=90)



✅ Requirements:
- Ensure arknet_node.py contains your updated ArkNode with glyph mutation, dreaming, persona assignment, and entropy_score.
- This orchestrator:
- Spawns multiple nodes with different ports
- Hooks ArkHive into their pulse stream
- Periodically shows resonance states + entropy scores


