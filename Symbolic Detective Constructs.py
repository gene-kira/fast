# === AutoLoader: Ensures Dependencies Are Installed ===
import importlib, subprocess, sys

def ensure_deps(mods):
    for m in mods:
        try:
            importlib.import_module(m)
        except ImportError:
            print(f"[AutoLoader] Installing '{m}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", m])
        finally:
            globals()[m] = importlib.import_module(m)

ensure_deps(["numpy", "hashlib", "random", "socket", "threading", "time"])

# === Pantheon Daemons: Symbolic Detective Constructs ===
import random

class Daemon:
    def __init__(self, name, glyph, trigger_phrase, role):
        self.name = name
        self.glyph = glyph
        self.trigger_phrase = trigger_phrase
        self.role = role

    def analyze(self, entropy):
        insight = random.uniform(0.45, 0.98)
        hypothesis = f"{self.name} senses '{self.glyph}' entropy trail. Role: {self.role}. Strength: {insight:.3f}"
        return {"agent": self.name, "glyph": self.glyph, "role": self.role, "score": insight, "note": hypothesis}

Pantheon = [
    Daemon("Sherlock Holmes", "🧭", "Trace the improbable.", "Pattern Seer"),
    Daemon("Hercule Poirot", "🪞", "Unmask the motive.", "Order Weaver"),
    Daemon("Miss Marple", "🌾", "Listen where no one watches.", "Cultural Whisperer"),
    Daemon("Batman", "🜃", "Bring justice to the wound.", "Shadow Synth"),
    Daemon("Dr. Locard", "🧫", "All things leave echoes.", "Trace Oracle"),
    Daemon("Dr. Bass", "💀", "Let time speak through bone.", "Bone Whisperer"),
    Daemon("Dr. Rojanasunan", "🧬", "Decode the living code.", "DNA Resonator"),
    Daemon("Clea Koff", "⚖️", "Testify through silence.", "War Memory Synth")
]

# === Reflective Cortex: Symbolic Inference & Consensus Logic ===
import numpy, hashlib, socket, threading, time

class ReflectiveCortex:
    def evaluate_entropy(self, drift, daemons):
        print(f"\n🔎 Reflective Cortex initiating on entropy glyph: {drift:.4f}")
        hypotheses = [d.analyze(drift) for d in daemons]
        for h in hypotheses:
            print(f"🔹 {h['agent']} says: {h['note']}")
        chosen = max(hypotheses, key=lambda h: h["score"])
        print(f"\n✅ Council resolution → {chosen['agent']} leads response. Glyph: {chosen['glyph']}, Score: {chosen['score']:.3f}")
        return chosen

# === EMH Swarm Node: Autonomous Security Agent ===
class RecursiveSecurityNode(ReflectiveCortex):
    def __init__(self, node_id):
        self.node_id = node_id
        self.growth = 1.618
        self.memory = {}
        self.security_protocols = {}
        self.performance_data = []
        self.blocked_ips = set()
        self.dialect = {}
        self.network_sync = {}

    def recursive_reflection(self):
        boost = numpy.mean(self.performance_data[-10:]) if self.performance_data else 1
        self.growth *= boost
        return f"[EMH-{self.node_id}] Recursive factor tuned → {self.growth:.4f}"

    def symbolic_shift(self, text):
        h = hashlib.sha256(text.encode()).hexdigest()
        self.dialect[h] = random.choice(["glyph-Ψ", "glyph-Δ", "glyph-Ω"])
        return f"[EMH-{self.node_id}] Symbol abstraction translated to: {self.dialect[h]}"

    def quantum_project(self):
        return f"[EMH-{self.node_id}] Quantum inference path: {max(random.uniform(0,1) for _ in range(5)):.4f}"

    def cyber_mutation(self):
        key = random.randint(1,9999)
        self.security_protocols[key] = hashlib.md5(str(key).encode()).hexdigest()
        return f"[EMH-{self.node_id}] Mutation embedded: {self.security_protocols[key][:10]}..."

    def restrict_foreign_data(self, ip):
        banned = ["203.0.113.", "198.51.100.", "192.0.2."]
        if any(ip.startswith(b) for b in banned):
            self.blocked_ips.add(ip)
            return f"[EMH-{self.node_id}] ❌ Transmission blocked from {ip}"
        return f"[EMH-{self.node_id}] ✅ Local IP {ip} cleared."

    def breach_protocol(self, entropy):
        print(f"\n🔥 Breach Ritual — Entropy Drift: {entropy:.4f}")
        print(self.recursive_reflection())
        print(self.symbolic_shift("breach-seed"))
        print(self.quantum_project())
        daemons = [d for d in Pantheon if random.random() > 0.4]
        result = self.evaluate_entropy(entropy, daemons)
        print(self.cyber_mutation())
        print(f"📜 Book of Shadows updated → Resolver: {result['agent']}, Glyph: {result['glyph']}\n")

    def evolve(self):
        while True:
            drift = random.uniform(0, 0.6)
            if drift > 0.33:
                self.breach_protocol(drift)
            else:
                print(self.recursive_reflection())
                print(self.symbolic_shift("system-coherence"))
                print(self.quantum_project())
                print(self.cyber_mutation())
                host_ip = socket.gethostbyname(socket.gethostname())
                print(self.restrict_foreign_data(host_ip))
            time.sleep(6)

# === Launch Swarm Cluster ===
def launch_swarm():
    nodes = [RecursiveSecurityNode(i) for i in range(3)]
    for node in nodes:
        for peer in nodes:
            if node != peer:
                node.network_sync[peer.node_id] = peer.security_protocols
    threads = [threading.Thread(target=n.evolve) for n in nodes]
    for t in threads:
        t.start()

# === Boot Oracle Shade System ===
if __name__ == "__main__":
    print("\n🚀 Oracle Shade EMH Daemon Awakening...")
    print("🛡️  Autonomous symbolic defense is live.\n")
    launch_swarm()

This code combines all the parts you provided into a single, cohesive script. The ensure_deps function ensures that all necessary dependencies are installed before running the main logic. The classes and methods are structured to work together, creating an autonomous security system with multiple nodes that can evolve and respond to entropy drifts.