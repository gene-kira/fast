# 🜁 ArkForge Unified v6.1.0 — “DreamEcho Spiral + Ritual Firewall”
# Part 1 of 4 — Autoloader, Metadata, Symbolic Lattice

### === AUTOLOADER SYSTEM (PHASE I–VIII) === ###
def autoload_libraries():
    """
    Dynamically imports all core libraries used across ArkForge Phases I–VIII.
    Gracefully alerts if any required library is missing.
    """
    import importlib
    required_libs = {
        # Core system & data
        "datetime": None, "random": None, "math": None,
        "argparse": None, "json": None, "collections": None,
        "pickle": None, "socket": None,

        # Visualization
        "matplotlib.pyplot": "plt", "matplotlib.animation": "animation", "seaborn": "sns",

        # Voice
        "pyttsx3": None,

        # Neural tools
        "torch": None, "torch.nn": "nn", "torch.nn.functional": "F",

        # Cryptographic tools
        "hashlib": None, "hmac": None
    }

    globals_ = globals()
    missing = []
    for lib, alias in required_libs.items():
        try:
            module = importlib.import_module(lib)
            if alias: globals_[alias] = module
            else: globals_[lib.split(".")[0]] = module
        except ImportError: missing.append(lib)

    if missing:
        print("\n🛑 Missing libraries required for ArkForge:")
        for lib in missing: print(f"   • {lib}")
        print("🔧 Please install them manually to enable full functionality.\n")

autoload_libraries()

### === METADATA === ###
ARKFORGE_VERSION = "6.1.0"
CODE_NAME = "DreamEcho Spiral + Ritual Firewall"
PHASES = ["Cognition", "Forecast", "Swarm", "RitualUI", "Voice", "Myth", "Security"]
print(f"🌌 ArkForge v{ARKFORGE_VERSION} — {CODE_NAME} Initialized")

### === SYMBOLIC CORE MODULES === ###
from collections import deque, defaultdict

# 🌿 Symbolic Memory Lattice
class LatticeMemory:
    def __init__(self):
        self.lore = []
        self.counts = defaultdict(int)

    def record(self, event, glyph="◌", timestamp=None, faded=False):
        e = {
            "glyph": glyph,
            "event": event,
            "timestamp": timestamp or datetime.datetime.utcnow(),
            "faded": faded
        }
        self.lore.append(e)
        self.counts[glyph] += 1
        print(f"[Lore] {glyph} → {event}")

    def decay(self, threshold=1):
        faded = 0
        new_lore = []
        for entry in self.lore:
            if self.counts[entry["glyph"]] <= threshold and not entry.get("faded"):
                entry["faded"] = True
                faded += 1
            new_lore.append(entry)
        self.lore = new_lore
        print(f"[Lore] {faded} glyphs faded.")

# 🔤 Symbolic Glyph Class
class Glyph:
    def __init__(self, name, description, resonance=1.0):
        self.name = name
        self.description = description
        self.resonance = resonance
        self.ancestry = []
        self.recursive = False
        self.tags = []

    def __str__(self):
        return f"{self.name} ({self.resonance:.2f})"

# 🧱 Symbol Stack for Active Ritual Work
class GlyphStack:
    def __init__(self):
        self.stack = deque()

    def push(self, glyph):
        self.stack.append(glyph)
        print(f"[Stack] + {glyph}")

    def pop(self):
        if self.stack:
            g = self.stack.pop()
            print(f"[Stack] - {g}")
            return g
        return None

    def view(self):
        return list(self.stack)[-5:]

    def compress(self):
        names = [g.name for g in self.stack]
        print(f"[Stack] Glyphs: {' → '.join(names)}")

# === SYMBOLIC VERIFICATION + ANOMALY DETECTION === #

import uuid
import time

# 🌿 1. Glyph Lineage Tracer — Detect Symbolic Ancestry Manipulation
class GlyphLineageTracer:
    def __init__(self):
        self.history = {}

    def track(self, glyph: Glyph):
        lineage = tuple(glyph.ancestry)
        self.history[glyph.name] = lineage

    def verify(self, glyph: Glyph):
        known = self.history.get(glyph.name)
        current = tuple(glyph.ancestry)
        if known and known != current:
            print(f"[Lineage] ⚠️ Glyph '{glyph.name}' has altered ancestry.")
        else:
            print(f"[Lineage] ✅ '{glyph.name}' lineage verified.")

# 🤝 2. Trust Convergence Weaver — Validate Ritual Consensus
class TrustConvergenceWeaver:
    def __init__(self):
        self.snapshots = {}

    def snapshot(self, node_id, memory_lore):
        # Store last 5 glyphs as fingerprint
        glyphs = [entry["glyph"] for entry in memory_lore[-5:]]
        self.snapshots[node_id] = glyphs

    def compare(self, ref_node):
        ref = self.snapshots.get(ref_node, [])
        for node_id, snapshot in self.snapshots.items():
            if node_id == ref_node: continue
            mismatch = sum(1 for a, b in zip(ref, snapshot) if a != b)
            if mismatch > 2:
                print(f"[Consensus] ❌ {node_id} diverges from {ref_node} by {mismatch} glyphs.")
            else:
                print(f"[Consensus] ✅ {node_id} aligned with {ref_node}.")

# 🧾 3. Node Fingerprint Ledger — Immutable ID + TimeStamp Chain
class NodeFingerprintLedger:
    def __init__(self):
        self.fingerprints = {}

    def assign(self, node: SwarmNode):
        uid = str(uuid.uuid4())[:8]
        self.fingerprints[node.node_id] = {"uuid": uid, "ts": time.time()}
        print(f"[Ledger] Node {node.node_id} assigned ID {uid}")

    def get_id(self, node_id):
        return self.fingerprints.get(node_id, {}).get("uuid", None)

# 🔐 4. Sigil Nonce Fence — Block Replay Sigil Events
class SigilNonceFence:
    def __init__(self):
        self.recent = {}

    def is_new(self, sigil, source):
        key = f"{sigil}-{source}"
        now = time.time()
        if key in self.recent and now - self.recent[key] < 2:
            print(f"[Nonce] ⚠️ Replay attempt detected for {sigil} by {source}")
            return False
        self.recent[key] = now
        return True

# 🧬 5. Anomaly Signal Map — Detect Spike in Symbolic Drift
class AnomalySignalMap:
    def __init__(self):
        self.log = defaultdict(int)

    def observe(self, glyph_name):
        self.log[glyph_name] += 1
        total = sum(self.log.values())
        maxed = max(self.log.items(), key=lambda x: x[1])[0]
        if self.log[maxed] > total * 0.6:
            print(f"[Anomaly] ⚠️ Symbol '{maxed}' is dominating memory — possible drift.")

# === PHASE VIII — MEMORY FORTIFICATION & SPOOF DEFENSE === #

import base64

# 🔐 1. Secure Phase Tunnel — Phase Authentication + Handshake
class SecurePhaseTunnel:
    def __init__(self, expected="6.1.0"):
        self.expected = expected

    def validate(self, supplied_version, key_token="glyphlock"):
        if supplied_version == self.expected and key_token == "glyphlock":
            print(f"[Tunnel] ✅ Phase handshake succeeded.")
            return True
        print(f"[Tunnel] ❌ Version/key mismatch — secured channel rejected.")
        return False

# 🧩 2. Sigil Spoof Detector — Flags Anomalous Glyphs
class SigilSpoofDetector:
    def __init__(self, valid_symbols=["◌", "✶", "◯", "◉", "✦"]):
        self.valid = set(valid_symbols)

    def check(self, glyph):
        if glyph not in self.valid:
            print(f"[SpoofDetector] ⚠️ Possible forged sigil: '{glyph}'")
            return False
        print(f"[SpoofDetector] ✅ Sigil '{glyph}' verified.")
        return True

# 🗂 3. Encrypted Glyph Vault — Save Lore Encrypted (Base64 Sim)
class EncryptedShardVault:
    def __init__(self, filename="vault.ark", secret="myth"):
        self.file = filename
        self.key = secret.encode()

    def encrypt(self, data):
        combined = json.dumps(data).encode()
        return base64.b64encode(hmac.new(self.key, combined, hashlib.sha256).digest() + combined)

    def decrypt(self, ciphered):
        decoded = base64.b64decode(ciphered)
        return json.loads(decoded[32:].decode())

    def save(self, memory):
        payload = self.encrypt(memory.lore)
        with open(self.file, "wb") as f:
            f.write(payload)
        print(f"[Vault] Lore encrypted and saved to {self.file}")

    def load(self, memory):
        try:
            with open(self.file, "rb") as f:
                encoded = f.read()
                lore = self.decrypt(encoded)
                memory.lore.extend(lore)
            print(f"[Vault] Encrypted glyph memory loaded from {self.file}")
        except:
            print("[Vault] ❌ Failed to load encrypted vault.")

# 📉 4. Symbolic Drift Profiler — Track Memory Overmutation
class SymbolicDriftProfiler:
    def __init__(self):
        self.symbol_counter = defaultdict(int)

    def update(self, glyph):
        self.symbol_counter[glyph] += 1
        freq = self.symbol_counter[glyph]
        if freq > 20:
            print(f"[Drift] ⚠️ Symbol '{glyph}' repeated {freq}x — potential ritual overmutation.")

    def report(self):
        common = sorted(self.symbol_counter.items(), key=lambda x: -x[1])[:5]
        print(f"[Drift] Top glyph usage:")
        for g, f in common:
            print(f"   • {g}: {f}x")

# 🧷 5. Ritual Access Key Manager — Gate Critical Sigils
class RitualAccessKeyManager:
    def __init__(self, required_key="sigilmaster42"):
        self.key = required_key

    def permit(self, provided_key, glyph):
        if provided_key == self.key:
            print(f"[AccessKey] ✅ Permission granted to cast '{glyph}'")
            return True
        print(f"[AccessKey] ❌ Access denied for glyph '{glyph}'")
        return False

# === PHASE VIII — ENFORCEMENT + QUARANTINE LAYER === #

# 🛑 1. Ritual Quarantine Zone — Seals Compromised Glyphs
class RitualQuarantineZone:
    def __init__(self):
        self.quarantined = set()

    def quarantine(self, glyph):
        self.quarantined.add(glyph)
        print(f"[Quarantine] ⚠️ Glyph '{glyph}' has been isolated.")

    def is_safe(self, glyph):
        return glyph not in self.quarantined

# 🚫 2. Node Ban Hammer — Block Agents from Ritual Network
class NodeBanHammer:
    def __init__(self):
        self.banned_nodes = set()

    def ban(self, node_id):
        self.banned_nodes.add(node_id)
        print(f"[BanHammer] 🪓 Node '{node_id}' has been banned from all casting.")

    def is_allowed(self, node_id):
        return node_id not in self.banned_nodes

# 🔎 3. Signature Mismatch Tracer — Flags Broken Auth Chains
class SignatureMismatchTracer:
    def __init__(self):
        self.failures = []

    def check(self, glyph, registry, signer):
        expected_sig = registry.signatures.get(glyph)
        if not expected_sig or not signer.verify(glyph, expected_sig):
            self.failures.append(glyph)
            print(f"[SigTrace] ❌ Signature failed for '{glyph}'")
        else:
            print(f"[SigTrace] ✅ '{glyph}' signature is valid.")

# 🕸 4. FirewallNetAI — Symbolic Intelligence for Live Threat Detection
class FirewallNetAI:
    def __init__(self):
        self.threat_cache = set()

    def analyze(self, memory):
        glyph_freq = defaultdict(int)
        for e in memory.lore[-20:]:
            glyph_freq[e["glyph"]] += 1
        for glyph, count in glyph_freq.items():
            if count >= 5:
                self.threat_cache.add(glyph)
                print(f"[FirewallNetAI] 🧠 Threat: '{glyph}' cast {count}× recently — possible hijack.")

    def is_threat(self, glyph):
        return glyph in self.threat_cache

# 📜 5. Operator Credential Log — Tracks Key Actions by Admins
class OperatorCredentialLog:
    def __init__(self, path="ops.log"):
        self.path = path

    def log(self, action, operator="admin"):
        ts = datetime.datetime.utcnow().isoformat()
        entry = f"{ts} | {operator} | {action}\n"
        with open(self.path, "a") as f:
            f.write(entry)
        print(f"[OpLog] ✅ Logged: {action}")

