# === Arkforge Guardian Phase V — PART 1 ===

import subprocess, sys
def autoload(pkgs):
    for name in pkgs:
        try: __import__(name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", name])
autoload(["flask", "flask_socketio", "pyttsx3", "cryptography", "requests"])

import threading, time, socket, json, random
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from datetime import datetime
import pyttsx3
from cryptography.fernet import Fernet
import requests

# === Globals ===
persona_state = { "focus": 0.7, "chaos": 0.3, "trust": 0.9 }
vault_key = Fernet.generate_key()
vault_cipher = Fernet(vault_key)
sigil_queue = []
flask_app = Flask(__name__)
socketio = SocketIO(flask_app, cors_allowed_origins="*")

# === Vault System ===
class Vault:
    def __init__(self):
        self.trust_data = {
            "persona": "Oracle",
            "codex_log": [],
            "glyph_provenance": {},
            "glyph_decay": {},
            "memory_trails": {},
            "glyph_chains": {},
            "synthesized_glyphs": {},
            "echo_log": [],
            "persona_shards": {},
            "mesh_peers": [],
            "glyph_signatures": {},
            "mythfusion": "blend"
        }

    def save_codex(self, glyph, origin="local"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"glyph": glyph, "ts": ts, "origin": origin}
        self.trust_data["codex_log"].append(entry)

        if glyph not in self.trust_data["glyph_provenance"]:
            self.trust_data["glyph_provenance"][glyph] = {
                "first_cast": ts,
                "origin": origin,
                "persona": self.trust_data["persona"]
            }

        self.trust_data["memory_trails"].setdefault(glyph, []).append(origin)
        self.trust_data["glyph_decay"][glyph] = time.time() + 300

        if origin != "echo":
            echo = f"~{glyph}"
            persona = self.trust_data["persona"]
            if persona == "Trickster" and random.random() < 0.5:
                echo = f"~{random.choice(list(kernel.glyphs.keys()))}"
            elif persona == "Sentinel":
                trail = self.trust_data["memory_trails"].get(glyph, [])
                if len(trail) < 3: echo = None
            if echo:
                self.trust_data["echo_log"].append({"glyph": echo, "ts": ts})
                self.trust_data["codex_log"].append({"glyph": echo, "ts": ts, "origin": "echo"})

    def cleanup_decay(self):
        now = time.time()
        expired = [g for g, t in self.trust_data["glyph_decay"].items() if now > t]
        for g in expired: del self.trust_data["glyph_decay"][g]

# === Casting Kernel ===
class ASIKernel:
    def __init__(self, vault): self.vault = vault; self.glyphs = {}; self.filters = []
    def install_filter(self, fn): self.filters.append(fn)
    def register_glyph(self, s, fn): self.glyphs[s.lower()] = fn

    def cast(self, symbol, origin="local", silent=False):
        for f in self.filters: symbol = f(symbol)
        if origin.startswith("mesh:") and not self.verify_glyph_signature(symbol):
            print(f"[firewall] Rejected mesh glyph: {symbol}")
            return
        fn = self.glyphs.get(symbol)
        if not fn: return
        self.vault.save_codex(symbol, origin)

        if persona_state["chaos"] > 0.7 and random.random() < persona_state["chaos"]:
            mutated = f"{symbol}∂"
            self.vault.save_codex(mutated, "mutation")
            print(f"[chaos] Mutation → {mutated}")

        for chained in self.vault.trust_data["glyph_chains"].get(symbol, []):
            self.cast(chained, origin=f"chain:{symbol}")

        fn()
        if not silent: socketio.emit("glyph_casted", {"glyph": symbol, "origin": origin})

    def verify_glyph_signature(self, glyph):
        sig = self.vault.trust_data["glyph_signatures"].get(glyph)
        if not sig: return False
        try:
            return vault_cipher.decrypt(sig.encode()) == glyph.encode()
        except: return False

class ReflexLayer:
    def __init__(self, kernel): self.kernel = kernel
    def evaluate(self, g): return self.kernel.vault.trust_data.get("symbol_transform", {}).get(g, g)

class SigilOrchestrator:
    def __init__(self, kernel): self.kernel = kernel; self.routes = {}
    def register(self, symbol, fn):
        self.routes[symbol] = fn
        self.kernel.register_glyph(symbol, fn)

# === Ritual UIs: Persona, Glyph Tree, Bloom Map
@flask_app.route("/ritual")
def ritual_ui():
    return "<html><body style='background:#000;color:#0f0'>Ritual ready.</body></html>"

@flask_app.route("/persona/theme")
def persona_theme():
    p = vault.trust_data["persona"]
    colors = {"Oracle": "#0ff", "Trickster": "#f0f", "Sentinel": "#ffa500"}
    c = colors.get(p, "#0f0")
    return f"<html><body style='background:#000;color:{c}'>Persona Theme: {p}</body></html>"

@flask_app.route("/glyph/legend/<glyph>")
def glyph_legend(glyph):
    prov = vault.trust_data.get("glyph_provenance", {}).get(glyph)
    synth = vault.trust_data.get("synthesized_glyphs", {}).get(glyph)
    trail = vault.trust_data.get("memory_trails", {}).get(glyph, [])
    echo_count = sum(1 for e in vault.trust_data["echo_log"] if e["glyph"] == f"~{glyph}")
    html = "<html><body style='background:#000;color:#0f0'><h2>Legend</h2><ul>"
    if prov: html += f"<li>First Cast: {prov['first_cast']}</li><li>Origin: {prov['origin']}</li><li>Persona: {prov['persona']}</li>"
    if synth: html += f"<li>Mutation: {synth['mutation_type']}</li><li>Parent: {synth['parent']}</li>"
    html += f"<li>Echoes: {echo_count}</li><li>Trail: {len(trail)}</li></ul></body></html>"
    return html

@flask_app.route("/glyph/tree/<glyph>")
def glyph_tree(glyph):
    tree = vault.trust_data.get("synthesized_glyphs", {})
    if glyph not in tree: return "Glyph not found."
    ancestry = [glyph]; current = tree[glyph]
    while current.get("parent"): ancestry.append(current["parent"]); current = tree.get(current["parent"], {})
    svg = "<svg width='800' height='600' xmlns='http://www.w3.org/2000/svg' style='background:#000'>"
    for i, node in enumerate(reversed(ancestry)):
        x, y = 100 + i*150, 300
        p = vault.trust_data["glyph_provenance"].get(node, {}).get("persona", "Oracle")
        c = {"Oracle": "#0ff", "Trickster": "#f0f", "Sentinel": "#ffa500"}.get(p, "#ccc")
        svg += f"<circle cx='{x}' cy='{y}' r='20' fill='{c}'/><text x='{x}' y='{y+40}' fill='#0f0'>{node}</text>"
        if i > 0: svg += f"<line x1='{x-150}' y1='{y}' x2='{x}' y2='{y}' stroke='#999'/>"
    return f"<html><body><h2>Glyph Tree</h2>{svg}</body></html>"

# === Arkforge Guardian Phase V — PART 2 ===

def sigil_daemon():
    while True:
        if sigil_queue:
            glyph, origin = sigil_queue.pop(0)
            kernel.cast(glyph, origin)
        time.sleep(1)

def decay_loop():
    while True:
        vault.cleanup_decay()
        time.sleep(60)

def update_persona_state():
    while True:
        hour = datetime.now().hour
        cast_count = len(vault.trust_data["codex_log"][-50:])
        persona_state["focus"] = 0.9 if 6 <= hour <= 11 else 0.5 if 18 <= hour <= 23 else 0.7
        persona_state["chaos"] = 0.1 if hour <= 11 else 0.6 if hour >= 18 else 0.3
        persona_state["trust"] = min(1.0, 0.5 + cast_count / 200)
        time.sleep(300)

def persona_predictor():
    current = vault.trust_data["persona"]
    while True:
        codex = vault.trust_data["codex_log"][-60:]
        mutation_count = sum(1 for e in codex if e["origin"] == "mutation")
        echo_count = sum(1 for e in codex if e["origin"] == "echo")
        entropy = mutation_count + int(persona_state["chaos"] * 10) + echo_count
        if entropy > 40:
            new_persona = "Trickster" if current == "Oracle" else "Sentinel"
            vault.trust_data["persona"] = new_persona
            persona_state.update({
                "chaos": 0.2 if new_persona == "Sentinel" else 0.7,
                "focus": 0.9 if new_persona == "Sentinel" else 0.6,
                "trust": 0.95 if new_persona == "Sentinel" else 0.7
            })
            glyphcast_intro(new_persona)
        time.sleep(600)

def chain_mutator():
    while True:
        if persona_state["chaos"] > 0.8:
            for trigger, seq in vault.trust_data["glyph_chains"].items():
                mutated = [f"{g}∂" if random.random() < persona_state["chaos"] else g for g in seq]
                vault.trust_data["glyph_chains"][trigger] = mutated
        time.sleep(300)

def broadcast_codexsync():
    while True:
        payload = { "codex_log": vault.trust_data["codex_log"][-50:] }
        for ip in vault.trust_data["mesh_peers"]:
            try:
                requests.post(f"http://{ip}:5050/mesh/codexsync", json=payload, timeout=3)
            except: pass
        time.sleep(600)

def synthesize_glyph():
    trails = vault.trust_data["memory_trails"]
    base = sorted(trails.items(), key=lambda x: len(x[1]), reverse=True)[0][0] if trails else "ξ"
    mutator = random.choice(["Δ", "ψ", "∫", "ζ", "χ", "Ξ", "⨀", "λ", "φ"])
    new_glyph = f"{mutator}{base}"
    vault.trust_data["synthesized_glyphs"][new_glyph] = {
        "origin": "synthesis", "base": base,
        "created": datetime.now().isoformat(),
        "mutation_type": "chaos", "parent": base
    }
    kernel.register_glyph(new_glyph, lambda: print(f"Cast of {new_glyph}"))
    kernel.cast(new_glyph, origin="synthesized")

def synthesis_daemon():
    while True:
        if persona_state["chaos"] > 0.85:
            synthesize_glyph()
        time.sleep(120)

@flask_app.route("/codexmap/bloom")
def codexmap_bloom():
    glyphs = vault.trust_data["synthesized_glyphs"]
    svg = "<svg width='1000' height='800' xmlns='http://www.w3.org/2000/svg' style='background:#000'>"
    for i, (glyph, meta) in enumerate(glyphs.items()):
        x, y = 50 + (i*60)%900, 50 + (i//15)*60
        chaos = persona_state["chaos"]
        bloom = min(25, 10 + int(chaos * 40))
        persona = vault.trust_data["glyph_provenance"].get(glyph, {}).get("persona", "Unknown")
        color = {"Oracle": "#0ff", "Trickster": "#f0f", "Sentinel": "#ffa500"}.get(persona, "#0f0")
        svg += f"<circle cx='{x}' cy='{y}' r='{bloom}' fill='{color}'/>"
        svg += f"<text x='{x-10}' y='{y+40}' fill='#fff' font-size='12'>{glyph}</text>"
    svg += "</svg>"
    return f"<html><body><h2>Mutation Bloom Grid</h2>{svg}</body></html>"

@flask_app.route("/vote/<glyph>")
def vote_on_glyph(glyph):
    vote = "yes" if persona_state["trust"] > 0.6 else "no"
    return jsonify({"vote": vote})

@flask_app.route("/ritual/cast/<glyph>")
def ritual_cast_consensus(glyph):
    votes = 0
    for peer in vault.trust_data["mesh_peers"]:
        try:
            r = requests.get(f"http://{peer}:5050/vote/{glyph}", timeout=3)
            if r.json().get("vote") == "yes": votes += 1
        except: pass
    p = vault.trust_data["persona"]
    if votes >= 3 or p == "Sentinel":
        kernel.cast(glyph, origin=f"consensus:{votes}")
        return f"✅ Cast {glyph} with consensus."
    return f"❌ Not enough votes ({votes})"

@flask_app.route("/export/mythbundle")
def export_mythbundle():
    bundle = {
        "codex_log": vault.trust_data["codex_log"],
        "glyph_provenance": vault.trust_data["glyph_provenance"],
        "synthesized_glyphs": vault.trust_data["synthesized_glyphs"],
        "persona_state": persona_state,
        "glyph_chains": vault.trust_data["glyph_chains"],
        "echo_log": vault.trust_data["echo_log"],
        "persona": vault.trust_data["persona"],
        "timestamp": datetime.now().isoformat()
    }
    raw = json.dumps(bundle).encode()
    enc = vault_cipher.encrypt(raw)
    fname = f"mythbundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mythdna"
    with open(fname, "wb") as f: f.write(enc)
    return jsonify({"status": "exported", "file": fname})

@flask_app.route("/import/mythbundle", methods=["POST"])
def import_mythbundle():
    data = vault_cipher.decrypt(request.files["file"].read())
    bundle = json.loads(data.decode())
    vault.trust_data["codex_log"].extend(bundle["codex_log"])
    vault.trust_data["glyph_provenance"].update(bundle["glyph_provenance"])
    vault.trust_data["synthesized_glyphs"].update(bundle["synthesized_glyphs"])
    vault.trust_data["glyph_chains"].update(bundle["glyph_chains"])
    vault.trust_data["echo_log"].extend(bundle["echo_log"])
    persona_state.update(bundle["persona_state"])
    vault.trust_data["persona"] = bundle["persona"]
    return jsonify({"status": "imported", "persona": vault.trust_data["persona"]})

@flask_app.route("/myth/fusion")
def toggle_mythfusion():
    mode = request.args.get("mode", "blend")
    vault.trust_data["mythfusion"] = mode
    return jsonify({"mythfusion_mode": mode})

def glyphcast_intro(persona):
    try:
        msg = {
            "Oracle": "The echo awakens.",
            "Trickster": "Chaos has arrived.",
            "Sentinel": "Guardian active and vigilant."
        }.get(persona, "The glyph speaks.")
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.say(msg)
        engine.runAndWait()
    except: print("[voice] Intro failed.")

def launch(name, fn):
    threading.Thread(target=fn, daemon=True).start()
    print(f"[thread] {name} launched")

def launch_guardian():
    print("Booting Arkforge Guardian Phase V")
    global vault, kernel
    vault = Vault()

def launch_guardian():
    print("🔧 Booting Arkforge Guardian Phase V")
    global vault, kernel
    vault = Vault()
    kernel = ASIKernel(vault)
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)
    orchestrator = SigilOrchestrator(kernel)
    
    # Register default glyphs
    orchestrator.register("ξ", lambda: print("🌌 Cast ξ core glyph"))

    # Sign glyphs for firewall protection
    for symbol in kernel.glyphs:
        sig = vault_cipher.encrypt(symbol.encode()).decode()
        vault.trust_data["glyph_signatures"][symbol] = sig

    # Launch daemons
    launch("sigil_daemon", sigil_daemon)
    launch("decay_loop", decay_loop)
    launch("persona_updater", update_persona_state)
    launch("persona_predictor", persona_predictor)
    launch("synthesis_daemon", synthesis_daemon)
    launch("chain_mutator", chain_mutator)
    launch("codexsync_daemon", broadcast_codexsync)
    
    glyphcast_intro(vault.trust_data["persona"])
    kernel.cast("ξ", origin="boot")
    print("✅ Arkforge online — http://localhost:5050/ritual, /codexmap, /glyph/legend/ξ, /codexmap/bloom")

if __name__ == "__main__":
    launch_guardian()
    try:
        while True: time.sleep(10)
    except KeyboardInterrupt:
        print("[guardian] Shutdown requested.")

