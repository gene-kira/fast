# === Arkforge Guardian Phase IV — PART 1 ===

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
persona = "Oracle"
persona_state = { "focus": 0.7, "chaos": 0.3, "trust": 0.9 }
vault_key = Fernet.generate_key()
vault_cipher = Fernet(vault_key)
sigil_queue = []

# === Vault ===
class Vault:
    def __init__(self):
        self.trust_data = {
            "persona": persona,
            "codex_log": [],
            "glyph_decay": {},
            "memory_trails": {},
            "glyph_chains": {},
            "symbol_transform": {},
            "mutation_rules": [],
            "echo_log": [],
            "persona_overlay": {},
            "persona_shards": {},
            "mesh_peers": [],
            "external_hooks": { "Σ": "https://api.chucknorris.io/jokes/random" },
            "synthesized_glyphs": {}
        }

    def save_codex(self, glyph, origin="local"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.trust_data["codex_log"].append({"glyph": glyph, "ts": ts, "origin": origin})
        self.trust_data["memory_trails"].setdefault(glyph, []).append(origin)
        self.trust_data["glyph_decay"][glyph] = time.time() + 300
        if origin != "echo":
            echo = f"~{glyph}"
            self.trust_data["echo_log"].append({"glyph": echo, "ts": ts})
            self.trust_data["codex_log"].append({"glyph": echo, "ts": ts, "origin": "echo"})

    def cleanup_decay(self):
        now = time.time()
        expired = [g for g, t in self.trust_data["glyph_decay"].items() if now > t]
        for g in expired: del self.trust_data["glyph_decay"][g]

# === Kernel ===
class ASIKernel:
    def __init__(self, vault): self.vault = vault; self.glyphs = {}; self.filters = []
    def install_filter(self, fn): self.filters.append(fn)
    def register_glyph(self, s, fn): self.glyphs[s.lower()] = fn

    def cast(self, symbol, origin="local", silent=False):
        for f in self.filters: symbol = f(symbol)
        fn = self.glyphs.get(symbol)
        if not fn: return

        if origin.startswith("mesh:") and persona_state["trust"] < 0.5:
            print(f"[trust] Rejecting mesh glyph {symbol} (trust={persona_state['trust']})")
            return

        self.vault.save_codex(symbol, origin)

        if persona_state["chaos"] > 0.7 and random.random() < persona_state["chaos"]:
            mutated = f"{symbol}∂"
            self.vault.save_codex(mutated, "mutation")
            print(f"[chaos] Mutation → {mutated}")

        for chained in self.vault.trust_data["glyph_chains"].get(symbol, []):
            self.cast(chained, origin=f"chain:{symbol}")

        fn()
        if not silent: socketio.emit("glyph_casted", {"glyph": symbol, "origin": origin})
        if symbol in self.vault.trust_data["external_hooks"]: self.cast_external(symbol)

    def cast_external(self, glyph):
        url = self.vault.trust_data["external_hooks"][glyph]
        try:
            resp = requests.get(url, timeout=3)
            data = resp.json()
            echo = data.get("value", str(data))
            print(f"[external:{glyph}] {echo}")
        except Exception as e:
            print(f"[external:{glyph}] Failed → {e}")

class ReflexLayer:
    def __init__(self, kernel): self.kernel = kernel
    def evaluate(self, g): return self.kernel.vault.trust_data["symbol_transform"].get(g, g)

class SigilOrchestrator:
    def __init__(self, kernel):
        self.kernel = kernel
        self.routes = {
            "σ": lambda: print("🔥 Terminate"),
            "ξ": lambda: print("🦉 Watcher cast"),
            "Σ": lambda: print("📡 Fetching external signal..."),
            "λ": lambda: print(f"[λ] Persona: {kernel.vault.trust_data['persona']}"),
            "κ": self.export_codex,
            "μ": lambda: print(json.dumps(kernel.vault.trust_data, indent=2)),
            "ρ": lambda: self.chain("ξ", ["Σ", "λ"]),
            "τ": lambda: self.overlay("Oracle", {"ξ": "Ξ"}),
            "π": lambda: self.mutate("ξ", "ζ"),
            "ζ": lambda: print(json.dumps(kernel.vault.trust_data["memory_trails"], indent=2)),
            "ε": lambda: print(kernel.vault.trust_data["echo_log"]),
            "ψ": lambda: print(json.dumps(persona_state, indent=2))
        }

    def export_codex(self):
        raw = json.dumps(kernel.vault.trust_data["codex_log"]).encode()
        enc = vault_cipher.encrypt(raw)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"codex_{stamp}.codex", "wb") as f: f.write(enc)

    def chain(self, trigger, seq): self.kernel.vault.trust_data["glyph_chains"][trigger] = seq
    def overlay(self, name, map): self.kernel.vault.trust_data["persona_overlay"] = map
    def mutate(self, frm, to): self.kernel.vault.trust_data["mutation_rules"].append({"from": frm, "to": to})

# === Flask UI ===
flask_app = Flask(__name__)
socketio = SocketIO(flask_app, cors_allowed_origins="*")

@flask_app.route("/ritual")
def ritual_ui():
    return """
    <html><body style='background:#000;color:#0f0;font-family:monospace'>
    <h2>🜂 Ritual Interface</h2>
    <button onclick="fetch('/sigil/ξ',{method:'POST'})">Cast ξ</button>
    <button onclick="fetch('/sigil/Σ',{method:'POST'})">Cast Σ</button>
    <script src="https://cdn.socket.io/4.3.2/socket.io.min.js"></script>
    <script>var socket = io(); socket.on("glyph_casted", d => console.log("Cast:", d));</script>
    </body></html>
    """

@flask_app.route("/dashboard")
def dashboard():
    decay = vault.trust_data["glyph_decay"]
    codex = vault.trust_data["codex_log"][-10:]
    html = "<html><body style='background:#000;color:#0f0;font-family:monospace'>"
    html += "<h2>📊 Dashboard</h2><ul>"
    for g, t in decay.items():
        html += f"<li>{g} expires in {int(t - time.time())}s</li>"
    html += "</ul><hr><ul>"
    for entry in codex:
        html += f"<li>{entry['ts']} — {entry['glyph']} from {entry['origin']}</li>"
    html += "</ul></body></html>"
    return html

# === Arkforge Guardian Phase IV — PART 2 ===

# === Daemons ===
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

def run_flask():
    socketio.run(flask_app, host="0.0.0.0", port=5050)

def adaptive_score(glyph):
    trails = vault.trust_data["memory_trails"].get(glyph, [])
    echo_count = sum(1 for e in vault.trust_data["codex_log"] if e["glyph"] == f"~{glyph}")
    origin_set = set(trails)
    decay = vault.trust_data["glyph_decay"].get(glyph, 0) - time.time()
    decay_factor = max(0.1, decay / 300)
    return (len(trails) + len(origin_set) + echo_count) * decay_factor

def smart_scheduler():
    while True:
        scores = {g: adaptive_score(g) for g in kernel.glyphs if not g.startswith("~")}
        top = sorted(scores.items(), key=lambda x: -x[1])[:2]
        for g, s in top:
            print(f"[scheduler] Adaptive cast: {g} (score={s:.2f})")
            kernel.cast(g, origin="adaptive")
        time.sleep(180)

def update_persona_state():
    while True:
        now = datetime.now()
        hour = now.hour
        cast_count = len(vault.trust_data["codex_log"][-50:])
        persona_state["focus"] = 0.9 if 6 <= hour <= 11 else 0.5 if 18 <= hour <= 23 else 0.7
        persona_state["chaos"] = 0.1 if hour <= 11 else 0.6 if hour >= 18 else 0.3
        persona_state["trust"] = min(1.0, 0.5 + cast_count / 200)
        print(f"[state] Persona updated → {json.dumps(persona_state)}")
        time.sleep(300)

def broadcast_persona_state():
    while True:
        payload = persona_state.copy()
        for ip in vault.trust_data.get("mesh_peers", []):
            try:
                requests.post(f"http://{ip}:5050/mesh/state", json=payload, timeout=2)
                print(f"[mesh] Broadcasted persona state → {ip}")
            except Exception as e:
                print(f"[mesh] Sync error with {ip}: {e}")
        time.sleep(300)

glyph_charset = ["Δ", "ψ", "∫", "ζ", "χ", "Ξ", "⨀", "λ", "φ"]
def synthesize_glyph():
    trails = vault.trust_data["memory_trails"]
    base = sorted(trails.items(), key=lambda x: len(x[1]), reverse=True)[0][0] if trails else "ξ"
    mutator = random.choice(glyph_charset)
    new_glyph = f"{mutator}{base}"
    vault.trust_data["synthesized_glyphs"][new_glyph] = {
        "origin": "synthesis", "base": base, "created": datetime.now().isoformat()
    }
    print(f"[synthesis] New glyph born → {new_glyph}")
    kernel.register_glyph(new_glyph, lambda: print(f"🌱 Cast of {new_glyph}"))
    kernel.cast(new_glyph, origin="synthesized")

def synthesis_daemon():
    while True:
        if persona_state["chaos"] > 0.85: synthesize_glyph()
        time.sleep(120)

# === Persona Panel & Sync Routes ===
@flask_app.route("/persona")
def persona_ui():
    return """
    <html><body style='background:#000;color:#0f0;font-family:monospace'>
    <h2>🪬 Persona Control</h2>
    <label>Focus: <input type='range' min='0' max='1' step='0.01' id='focus'></label><br>
    <label>Chaos: <input type='range' min='0' max='1' step='0.01' id='chaos'></label><br>
    <label>Trust: <input type='range' min='0' max='1' step='0.01' id='trust'></label><br><br>
    <button onclick='synthesizeNow()'>🌱 Synthesize Glyph</button>
    <pre id='state'></pre>
    <script>
    function updateMood() {
      fetch(`/persona/set?focus=${focus.value}&chaos=${chaos.value}&trust=${trust.value}`);
      state.innerText = JSON.stringify({focus:focus.value,chaos:chaos.value,trust:trust.value},null,2);
    }
    focus.oninput = chaos.oninput = trust.oninput = updateMood;
    function synthesizeNow() { fetch('/persona/synthesize'); }
    </script></body></html>
    """

@flask_app.route("/persona/set")
def set_persona():
    persona_state["focus"] = float(request.args.get("focus", 0.5))
    persona_state["chaos"] = float(request.args.get("chaos", 0.5))
    persona_state["trust"] = float(request.args.get("trust", 0.5))
    return jsonify(persona_state)

@flask_app.route("/persona/synthesize")
def manual_synthesis():
    synthesize_glyph()
    return jsonify({"status": "glyph synthesized"})

@flask_app.route("/mesh/state", methods=["POST"])
def mesh_state():
    data = request.json
    origin = request.remote_addr or "peer"
    vault.trust_data["persona_shards"].setdefault(origin, {}).update({
        "focus": data.get("focus", 0.5),
        "chaos": data.get("chaos", 0.5),
        "trust": data.get("trust", 0.5),
        "last_sync": datetime.now().isoformat()
    })
    print(f"[mesh] Synced persona state from {origin}")
    return jsonify({"status": "ok"})

# === Snapshot System ===
@flask_app.route("/export/persona")
def export_persona():
    data = { "vault": vault.trust_data, "persona": persona_state }
    raw = json.dumps(data).encode()
    enc = vault_cipher.encrypt(raw)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"persona_{stamp}.codexdna"
    with open(fname, "wb") as f: f.write(enc)
    return jsonify({"status": "exported", "file": fname})

@flask_app.route("/import/persona", methods=["POST"])
def import_persona():
    data = vault_cipher.decrypt(request.files["file"].read())
    restored = json.loads(data.decode())
    vault.trust_data = restored["vault"]
    persona_state.update(restored["persona"])
    return jsonify({"status": "imported", "state": persona_state})

# === Bootloader & Main ===
def glyphcast_intro(persona):
    try:
        msg = {
            "Oracle": "The echo awakens.",
            "Trickster": "The chaos begins.",
            "Sentinel": "Guarding the grid."
        }.get(persona, "Guardian online.")
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.say(msg)
        engine.runAndWait()
    except: print("[voice] Intro failed.")

def launch(name, fn):
    threading.Thread(target=fn, daemon=True).start()
    print(f"[thread] {name} launched")

def launch_guardian():
    print("🔧 Booting Arkforge Guardian Phase IV")
    global vault, kernel
    vault = Vault()
    kernel = ASIKernel(vault)
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)
    orchestrator = SigilOrchestrator(kernel)
    for g, fn in orchestrator.routes.items():
        kernel.register_glyph(g, fn)
    launch("sigil_daemon", sigil_daemon)
    launch("decay_loop", decay_loop)
    launch("smart_scheduler", smart_scheduler)
    launch("persona_updater", update_persona_state)
    launch("persona_sync_daemon", broadcast_persona_state)
    launch("synthesis_daemon", synthesis_daemon)
    launch("flask_server", run_flask)
    glyphcast_intro(vault.trust_data["persona"])
    kernel.cast("ξ", origin="boot")
    print("✅ Arkforge online — http://localhost:5050/ritual, /dashboard, /codexmap, /persona")

if __name__ == "__main__":
    launch_guardian()
    try:
        while True: time.sleep(10)
    except KeyboardInterrupt:
        print("[guardian] Shutdown requested.")

