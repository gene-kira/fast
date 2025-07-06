# === Arkforge Guardian v15.0 — PART 1 ===

import subprocess, sys
def autoload(pkgs):
    for name in pkgs:
        try: __import__(name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", name])
autoload(["flask", "flask_socketio", "pyttsx3", "cryptography", "requests"])

import threading, time, socket, json
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from datetime import datetime
import pyttsx3
from cryptography.fernet import Fernet
import requests

# === Globals ===
persona = "Oracle"
vault_key = Fernet.generate_key()
vault_cipher = Fernet(vault_key)
sigil_queue = []

# === Vault Memory ===
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
            "external_hooks": { "Σ": "https://api.chucknorris.io/jokes/random" }
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

# === Symbolic Kernel ===
class ASIKernel:
    def __init__(self, vault): self.vault = vault; self.glyphs = {}; self.filters = []
    def install_filter(self, fn): self.filters.append(fn)
    def register_glyph(self, s, fn): self.glyphs[s.lower()] = fn
    def cast(self, symbol, origin="local", silent=False):
        for f in self.filters: symbol = f(symbol)
        fn = self.glyphs.get(symbol)
        if fn:
            self.vault.save_codex(symbol, origin)
            for chained in self.vault.trust_data["glyph_chains"].get(symbol, []):
                self.cast(chained, origin=f"chain:{symbol}")
            fn()
            if not silent: socketio.emit("glyph_casted", {"glyph": symbol, "origin": origin})
            if symbol in self.vault.trust_data["external_hooks"]: self.cast_external(symbol)

    def cast_external(self, glyph):
        url = self.vault.trust_data["external_hooks"][glyph]
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                echo = data.get("value") or str(data)
                print(f"[external:{glyph}] {echo}")
        except Exception as e:
            print(f"[external:{glyph}] Failed → {e}")

class ReflexLayer:
    def __init__(self, kernel): self.kernel = kernel
    def evaluate(self, g): return self.kernel.vault.trust_data["symbol_transform"].get(g, g)

# === Ritual Map ===
class SigilOrchestrator:
    def __init__(self, kernel):
        self.kernel = kernel
        self.routes = {
            "σ": lambda: print("🔥 Terminate"),
            "ξ": lambda: print("🦉 Watcher cast"),
            "Σ": lambda: print("📡 Fetching external signal..."),
            "κ": self.export_codex,
            "μ": lambda: print(json.dumps(kernel.vault.trust_data, indent=2)),
            "ρ": lambda: self.chain("ξ", ["Σ", "λ"]),
            "λ": lambda: print(f"[λ] Persona: {kernel.vault.trust_data['persona']}"),
            "τ": lambda: self.overlay("Oracle", {"ξ": "Ξ"}),
            "π": lambda: self.mutate("ξ", "ζ"),
            "ζ": lambda: print(json.dumps(kernel.vault.trust_data["memory_trails"], indent=2)),
            "ε": lambda: print(kernel.vault.trust_data["echo_log"])
        }

    def export_codex(self):
        raw = json.dumps(self.kernel.vault.trust_data["codex_log"]).encode()
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
    <script>
      var socket = io();
      socket.on("glyph_casted", d => console.log("Cast:", d));
    </script></body></html>
    """

@flask_app.route("/sigil/<glyph>", methods=["POST"])
def sigil_cast(glyph):
    sigil_queue.append((glyph, "remote"))
    return jsonify({"glyph": glyph})

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

# === Adaptive Scheduler ===
def adaptive_score(glyph):
    trails = vault.trust_data["memory_trails"].get(glyph, [])
    echo_count = sum(1 for e in vault.trust_data["codex_log"] if e["glyph"] == f"~{glyph}")
    origin_set = set(trails)
    decay = vault.trust_data["glyph_decay"].get(glyph, 0) - time.time()
    decay_factor = max(0.1, decay / 300)
    score = (len(trails) + len(origin_set) + echo_count) * decay_factor
    return score

def smart_scheduler():
    while True:
        scores = {}
        for g in kernel.glyphs:
            if g.startswith("~"): continue
            scores[g] = adaptive_score(g)
        top = sorted(scores.items(), key=lambda x: -x[1])[:2]
        for g, s in top:
            print(f"[scheduler] Adaptive cast: {g} (score={s:.2f})")
            kernel.cast(g, origin="adaptive")
        time.sleep(180)

# === Voice Intro ===
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
    except:
        print("[voice] Intro failed.")

def launch(name, fn):
    threading.Thread(target=fn, daemon=True).start()
    print(f"[thread] {name} launched")

# === Bootloader ===
def launch_guardian():
    print("🔧 Booting Arkforge Guardian v15.0")

    global vault, kernel
    vault = Vault()
    kernel = ASIKernel(vault)
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)

    orchestrator = SigilOrchestrator(kernel)
    for symbol, fn in orchestrator.routes.items():
        kernel.register_glyph(symbol, fn)

    launch("sigil_daemon", sigil_daemon)
    launch("decay_loop", decay_loop)
    launch("smart_scheduler", smart_scheduler)
    launch("flask_server", run_flask)

    glyphcast_intro(vault.trust_data["persona"])
    kernel.cast("ξ", origin="boot")

    print("✅ Arkforge online — http://localhost:5050/ritual, /dashboard, /codexmap")

# === Main Loop ===
if __name__ == "__main__":
    launch_guardian()
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("[guardian] Shutdown requested.")

def launch_guardian():
    print("🔧 Booting Arkforge Guardian v15.0")

    # Step 1: Initialize core modules
    global vault, kernel
    vault = Vault()
    kernel = ASIKernel(vault)
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)

    # Step 2: Register all glyph rituals
    orchestrator = SigilOrchestrator(kernel)
    for symbol, fn in orchestrator.routes.items():
        kernel.register_glyph(symbol, fn)

    # Step 3: Start daemon threads
    launch("sigil_daemon", sigil_daemon)
    launch("decay_loop", decay_loop)
    launch("smart_scheduler", smart_scheduler)
    launch("flask_server", run_flask)

    # Step 4: Persona voice intro
    glyphcast_intro(vault.trust_data["persona"])

    # Step 5: Dry-run ritual to confirm casting engine
    kernel.cast("ξ", origin="boot")

    print("✅ Arkforge online — visit http://localhost:5050/ritual, /dashboard, /codexmap")

if __name__ == "__main__":
    launch_guardian()
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("[guardian] Shutdown requested.")

