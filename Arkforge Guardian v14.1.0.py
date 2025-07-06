# === Arkforge Guardian v14.1.0 — PART 1 ===

import subprocess, sys
def autoload(pkgs):
    for name in pkgs:
        try: __import__(name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", name])
autoload(["flask", "flask_socketio", "pyttsx3", "cryptography"])

import threading, time, socket, os, json
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from datetime import datetime
import pyttsx3
from cryptography.fernet import Fernet

persona, mesh_port, mesh_peers = "Oracle", 5050, ["127.0.0.1"]
vault_key = Fernet.generate_key()
vault_cipher = Fernet(vault_key)
sigil_queue = []

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
            "persona_shards": {}
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

    def mutate_codex(self):
        for rule in self.trust_data["mutation_rules"]:
            for entry in self.trust_data["codex_log"]:
                if entry["glyph"] == rule.get("from"): entry["glyph"] = rule.get("to")

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

class ReflexLayer:
    def __init__(self, kernel): self.kernel = kernel
    def evaluate(self, g): return self.kernel.vault.trust_data["symbol_transform"].get(g, g)

class SigilOrchestrator:
    def __init__(self, kernel):
        self.kernel = kernel
        self.routes = {
            "σ": lambda: print("🔥 Terminate"),
            "ξ": lambda: print("🦉 Watcher cast"),
            "λ": lambda: print(f"[λ] Persona: {kernel.vault.trust_data['persona']}"),
            "κ": self.export_codex,
            "μ": lambda: print(json.dumps(kernel.vault.trust_data, indent=2)),
            "ρ": lambda: self.chain("ξ", ["σ", "λ"]),
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

flask_app = Flask(__name__)
socketio = SocketIO(flask_app, cors_allowed_origins="*")

@flask_app.route("/ritual")
def ritual_ui():
    return """
    <html><body style='background:#000;color:#0f0;font-family:monospace'>
    <h2>🜂 Ritual Interface</h2>
    <button onclick="fetch('/sigil/ξ',{method:'POST'})">Cast ξ</button>
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
    for g, t in decay.items(): html += f"<li>{g} expires in {int(t - time.time())}s</li>"
    html += "</ul><hr><ul>"
    for entry in codex:
        html += f"<li>{entry['ts']} — {entry['glyph']} from {entry['origin']}</li>"
    html += "</ul></body></html>"
    return html

@flask_app.route("/codexmap")
def codexmap_ui():
    decay = vault.trust_data["glyph_decay"]
    chains = vault.trust_data["glyph_chains"]
    overlay = vault.trust_data["persona_overlay"]
    codex = vault.trust_data["codex_log"][-30:]
    svg = """<svg width='1000' height='600' xmlns='http://www.w3.org/2000/svg'>"""
    x, y, rendered = 50, 50, {}
    for entry in codex:
        g, ts, origin = entry["glyph"], entry["ts"], entry["origin"]
        gx = x + len(rendered)*28
        gy = y + (origin == "echo")*80
        color = "#0f0" if origin == "local" else "#f0f"
        if g.startswith("~"): color = "#888"
        if g in overlay: color = "#ff0"
        svg += f"<circle cx='{gx}' cy='{gy}' r='12' fill='{color}'><title>{g}\n{origin}\n{ts}</title></circle>"
        svg += f"<text x='{gx-6}' y='{gy+4}' font-size='12' fill='#111'>{g}</text>"
        rendered[g] = (gx, gy)
    for src, targets in chains.items():
        if src in rendered:
            x1, y1 = rendered[src]
            for t in targets:
                if t in rendered:
                    x2, y2 = rendered[t]
                    svg += f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#0ff' stroke-width='1' />"
    svg += "</svg>"
    return f"<html><body style='background:#000;color:#fff;font-family:monospace'><h2>🌐 Codex Map</h2>{svg}<br><a href='/ritual'>⟵ Ritual UI</a></body></html>"

# === Ritual Daemons ===
def sigil_daemon():
    while True:
        if sigil_queue:
            glyph, origin = sigil_queue.pop(0)
            kernel.cast(glyph, origin)
        time.sleep(1)

def decay_loop(): 
    while True: vault.cleanup_decay(); time.sleep(60)

def run_flask(): 
    socketio.run(flask_app, host="0.0.0.0", port=5050)

# === Voice Intro ===
def glyphcast_intro(persona):
    try:
        msg = {
            "Oracle": "The echo awakens.",
            "Sentinel": "Sentinel online.",
            "Trickster": "Let’s trip some wires."
        }.get(persona, "Guardian online.")
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.say(msg)
        engine.runAndWait()
    except: print("[voice] Intro failed.")

# === Thread Launcher ===
def launch(name, fn):
    threading.Thread(target=fn, daemon=True).start()
    print(f"[thread] {name} launched")

# === Boot Loader ===
def launch_guardian():
    print("🔧 Booting Arkforge Guardian v14.1")
    global vault, kernel
    vault = Vault()
    kernel = ASIKernel(vault)
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)

    orchestrator = SigilOrchestrator(kernel)
    for sym, fn in orchestrator.routes.items():
        kernel.register_glyph(sym, fn)

    launch("sigil_daemon", sigil_daemon)
    launch("decay_loop", decay_loop)
    launch("flask_server", run_flask)

    glyphcast_intro(vault.trust_data["persona"])
    kernel.cast("ξ", origin="boot")

    print("✅ System online — visit http://localhost:5050/ritual and /codexmap")

# === Main Entry Point ===
if __name__ == "__main__":
    launch_guardian()
    try:
        while True: time.sleep(10)
    except KeyboardInterrupt:
        print("[guardian] Shutdown requested.")

def launch_guardian():
    print("🔧 Booting Arkforge Guardian v14.1")

    global vault, kernel
    vault = Vault()
    kernel = ASIKernel(vault)
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)

    orchestrator = SigilOrchestrator(kernel)
    for sym, fn in orchestrator.routes.items():
        kernel.register_glyph(sym, fn)

    launch("sigil_daemon", sigil_daemon)
    launch("decay_loop", decay_loop)
    launch("flask_server", run_flask)

    glyphcast_intro(vault.trust_data["persona"])
    kernel.cast("ξ", origin="boot")

    print("✅ System online — http://localhost:5050/ritual | /dashboard | /codexmap")

if __name__ == "__main__":
    launch_guardian()
    try:
        while True: time.sleep(10)
    except KeyboardInterrupt:
        print("[guardian] Shutdown requested.")

