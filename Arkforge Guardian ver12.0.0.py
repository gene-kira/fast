import subprocess, sys
def autoload(pkgs):
    for name in pkgs:
        try: __import__(name)
        except: subprocess.check_call([sys.executable, "-m", "pip", "install", name])
autoload(["flask", "pyttsx3", "flask_socketio", "cryptography"])

import threading, time, socket, os, json
from flask import Flask, jsonify, request
import pyttsx3
from flask_socketio import SocketIO
from datetime import datetime
import importlib.util
from cryptography.fernet import Fernet

persona, mesh_port, mesh_peers = "Oracle", 6060, ["127.0.0.1"]
sigil_queue = []
vault_key = Fernet.generate_key()
vault_cipher = Fernet(vault_key)

class Vault:
    def __init__(self):
        self.trust_data = {
            "persona": persona,
            "trusted_ips": set(mesh_peers),
            "codex_log": [],
            "glyph_decay": {},
            "persona_overlay": {},
            "signature_peers": {},
            "glyph_chains": {}
        }
    def save_codex(self, glyph, origin="local"):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.trust_data["codex_log"].append({"glyph": glyph, "ts": ts, "origin": origin})
        self.trust_data["glyph_decay"][glyph] = time.time() + 300
    def mutate_codex(self):
        for entry in self.trust_data["codex_log"]:
            overlay = self.trust_data["persona_overlay"]
            if entry["glyph"] in overlay:
                entry["glyph"] = overlay[entry["glyph"]]
    def cleanup_decay(self):
        now = time.time()
        expired = [g for g, t in self.trust_data["glyph_decay"].items() if now > t]
        for g in expired: del self.trust_data["glyph_decay"][g]

class ASIKernel:
    def __init__(self, vault):
        self.vault = vault
        self.persona = vault.trust_data["persona"]
        self.filters = []
        self.glyphs = {}
    def install_filter(self, fn): self.filters.append(fn)
    def register_glyph(self, s, fn): self.glyphs[s.lower()] = fn
    def cast(self, symbol, origin="local", silent=False):
        for f in self.filters: symbol = f(symbol)
        fn = self.glyphs.get(symbol)
        if fn:
            self.vault.save_codex(symbol, origin)
            if symbol in self.vault.trust_data["glyph_chains"]:
                for linked in self.vault.trust_data["glyph_chains"][symbol]:
                    self.cast(linked, origin=f"chain:{symbol}")
            fn()
            if origin != "relay": mesh_cast(symbol)
            socketio.emit("glyph_casted", {"glyph": symbol, "origin": origin})

class ReflexLayer:
    def __init__(self, kernel): self.kernel = kernel
    def evaluate(self, glyph): return glyph.lower()

class SigilOrchestrator:
    def __init__(self, kernel):
        self.routes = {
            "σ": lambda: print("🔥 terminate"),
            "ω": lambda: print("🛡 defense"),
            "ψ": lambda: print("📡 scan"),
            "ξ": lambda: print("🦉 watcher"),
            "φ": lambda: print("🧹 purge"),
            "γ": lambda: os.system("echo 'Γ external command'"),
            "λ": lambda: print(f"Persona: {kernel.persona} | Decay: {len(kernel.vault.trust_data['glyph_decay'])}"),
            "κ": lambda: self.export_codex_encrypted(kernel.vault),
            "μ": lambda: print(json.dumps(kernel.vault.trust_data, indent=2)),
            "ν": lambda: print(list(kernel.glyphs.keys())),
            "χ": lambda: print(f"[persona] {kernel.persona}"),
            "ρ": lambda: self.schedule_chain("ξ", ["ω", "ψ"]),
            "τ": lambda: self.overlay_persona("Ω", {"ψ": "Ψ"}),
            "δ": lambda: kernel.vault.mutate_codex()
        }
    def export_codex_encrypted(self, vault):
        data = json.dumps(vault.trust_data["codex_log"]).encode()
        enc = vault_cipher.encrypt(data)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"codex_{stamp}.codex", "wb") as f: f.write(enc)
        print(f"[codex] Saved: codex_{stamp}.codex")
    def schedule_chain(self, trigger, sequence):
        kernel.vault.trust_data["glyph_chains"][trigger] = sequence
        print(f"[chain] {trigger} → {sequence}")
    def overlay_persona(self, name, map):
        kernel.vault.trust_data["persona_overlay"] = map
        print(f"[overlay] Persona '{name}' mapping: {map}")

flask_app = Flask(__name__)
socketio = SocketIO(flask_app, cors_allowed_origins="*")

@flask_app.route("/ritual")
def ritual_ui():
    return """
    <html><body style='background:#111;color:#eee;font-family:monospace;text-align:center'>
    <h2>🜂 Arkforge Ritual UI</h2><div style='font-size:2em'>
    <button onclick="cast('Σ')">Σ</button><button onclick="cast('Ω')">Ω</button>
    <button onclick="cast('Ψ')">Ψ</button><button onclick="cast('Ξ')">Ξ</button>
    <button onclick="cast('Φ')">Φ</button><button onclick="cast('Γ')">Γ</button>
    <button onclick="cast('Λ')">Λ</button><button onclick="cast('Κ')">Κ</button>
    <button onclick="cast('Μ')">Μ</button><button onclick="cast('Ν')">Ν</button>
    <button onclick="cast('Χ')">Χ</button><button onclick="cast('Ρ')">Ρ</button>
    <button onclick="cast('Τ')">Τ</button><button onclick="cast('Δ')">Δ</button>
    </div><hr><a href='/editor'>🎴 Ritual Editor</a> • <a href='/dashboard'>📊 Dashboard</a>
    <script src="https://cdn.socket.io/4.3.2/socket.io.min.js"></script>
    <script>var socket = io(); function cast(g){ fetch("/sigil/" + g,{method:"POST"}); }</script>
    </body></html>
    """

@flask_app.route("/dashboard")
def dashboard():
    p = vault.trust_data["persona"]
    decay = vault.trust_data["glyph_decay"]
    codex = vault.trust_data["codex_log"][-15:]
    html = f"<html><body style='background:#000;color:#0f0;font-family:monospace'><h2>📊 Dashboard</h2><p>Persona: <b>{p}</b></p><hr><ul>"
    for g, t in decay.items(): html += f"<li>{g} — {int(t - time.time())}s</li>"
    html += "</ul><hr><ul>"
    for c in reversed(codex): html += f"<li>{c['ts']} — {c['glyph']} from {c['origin']}</li>"
    html += "</ul></body></html>"
    return html

@flask_app.route("/editor")
def ritual_editor():
    return """
    <html><body style='background:#111;color:#fff;font-family:monospace;text-align:center'>
    <h2>🎴 Ritual Editor</h2><form method="POST" action="/define">
    Glyph: <input name="symbol" maxlength="1"><br><br>
    Python:<br><textarea name="code" rows="10" cols="60">def execute():\n  print("🔥 Glyph")</textarea><br>
    <button type="submit">Define</button></form><br><a href="/ritual">⟵ Ritual UI</a></body></html>
    """

@flask_app.route("/define", methods=["POST"])
def define_glyph():
    symbol = request.form.get("symbol", "").strip()
    code = request.form.get("code", "").strip()
    try:
        exec(code, globals())
        fn = globals().get("execute")
        if fn: kernel.register_glyph(symbol, fn)
        return f"<html><body style='color:lime;'>Glyph '{symbol}' added.</body></html>"
    except Exception as e:
        return f"<html><body style='color:red;'>Error: {e}</body></html>"

@flask_app.route("/sigil/<glyph>", methods=["POST"])
def sigil_cast(glyph):
    sigil_queue

# === Bootloader ===
if __name__ == "__main__":
    print("🔧 Booting Arkforge Guardian v11.0.0")

    vault = Vault()
    kernel = ASIKernel(vault)
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)

    orchestrator = SigilOrchestrator(kernel)
    for symbol, func in orchestrator.routes.items():
        kernel.register_glyph(symbol, func)

    # Optional: load plugins if you have a /glyphs folder
    def load_plugins(folder="glyphs"):
        if not os.path.isdir(folder): return
        for file in os.listdir(folder):
            if file.endswith(".py"):
                path = os.path.join(folder, file)
                spec = importlib.util.spec_from_file_location(file[:-3], path)
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    for sym, fn in mod.exports.items():
                        kernel.register_glyph(sym, fn)
                        print(f"[plugin] Loaded '{sym}' from {file}")
                except Exception as e:
                    print(f"[plugin] Error in {file}: {e}")

    load_plugins()

    def launch(name, target):
        try:
            threading.Thread(target=target, daemon=True).start()
            print(f"[thread] {name} launched")
        except Exception as e:
            print(f"[thread] {name} failed: {e}")

    def decay_loop(): vault.cleanup_decay(); time.sleep(60)
    def sigil_daemon():
        while True:
            if sigil_queue:
                symbol, origin = sigil_queue.pop(0)
                kernel.cast(symbol, origin, silent=(origin == "whisper"))
            time.sleep(1)
    def network_sim(): 
        while True:
            print(f"[mesh] Persona '{kernel.persona}' heartbeat ping")
            time.sleep(10)
    def ritual_scheduler():
        while True:
            kernel.cast("ξ", origin="scheduler")
            time.sleep(1800)
    def mesh_server():
        try:
            sock = socket.socket()
            sock.bind(("0.0.0.0", mesh_port))
            sock.listen(5)
            while True:
                conn, addr = sock.accept()
                data = conn.recv(32).decode()
                if data:
                    print(f"[mesh_server] Received glyph '{data}' from {addr}")
                    kernel.cast(data, origin="relay")
                conn.close()
        except Exception as e:
            print(f"[mesh_server] Error: {e}")
    def mesh_cast(symbol):
        for peer in mesh_peers:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((peer, mesh_port))
                    s.sendall(symbol.encode())
            except Exception as e:
                print(f"[mesh_cast] Fail: {e}")
    def run_flask():
        print("[flask] Ritual UI: http://localhost:5050/ritual")
        try:
            socketio.run(flask_app, host="0.0.0.0", port=5050)
        except Exception as e:
            print(f"[flask] Error: {e}")
    def glyphcast_intro(persona):
        phrase = {
            "Sentinel": "Sentinel online.",
            "Oracle": "The echo awakens.",
            "Trickster": "Heh. Let's see who trips the wire."
        }.get(persona, "Guardian ready.")
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(phrase)
            engine.runAndWait()
        except Exception as e:
            print(f"[voice] Failed: {e}")

    launch("sigil_daemon", sigil_daemon)
    launch("decay_loop", decay_loop)
    launch("network_sim", network_sim)
    time.sleep(1)
    launch("ritual_scheduler", ritual_scheduler)
    launch("mesh_server", mesh_server)
    launch("flask_server", run_flask)

    try:
        glyphcast_intro(kernel.persona)
    except Exception as e:
        print(f"[boot] Voice intro failed: {e}")

    print("[guardian] System fully operational.")
    try:
        while True: time.sleep(10)
    except KeyboardInterrupt:
        print("[guardian] Shutdown requested.")

