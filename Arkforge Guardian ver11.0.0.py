# === Autoloader ===
import subprocess, sys
def autoload(packages):
    for name in packages:
        try:
            __import__(name)
        except ImportError:
            print(f"[autoload] Installing: {name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", name])
autoload(["flask", "pyttsx3", "flask_socketio", "cryptography"])

# === Imports ===
import threading, time, socket, os, json
from flask import Flask, jsonify, request
import pyttsx3
from flask_socketio import SocketIO
from datetime import datetime
import importlib.util
from cryptography.fernet import Fernet

# === Config Loader ===
def load_config(path="config.json"):
    try:
        with open(path) as f:
            cfg = json.load(f)
        persona = cfg.get("persona", "Sentinel")
        mesh_port = cfg.get("mesh_port", 6060)
        peers = cfg.get("mesh_peers", ["127.0.0.1"])
        print(f"[config] Loaded: persona={persona}, port={mesh_port}, peers={peers}")
        return persona, mesh_port, peers
    except Exception as e:
        print(f"[config] Failed to load: {e}")
        return "Sentinel", 6060, ["127.0.0.1"]

persona, mesh_port, mesh_peers = load_config()
sigil_queue = []
vault_key = Fernet.generate_key()
vault_cipher = Fernet(vault_key)

# === Vault ===
class Vault:
    def __init__(self):
        self.trust_data = {
            "persona": persona,
            "trusted_ips": set(mesh_peers),
            "codex_log": [],
            "glyph_decay": {}
        }

    def save_codex(self, glyph, origin="local"):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.trust_data["codex_log"].append({"glyph": glyph, "ts": ts, "origin": origin})
        self.trust_data["glyph_decay"][glyph] = time.time() + 300

    def cleanup_decay(self):
        now = time.time()
        expired = [g for g, exp in self.trust_data["glyph_decay"].items() if now > exp]
        for g in expired:
            del self.trust_data["glyph_decay"][g]

# === Kernel ===
class ASIKernel:
    def __init__(self, vault):
        self.vault = vault
        self.persona = vault.trust_data["persona"]
        self.filters = []
        self.glyphs = {}

    def install_filter(self, fn): self.filters.append(fn)
    def register_glyph(self, symbol, fn): self.glyphs[symbol.lower()] = fn

    def cast(self, symbol, origin="local", silent=False):
        for f in self.filters: symbol = f(symbol)
        fn = self.glyphs.get(symbol)
        if fn:
            print(f">> Casting glyph {symbol} from {origin}")
            self.vault.save_codex(symbol, origin)
            fn()
            if origin != "relay": mesh_cast(symbol)
            socketio.emit("glyph_casted", {"glyph": symbol, "origin": origin})

class ReflexLayer:
    def __init__(self, kernel): self.kernel = kernel
    def evaluate(self, glyph): return glyph.lower()

# === Orchestrator ===
class SigilOrchestrator:
    def __init__(self, kernel):
        self.routes = {
            "σ": lambda: print("🔥 Σ — terminate"),
            "ω": lambda: print("🛡 Ω — defense"),
            "ψ": lambda: print("📡 Ψ — scan"),
            "ξ": lambda: print("🦉 Ξ — watcher"),
            "φ": lambda: print("🧹 Φ — purge"),
            "γ": lambda: os.system("echo 'Γ triggered'"),
            "λ": lambda: print(f"[λ] Persona: {kernel.persona} | Decay: {len(kernel.vault.trust_data['glyph_decay'])}"),
            "κ": lambda: self.export_codex(kernel.vault)
        }

    def export_codex(self, vault):
        data = json.dumps(vault.trust_data["codex_log"]).encode()
        encrypted = vault_cipher.encrypt(data)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"codex_{stamp}.codex", "wb") as f: f.write(encrypted)
        print(f"[codex] Encrypted snapshot saved: codex_{stamp}.codex")

# === Flask Ritual UI ===
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
    </div><hr><a href='/editor'>🎴 Open Ritual Editor</a>
    <script src="https://cdn.socket.io/4.3.2/socket.io.min.js"></script>
    <script>
      var socket = io(); socket.on("glyph_casted", d => console.log("Cast:", d));
      function cast(g) { fetch("/sigil/" + g, {method:"POST"}); }
    </script></body></html>
    """

@flask_app.route("/editor")
def ritual_editor():
    return """
    <html><body style='background:#111;color:#fff;font-family:monospace;text-align:center'>
    <h2>🎴 Ritual Editor</h2><form method="POST" action="/define">
    Glyph Symbol: <input name="symbol" maxlength="1"><br><br>
    Python Function:<br><textarea name="code" rows="10" cols="60">def execute():\n  print("🔥 Custom glyph executed")</textarea><br>
    <button type="submit">Create Glyph</button>
    </form><br><a href="/ritual">⟵ Back to Ritual UI</a></body></html>
    """

@flask_app.route("/define", methods=["POST"])
def define_glyph():
    symbol = request.form.get("symbol", "").strip()
    code = request.form.get("code", "").strip()
    try:
        exec(code, globals())
        fn = globals().get("execute")
        if fn:
            kernel.register_glyph(symbol, fn)
            print(f"[editor] Glyph '{symbol}' registered via Ritual Editor")
            return f"<html><body style='color:lime;'>✅ Glyph '{symbol}' defined.</body></html>"
        return "<html><body style='color:red;'>Function 'execute' not found.</body></html>"
    except Exception as e:
        return f"<html><body style='color:red;'>Error: {e}</body></html>"

@flask_app.route("/sigil/<glyph>", methods=["POST"])
def sigil_cast(glyph): sigil_queue.append((glyph, "remote")); return jsonify({"glyph": glyph})

@flask_app.route("/persona/set/<mode>", methods=["POST"])
def set_persona(mode):
    if mode in voice_manifest:
        vault.trust_data["persona"] = mode
        kernel.persona = mode
        glyphcast_intro(mode)
        socketio.emit("persona_switched", {"persona": mode})
        return jsonify({"status": "ok"})
    return jsonify({"error": "Invalid persona"})

@flask_app.route("/status")
def status_page():
    p = vault.trust_data["persona"]
    dq = vault.trust_data["glyph_decay"]
    h = vault.trust_data["codex_log"][-10:]
    html = f"<html><body style='background:#000;color:#0f0;font-family:monospace'><h2>🜂 Status</h2><p>Persona: <b>{p}</b></p><hr><ul>"
    for g, exp in dq.items(): html += f"<li>{g} — {int(exp - time.time())}s</li>"
    html += "</ul><hr><ul>"
    for e in reversed(h): html += f"<li>{e['ts']} — {e['glyph']} from {e['origin']}</li>"
    html += "</ul></body></html>"
    return html

# === Plugin Loader ===
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
                    print(f"[plugin] Glyph '{sym}' loaded from {file}")
            except Exception as e:
                print(f"[plugin] Error in {file}: {e}")

# === Background Daemons ===
def decay_loop():
    while True:
        vault.cleanup_decay()
        time.sleep(60)

def ritual_scheduler():
    while True:
        kernel.cast("ξ", origin="scheduler")
        time.sleep(1800)

def network_sim():
    while True:
        print(f"[mesh] Persona '{kernel.persona}' heartbeat ping")
        time.sleep(10)

def sigil_daemon():
    while True:
        if sigil_queue:
            symbol, origin = sigil_queue.pop(0)
            kernel.cast(symbol, origin, silent=(origin == "whisper"))
        time.sleep(1)

def mesh_cast(symbol):
    for peer in mesh_peers:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((peer, mesh_port))
                s.sendall(symbol.encode())
        except Exception as e:
            print(f"[mesh_cast] Failed to {peer}: {e}")

def mesh_server():
    try:
        sock = socket.socket()
        sock.bind(("0.0.0.0", mesh_port))
        sock.listen(5)
        print(f"[mesh_server] Listening on {mesh_port}")
        while True:
            conn, addr = sock.accept()
            data = conn.recv(32).decode()
            if data:
                print(f"[mesh_server] Received glyph '{data}' from {addr}")
                kernel.cast(data, origin="relay")
            conn.close()
    except Exception as e:
        print(f"[mesh_server] Error: {e}")

def run_flask():
    print("[flask] Ritual UI: http://localhost:5050/ritual")
    try:
        socketio.run(flask_app, host="0.0.0.0", port=5050)
    except Exception as e:
        print(f"[flask] Error: {e}")

# === Persona Voice Intro ===
voice_manifest = {
    "Sentinel": "Sentinel online. Systems operational.",
    "Oracle": "The echo awakens. Oracle listens.",
    "Trickster": "Heh. Let's see who trips the wire."
}
def glyphcast_intro(persona):
    phrase = voice_manifest.get(persona, "Guardian ready.")
    try:
        engine = pyttsx3.init(); engine.setProperty("rate", 165)
        engine.say(phrase); engine.runAndWait()
    except Exception as e:
        print(f"[voice] Intro failed: {e}")

# === Bootloader ===
if __name__ == "__main__":
    print("🔧 Booting Arkforge Guardian v10.1.0")

    vault = Vault()
    kernel = ASIKernel(vault)
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)

    orchestrator = SigilOrchestrator(kernel)
    for sigil in orchestrator.routes:
        kernel.register_glyph(sigil, orchestrator.routes[sigil])

    load_plugins()

    def launch(name, target):
        try:
            threading.Thread(target=target, daemon=True).start()
            print(f"[thread] {name} launched")
        except Exception as e:
            print(f"[thread] {name} failed: {e}")

    launch("sigil_daemon", sigil_daemon)
    launch("decay_loop", decay_loop)
    launch("network_sim", network_sim)
    time.sleep(2)
    launch("ritual_scheduler", ritual_scheduler)
    launch("mesh_server", mesh_server)
    launch("flask_server", run_flask)

    try:
        glyphcast_intro(kernel.persona)
    except Exception as e:
        print(f"[boot] Voice intro error: {e}")

    print("[guardian] System fully operational.")
    try:
        while True: time.sleep(10)
    except KeyboardInterrupt:
        print("[guardian] Shutdown requested. Exiting.")

