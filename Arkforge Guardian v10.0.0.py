# === Autoloader ===
import subprocess, sys

def autoload(packages):
    for name in packages:
        try:
            __import__(name)
        except ImportError:
            print(f"[autoload] Installing: {name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", name])

autoload(["flask", "pyttsx3", "flask_socketio"])

# === Imports ===
import threading, time, socket, os
from flask import Flask, jsonify, request
import pyttsx3
from flask_socketio import SocketIO

sigil_queue = []

# === Vault ===
class Vault:
    def __init__(self):
        self.trust_data = {
            "persona": "Sentinel",
            "trusted_ips": {"127.0.0.1"},
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

    def install_filter(self, fn):
        self.filters.append(fn)

    def register_glyph(self, symbol, fn):
        self.glyphs[symbol.lower()] = fn

    def cast(self, symbol, origin="local", silent=False):
        for f in self.filters:
            symbol = f(symbol)
        fn = self.glyphs.get(symbol)
        if fn:
            print(f">> Casting glyph {symbol} from {origin}")
            self.vault.save_codex(symbol, origin)
            fn()
            if origin != "relay":
                mesh_cast(symbol)
            socketio.emit("glyph_casted", {"glyph": symbol, "origin": origin})

# === Reflex ===
class ReflexLayer:
    def __init__(self, kernel):
        self.kernel = kernel
    def evaluate(self, glyph):
        return glyph.lower()

# === Sigils ===
class SigilOrchestrator:
    def __init__(self, kernel):
        self.routes = {
            "σ": lambda: print("🔥 Activated Σ — terminate"),
            "ω": lambda: print("🛡 Activated Ω — defense"),
            "ψ": lambda: print("📡 Activated Ψ — scan"),
            "ξ": lambda: print("🦉 Activated Ξ — watcher"),
            "φ": lambda: print("🧹 Activated Φ — purge"),
            "γ": lambda: os.system("echo 'Guardian triggered external command Γ'"),
            "λ": lambda: print(f"[λ] Persona: {kernel.persona} | Decay: {len(kernel.vault.trust_data['glyph_decay'])}")
        }

# === Voice Boot ===
voice_manifest = {
    "Sentinel": "Sentinel online. Systems operational.",
    "Oracle": "The echo awakens. Oracle listens.",
    "Trickster": "Heh. Let's see who trips the wire."
}

def glyphcast_intro(persona):
    phrase = voice_manifest.get(persona, "Guardian ready.")
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.say(phrase)
        engine.runAndWait()
    except Exception as e:
        print(f"[voice] Boot intro failed: {e}")

# === Flask + WebSocket ===
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
    <button onclick="cast('Λ')">Λ</button><button onclick="cast_w('Σ')">Σ (W)</button>
    </div><script src="https://cdn.socket.io/4.3.2/socket.io.min.js"></script>
    <script>
      var socket = io(); socket.on("glyph_casted", d => console.log("Cast:", d));
      function cast(g) {{ fetch("/sigil/" + g, {{method:"POST"}}); }}
      function cast_w(g) {{ fetch("/sigil_w/" + g, {{method:"POST"}}); }}
    </script></body></html>
    """

@flask_app.route("/sigil/<glyph>", methods=["POST"])
def sigil_cast(glyph):
    sigil_queue.append((glyph, "remote"))
    return jsonify({"glyph": glyph})

@flask_app.route("/sigil_w/<glyph>", methods=["POST"])
def sigil_whisper(glyph):
    sigil_queue.append((glyph, "whisper"))
    return jsonify({"glyph": glyph})

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
    html = f"<html><body style='background:#000;color:#0f0;font-family:monospace'><h2>🜂 Status</h2><p>🧠 Persona: <b>{p}</b></p><hr><ul>"
    for g, exp in dq.items():
        html += f"<li>{g} — {int(exp - time.time())}s</li>"
    html += "</ul><hr><ul>"
    for e in reversed(h):
        html += f"<li>{e['ts']} — {e['glyph']} from {e['origin']}</li>"
    html += "</ul></body></html>"
    return html

# === Background Threads ===
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
    peers = ["127.0.0.1"]
    for peer in peers:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((peer, 6060))
                s.sendall(symbol.encode())
        except Exception as e:
            print(f"[mesh_cast] Fail: {e}")

def mesh_server():
    try:
        sock = socket.socket()
        sock.bind(("0.0.0.0", 6060))
        sock.listen(5)
        print("[mesh_server] Listening on port 6060")
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
    print("[flask] Ritual UI ready at http://localhost:5050/ritual")
    try:
        socketio.run(flask_app, host="0.0.0.0", port=5050)
    except Exception as e:
        print(f"[flask] Error: {e}")

# === Bootloader ===
if __name__ == "__main__":
    print("🔧 Booting Arkforge Guardian v9.0.2")

    vault = Vault()
    kernel = ASIKernel

# === Bootloader ===
if __name__ == "__main__":
    print("🔧 Booting Arkforge Guardian v9.0.2")

    vault = Vault()
    kernel = ASIKernel(vault)
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)
    orchestrator = SigilOrchestrator(kernel)
    for sigil in orchestrator.routes:
        kernel.register_glyph(sigil, orchestrator.routes[sigil])

    def safe_thread(name, target):
        try:
            threading.Thread(target=target, daemon=True).start()
            print(f"[thread] {name} launched")
        except Exception as e:
            print(f"[thread] {name} failed: {e}")

    safe_thread("sigil_daemon", sigil_daemon)
    safe_thread("decay_loop", decay_loop)
    safe_thread("network_sim", network_sim)
    time.sleep(2)
    safe_thread("ritual_scheduler", ritual_scheduler)
    safe_thread("mesh_server", mesh_server)
    safe_thread("flask_server", run_flask)

    try:
        glyphcast_intro(kernel.persona)
    except Exception as e:
        print(f"[voice] Persona boot failed: {e}")

    print("[guardian] System fully operational.")

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("[guardian] Shutdown requested. Exiting.")

