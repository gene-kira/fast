import threading, time
from flask import Flask, jsonify, request
import pyttsx3

# === Global Objects ===
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
            if not silent:
                print(f">> Casting glyph {symbol} from {origin}")
            self.vault.save_codex(symbol, origin)
            fn()

# === Reflex ===
class ReflexLayer:
    def __init__(self, kernel):
        self.kernel = kernel
    def evaluate(self, glyph):
        return glyph.lower()

# === Sigil Router ===
class SigilOrchestrator:
    def __init__(self, kernel):
        self.routes = {
            "σ": lambda: print("🔥 Activated Σ — terminate"),
            "ω": lambda: print("🛡 Activated Ω — defense"),
            "ψ": lambda: print("📡 Activated Ψ — scan"),
            "ξ": lambda: print("🦉 Activated Ξ — watcher"),
            "φ": lambda: print("🧹 Activated Φ — purge")
        }

# === Persona Voice Intro ===
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

# === Flask Ritual UI ===
flask_app = Flask(__name__)

@flask_app.route("/ritual")
def ritual_ui():
    return """
    <html><head><title>Arkforge Ritual</title></head>
    <body style='font-family:monospace;background:#111;color:#eee;text-align:center'>
    <h2>🜂 Arkforge Ritual UI</h2>
    <div style='font-size:2em'>
      <button onclick="cast('Σ')">Σ</button>
      <button onclick="cast('Ω')">Ω</button>
      <button onclick="cast('Ψ')">Ψ</button>
      <button onclick="cast('Ξ')">Ξ</button>
      <button onclick="cast('Φ')">Φ</button>
      <button onclick="cast_w('Σ')">Σ (W)</button>
    </div>
    <script>
      function cast(g) {{
        fetch('/sigil/' + g, {{method:'POST'}}).then(r => r.json()).then(x => alert("Cast: " + x.glyph));
      }}
      function cast_w(g) {{
        fetch('/sigil_w/' + g, {{method:'POST'}}).then(r => r.json()).then(x => alert("Whisper: " + x.glyph));
      }}
    </script>
    </body></html>
    """

@flask_app.route("/sigil/<glyph>", methods=["POST"])
def sigil_cast(glyph):
    sigil_queue.append((glyph, "remote"))
    return jsonify({"glyph": glyph, "status": "queued"})

@flask_app.route("/sigil_w/<glyph>", methods=["POST"])
def sigil_whisper(glyph):
    sigil_queue.append((glyph, "whisper"))
    return jsonify({"glyph": glyph, "mode": "silent"})

@flask_app.route("/persona/set/<mode>", methods=["POST"])
def set_persona(mode):
    if mode in voice_manifest:
        vault.trust_data["persona"] = mode
        kernel.persona = mode
        glyphcast_intro(mode)
        return jsonify({"status": "ok", "persona": mode})
    return jsonify({"error": "Invalid persona"})

# === Threads ===
def decay_loop():
    while True:
        vault.cleanup_decay()
        time.sleep(60)

def sigil_daemon():
    while True:
        if sigil_queue:
            symbol, origin = sigil_queue.pop(0)
            silent = origin == "whisper"
            kernel.cast(symbol, origin, silent)
        time.sleep(1)

def network_sim():
    while True:
        try:
            print(f"[mesh] Persona '{kernel.persona}' heartbeat ping")
        except Exception as e:
            print(f"[mesh] Error: {e}")
        time.sleep(10)

def run_flask():
    try:
        print("[flask] Ritual UI live at http://localhost:5050/ritual")
        flask_app.run("0.0.0.0", port=5050)
    except Exception as e:
        print(f"[flask] Flask error: {e}")

# === Bootloader ===
if __name__ == "__main__":
    print("🔧 Booting Arkforge Guardian v8.5.2")

    vault = Vault()
    kernel = ASIKernel(vault)
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)

    orchestrator = SigilOrchestrator(kernel)
    for sigil in orchestrator.routes:
        kernel.register_glyph(sigil, orchestrator.routes[sigil])

    threading.Thread(target=sigil_daemon, daemon=True).start()
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=decay_loop, daemon=True).start()
    threading.Thread(target=network_sim, daemon=True).start()

    glyphcast_intro(kernel.persona)
    print("[guardian] System fully operational.")

    # 🔒 Keepalive loop to prevent exit
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("[guardian] Shutdown requested. Exiting.")

