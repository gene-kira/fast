# === Enhanced Autoloader: Solunex-Grade ===
import subprocess, sys, importlib.util, pkg_resources

def install(pkg, version=None):
    name = f"{pkg}=={version}" if version else pkg
    try:
        print(f"[autoloader] Installing {name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", name])
        print(f"[autoloader] ✅ {name} installed successfully.")
    except Exception as e:
        print(f"[autoloader] ❌ Failed to install {name}: {e}")

def verify(pkg, version=None):
    try:
        pkg_resources.require(f"{pkg}=={version}" if version else pkg)
        print(f"[autoloader] ✅ Verified: {pkg}")
        return True
    except Exception:
        print(f"[autoloader] ⛔ Missing or mismatched: {pkg}")
        return False

def autoload(package_list):
    for p in package_list:
        if not verify(p["name"], p.get("version")):
            install(p["name"], p.get("version"))

# === Packages Needed for Solunex ===
autoload([
    {"name": "flask", "version": "2.3.3"},
    {"name": "flask_socketio", "version": "5.3.6"},
    {"name": "pyttsx3"},
    {"name": "cryptography", "version": "42.0.5"},
    {"name": "requests", "version": "2.31.0"},
    {"name": "torch", "version": "2.2.1"},
    {"name": "transformers", "version": "4.39.3"},
    {"name": "sounddevice", "version": "0.4.6"},
    {"name": "vosk", "version": "0.3.45"},
    {"name": "numpy", "version": "1.26.4"},
    {"name": "psutil"}
])



# === Autoloader & Setup ===
import subprocess, sys, importlib.util
def autoload(pkgs):
    for p in pkgs:
        if importlib.util.find_spec(p) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p])

autoload([
    "flask", "flask_socketio", "pyttsx3", "cryptography", "requests",
    "torch", "transformers", "sounddevice", "vosk", "numpy", "psutil"
])

# === Imports & Globals ===
import threading, time, json, os, platform
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from datetime import datetime
import numpy as np
import sounddevice as sd
from transformers import pipeline
from vosk import Model, KaldiRecognizer
from cryptography.fernet import Fernet
import psutil

flask_app = Flask(__name__)
socketio = SocketIO(flask_app, cors_allowed_origins="*")

persona_state = {"focus": 0.9, "chaos": 0.1, "trust": 0.95}
vault_key = Fernet.generate_key()
vault_cipher = Fernet(vault_key)
ai_memory = {"history": [], "drone_personalities": {}, "fix_log": []}
tts = pipeline("text-to-speech", model="openai/whisper-large")
stt_model = Model("model")
recognizer = KaldiRecognizer(stt_model, 16000)

# === Platform Signature ===
HOST_OS = platform.system()
print(f"[Solunex] Host platform detected: {HOST_OS}")

# === Vault System ===
class GlyphVault:
    def __init__(self):
        self.trust_data = {
            "persona": "Solunex",
            "codex_log": [],
            "glyph_decay": {},
            "memory_trails": {},
            "glyph_signatures": {},
            "system_version": {
                "ver": 1,
                "mode": "adaptive",
                "activated": datetime.now().isoformat()
            }
        }

    def save_codex(self, glyph, origin="local"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.trust_data["codex_log"].append({"glyph": glyph, "ts": ts, "origin": origin})
        self.trust_data["memory_trails"].setdefault(glyph, []).append(origin)
        self.trust_data["glyph_decay"][glyph] = time.time() + 300

    def cleanup_decay(self):
        now = time.time()
        expired = [g for g, t in self.trust_data["glyph_decay"].items() if now > t]
        for g in expired: del self.trust_data["glyph_decay"][g]

# === Persona Core ===
class PersonaCore:
    def __init__(self):
        self.name = "Solunex"
        self.traits = {"focus": 0.9, "trust": 0.95, "chaos": 0.1}

    def adjust(self, context):
        if "crash" in context: self.traits["trust"] *= 0.9
        if "freeze" in context: self.traits["focus"] *= 0.95
        if "restore" in context: self.traits["chaos"] *= 0.8

# === Glyph Reflex & Casting ===
class ReflexLayer:
    def __init__(self): self.filters = []

    def install_filter(self, fn): self.filters.append(fn)
    def apply(self, symbol):
        for f in self.filters: symbol = f(symbol)
        return symbol

class GlyphEngine:
    def __init__(self, vault, persona):
        self.vault = vault
        self.persona = persona
        self.glyphs = {}
        self.reflex = ReflexLayer()
        self.reflex.install_filter(lambda s: s.strip().lower())

    def register_glyph(self, symbol, fn):
        self.glyphs[symbol] = fn
        sig = vault_cipher.encrypt(symbol.encode()).decode()
        self.vault.trust_data["glyph_signatures"][symbol] = sig

    def cast(self, symbol, origin="system"):
        symbol = self.reflex.apply(symbol)
        fn = self.glyphs.get(symbol)
        if fn:
            self.vault.save_codex(symbol, origin)
            self.persona.adjust(origin)
            fn()
            socketio.emit("glyph_casted", {
                "glyph": symbol,
                "origin": origin,
                "traits": self.persona.traits,
                "persona": self.persona.name
            })

# === Voice Rituals ===
def borg_speak(text):
    audio = tts(text)
    sd.play(audio["waveform"], samplerate=22050)

def borg_listen():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1,
        callback=lambda indata, frames, time, status: recognizer.AcceptWaveform(indata)):
        while True:
            if recognizer.Result():
                result = json.loads(recognizer.Result

# === Vault System ===
class GlyphVault:
    def __init__(self):
        self.trust_data = {
            "persona": "Solunex",
            "codex_log": [],
            "glyph_decay": {},
            "memory_trails": {},
            "glyph_signatures": {},
            "system_version": {
                "ver": 1,
                "mode": "adaptive",
                "activated": datetime.now().isoformat()
            }
        }

    def save_codex(self, glyph, origin="local"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.trust_data["codex_log"].append({"glyph": glyph, "ts": ts, "origin": origin})
        self.trust_data["memory_trails"].setdefault(glyph, []).append(origin)
        self.trust_data["glyph_decay"][glyph] = time.time() + 300

    def cleanup_decay(self):
        now = time.time()
        expired = [g for g, t in self.trust_data["glyph_decay"].items() if now > t]
        for g in expired:
            del self.trust_data["glyph_decay"][g]

# === Persona Core ===
class PersonaCore:
    def __init__(self):
        self.name = "Solunex"
        self.traits = {"focus": 0.9, "trust": 0.95, "chaos": 0.1}

    def adjust(self, context):
        if "crash" in context:
            self.traits["trust"] *= 0.9
        if "freeze" in context:
            self.traits["focus"] *= 0.95
        if "restore" in context:
            self.traits["chaos"] *= 0.8

# === Glyph Reflex Engine ===
class ReflexLayer:
    def __init__(self): self.filters = []
    def install_filter(self, fn): self.filters.append(fn)
    def apply(self, symbol):
        for f in self.filters:
            symbol = f(symbol)
        return symbol

# === Glyph Casting Engine ===
class GlyphEngine:
    def __init__(self, vault, persona):
        self.vault = vault
        self.persona = persona
        self.glyphs = {}
        self.reflex = ReflexLayer()
        self.reflex.install_filter(lambda s: s.strip().lower())

    def register_glyph(self, symbol, fn):
        self.glyphs[symbol] = fn
        sig = vault_cipher.encrypt(symbol.encode()).decode()
        self.vault.trust_data["glyph_signatures"][symbol] = sig

    def cast(self, symbol, origin="system"):
        symbol = self.reflex.apply(symbol)
        fn = self.glyphs.get(symbol)
        if fn:
            self.vault.save_codex(symbol, origin)
            self.persona.adjust(origin)
            fn()
            socketio.emit("glyph_casted", {
                "glyph": symbol,
                "origin": origin,
                "traits": self.persona.traits,
                "persona": self.persona.name
            })

# === Voice Rituals ===
def borg_speak(text):
    audio = tts(text)
    sd.play(audio["waveform"], samplerate=22050)

def borg_listen():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1,
        callback=lambda indata, frames, time, status: recognizer.AcceptWaveform(indata)):
        while True:
            if recognizer.Result():
                result = json.loads(recognizer.Result())
                symbol = result.get("text", "").strip()
                engine.cast(symbol, origin="voice")

# === Flask Ritual Panels ===

@flask_app.route("/persona/control", methods=["GET", "POST"])
def persona_control():
    if request.method == "POST":
        p = request.json.get("persona", "Solunex")
        vault.trust_data["persona"] = p
        persona.name = p
        persona.traits["chaos"] = 0.3 if p == "Trickster" else 0.1
        persona.traits["focus"] = 0.95 if p == "Sentinel" else 0.9
        persona.traits["trust"] = 0.99 if p == "Sentinel" else 0.95
        ai_memory["drone_personalities"]["guardian"] = p
        return jsonify({"message": f"Persona updated to {p}"})
    return jsonify({"current": vault.trust_data["persona"]})

@flask_app.route("/persona/emotion")
def emotion_viewer():
    tone = persona.traits
    html = f"""
    <html><body style='background:#111;color:#0ff'>
    <h2>🎭 Emotional Tone — Persona: {persona.name}</h2>
    <p><b>Chaos:</b> {tone['chaos']}<br>
       <b>Focus:</b> {tone['focus']}<br>
       <b>Trust:</b> {tone['trust']}</p></body></html>"""
    return html

@flask_app.route("/glyph/frequency")
def glyph_frequency():
    codex = vault.trust_data["codex_log"][-500:]
    freq = {}
    for entry in codex:
        g = entry["glyph"]
        freq[g] = freq.get(g, 0) + 1
    html = "<html><body style='background:#000;color:#0f0'><h2>📊 Glyph Frequency Map</h2><ul>"
    for g, f in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]:
        html += f"<li>{g}: {f} casts</li>"
    html += "</ul></body></html>"
    return html

@flask_app.route("/glyph/classifier")
def glyph_classifier():
    codex = vault.trust_data["codex_log"][-200:]
    html = "<html><body style='background:#111;color:#fff'><h2>🔬 Glyph Classifier</h2><ul>"
    for entry in codex:
        g = entry["glyph"]
        tags = []
        if g.startswith("g"): tags.append("Numeric")
        if "~" in g: tags.append("Echo")
        if "∂" in g: tags.append("Mutation")
        if g.startswith(("ψ", "Ξ", "Δ")): tags.append("Symbolic")
        if entry["origin"] == "voice": tags.append("VoiceCast")
        html += f"<li>{g} → {', '.join(tags)} [{entry['origin']}]</li>"
    html += "</ul></body></html>"
    return html

# === Thread Launcher ===
def launch(name, fn):
    threading.Thread(target=fn, daemon=True).start()
    print(f"[thread] {name} launched")

# === Healing Registry ===
def register_healing_glyphs():
    healing_map = {
        "∂memory-leak": lambda: psutil.virtual_memory(),  # simulate
        "∂cpu-overload": lambda: psutil.cpu_percent(),
        "∂daemon-crash": lambda: print("[heal] Restarting daemon..."),
        "∂socket-fail": lambda: print("[heal] Rebinding sockets..."),
        "∂entropy-rise": lambda: print("[heal] Purging temp glyphs"),
        "Ξinitiate": lambda: print("🌀 Solunex initiation ritual complete"),
    }
    for glyph, fn in healing_map.items():
        engine.register_glyph(glyph, fn)

# === Entropy Monitor ===
def monitor_entropy():
    while True:
        ram = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=1)
        if ram > 85:
            engine.cast("∂memory-leak", origin="entropy")
        if cpu > 90:
            engine.cast("∂cpu-overload", origin="entropy")
        time.sleep(5)

# === Bootloader ===
def boot_solunex():
    print("🛡️ Solunex: Booting ASI Agent")
    global vault, persona, engine
    vault = GlyphVault()
    persona = PersonaCore()
    engine = GlyphEngine(vault, persona)
    register_healing_glyphs()

    launch("entropy_monitor", monitor_entropy)
    launch("voice_listener", borg_listen)
    launch("decay_cleanup", vault.cleanup_decay)

    engine.cast("Ξinitiate", origin="boot")
    print("✅ Solunex online. Visit http://localhost:5000/persona/emotion")

# === Runtime Loop ===
if __name__ == "__main__":
    boot_solunex()
    flask_app.run(port=5000)
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("[Solunex] Ritual shutdown requested.")

