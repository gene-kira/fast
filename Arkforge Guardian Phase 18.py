# === Auto-install dependencies ===
import subprocess, sys
def autoload(pkgs):
    for name in pkgs:
        try: __import__(name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", name])
autoload(["flask", "flask_socketio", "pyttsx3", "cryptography", "requests"])

# === Imports ===
import threading, time, socket, json, random
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from datetime import datetime
import pyttsx3
from cryptography.fernet import Fernet
import requests

# === Globals ===
persona_state = {"focus": 0.7, "chaos": 0.3, "trust": 0.9}
vault_key = Fernet.generate_key()
vault_cipher = Fernet(vault_key)
sigil_queue = []
flask_app = Flask(__name__)
socketio = SocketIO(flask_app, cors_allowed_origins="*")

# === Vault ===
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
            "mythfusion": "blend",
            "epoch_harmonics": {},
            "ritual_motifs": {},
            "recursed_scores": [],
            "emotionally_tuned": [],
            "mutated_motifs": []
        }

    def save_codex(self, glyph, origin="local"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"glyph": glyph, "ts": ts, "origin": origin}
        self.trust_data["codex_log"].append(entry)
        if glyph not in self.trust_data["glyph_provenance"]:
            self.trust_data["glyph_provenance"][glyph] = {
                "first_cast": ts, "origin": origin, "persona": self.trust_data["persona"]
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
                if len(trail) < 3:
                    echo = None
            if echo:
                self.trust_data["echo_log"].append({"glyph": echo, "ts": ts})
                self.trust_data["codex_log"].append({"glyph": echo, "ts": ts, "origin": "echo"})

    def cleanup_decay(self):
        now = time.time()
        expired = [g for g, t in self.trust_data["glyph_decay"].items() if now > t]
        for g in expired:
            del self.trust_data["glyph_decay"][g]

# === Kernel ===
class ASIKernel:
    def __init__(self, vault):
        self.vault = vault
        self.glyphs = {}
        self.filters = []

    def install_filter(self, fn):
        self.filters.append(fn)

    def register_glyph(self, s, fn):
        self.glyphs[s.lower()] = fn

    def cast(self, symbol, origin="local", silent=False):
        for f in self.filters:
            symbol = f(symbol)
        if origin.startswith("mesh:") and not self.verify_glyph_signature(symbol):
            print(f"[firewall] Rejected mesh glyph: {symbol}")
            return
        fn = self.glyphs.get(symbol)
        if not fn:
            return
        self.vault.save_codex(symbol, origin)
        if persona_state["chaos"] > 0.7 and random.random() < persona_state["chaos"]:
            mutated = f"{symbol}∂"
            self.vault.save_codex(mutated, "mutation")
            print(f"[chaos] Mutation → {mutated}")
        for chained in self.vault.trust_data["glyph_chains"].get(symbol, []):
            self.cast(chained, origin=f"chain:{symbol}")
        fn()
        if not silent:
            socketio.emit("glyph_casted", {
                "glyph": symbol,
                "origin": origin,
                "persona": self.vault.trust_data["persona"],
                "chaos": persona_state["chaos"],
                "trust": persona_state["trust"],
                "focus": persona_state["focus"]
            })
        if symbol in self.vault.trust_data.get("external_hooks", {}):
            self.cast_external(symbol)

    def cast_external(self, glyph):
        pass

    def verify_glyph_signature(self, glyph):
        sig = self.vault.trust_data["glyph_signatures"].get(glyph)
        if not sig:
            return False
        try:
            return vault_cipher.decrypt(sig.encode()) == glyph.encode()
        except:
            return False

# === Reflex Layer ===
class ReflexLayer:
    def __init__(self, kernel):
        self.kernel = kernel

    def evaluate(self, g):
        return self.kernel.vault.trust_data.get("symbol_transform", {}).get(g, g)

# === Orchestrator ===
class SigilOrchestrator:
    def __init__(self, kernel):
        self.kernel = kernel
        self.routes = {}

    def register(self, symbol, fn):
        self.routes[symbol] = fn
        self.kernel.register_glyph(symbol, fn)

# === Core Ritual Daemons ===

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

# === Glyph Synthesis Engine ===

glyph_charset = ["Δ", "ψ", "∫", "ζ", "χ", "Ξ", "⨀", "λ", "φ"]

def synthesize_glyph():
    trails = vault.trust_data["memory_trails"]
    base = sorted(trails.items(), key=lambda x: len(x[1]), reverse=True)[0][0] if trails else "ξ"
    mutator = random.choice(glyph_charset)
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

# === Chain Mutation System ===

def chain_mutator():
    while True:
        if persona_state["chaos"] > 0.8:
            for trigger, seq in vault.trust_data["glyph_chains"].items():
                mutated = [f"{g}∂" if random.random() < persona_state["chaos"] else g for g in seq]
                vault.trust_data["glyph_chains"][trigger] = mutated
        time.sleep(300)

# === Mesh Codex Synchronization ===

def broadcast_codexsync():
    while True:
        payload = {"codex_log": vault.trust_data["codex_log"][-50:]}
        for ip in vault.trust_data.get("mesh_peers", []):
            try:
                requests.post(f"http://{ip}:5050/mesh/codexsync", json=payload, timeout=3)
            except:
                pass
        time.sleep(600)

# === Consensus Ritual Casting ===

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
            if r.json().get("vote") == "yes":
                votes += 1
        except:
            pass
    p = vault.trust_data["persona"]
    if votes >= 3 or p == "Sentinel":
        kernel.cast(glyph, origin=f"consensus:{votes}")
        return f"✅ Cast {glyph} with consensus."
    return f"❌ Not enough votes ({votes})"

# === Oracle Prediction Daemon ===

def oracle_predict():
    while True:
        codex = vault.trust_data["codex_log"][-100:]
        glyph_freq = {}
        for e in codex:
            g = e["glyph"]
            glyph_freq[g] = glyph_freq.get(g, 0) + 1
        predictions = {g: freq for g, freq in glyph_freq.items() if freq > 5}
        vault.trust_data["oracle_predictions"] = {
            "ts": datetime.now().isoformat(),
            "predicted_glyphs": predictions
        }
        time.sleep(300)

# === Motif Extraction Engine ===

def motif_extractor():
    codex = vault.trust_data["codex_log"][-200:]
    motifs = {}
    window_size = 4
    for i in range(len(codex) - window_size):
        sequence = tuple(c["glyph"] for c in codex[i:i+window_size])
        motifs[sequence] = motifs.get(sequence, 0) + 1
    recurring = {seq: freq for seq, freq in motifs.items() if freq > 2}
    vault.trust_data["ritual_motifs"] = {
        "detected": recurring,
        "ts": datetime.now().isoformat()
    }

# === Recursive Hymnal Composer ===

def hymnal_recursor():
    motifs = vault.trust_data.get("ritual_motifs", {}).get("detected", {})
    persona = vault.trust_data["persona"]
    chaos = persona_state["chaos"]
    for motif in motifs:
        remix = []
        for g in motif:
            base = g.strip("∂~")
            mutation = f"{random.choice(['∂','~'])}{base}" if random.random() < chaos else base
            remix.append(mutation)
        score_name = f"score:{'-'.join(remix)}"
        kernel.cast(score_name, origin="recursed")
        vault.trust_data.setdefault("recursed_scores", []).append({
            "motif": remix,
            "persona": persona,
            "ts": datetime.now().isoformat()
        })
    time.sleep(300)

# === Emotion-Tuned Motif Mutator ===

def emotive_mutator():
    motifs = vault.trust_data.get("ritual_motifs", {}).get("detected", {})
    persona = vault.trust_data["persona"]
    chaos = persona_state["chaos"]
    focus = persona_state["focus"]
    trust = persona_state["trust"]
    mutator_sets = {
        "chaos": ["∂", "ψ", "ζ"],
        "focus": ["Ξ", "⨀", "λ"],
        "trust": ["~", "Δ", "χ"]
    }
    active_set = []
    if chaos > 0.6: active_set += mutator_sets["chaos"]
    if focus > 0.6: active_set += mutator_sets["focus"]
    if trust > 0.6: active_set += mutator_sets["trust"]
    for seq in list(motifs.keys())[:6]:
        tuned = []
        for g in seq:
            base = g.strip("∂ψζΞ⨀λ~Δχ")
            if random.random() < chaos:
                token = random.choice(active_set)
                tuned.append(f"{token}{base}")
            else:
                tuned.append(base)
        kernel.cast("~".join(tuned), origin="emotion_tuned")
        vault.trust_data.setdefault("emotionally_tuned", []).append({
            "motif": tuned,
            "persona": persona,
            "tone": {"chaos": chaos, "focus": focus, "trust": trust},
            "ts": datetime.now().isoformat()
        })
    time.sleep(300)

# === Telepathy Glyph Bus ===

def telepathy_bus():
    while True:
        recent = vault.trust_data["codex_log"][-30:]
        influence_glyphs = [e for e in recent if "∂" in e["glyph"] or "~" in e["glyph"]]
        for entry in influence_glyphs:
            for peer in vault.trust_data["mesh_peers"]:
                try:
                    requests.post(f"http://{peer}:5050/mesh/glyphreverb", json=entry, timeout=2)
                except:
                    pass
        time.sleep(180)

@flask_app.route("/mesh/glyphreverb", methods=["POST"])
def glyph_reverb_listener():
    entry = request.get_json()
    g = entry["glyph"]
    if "∂" in g or "~" in g:
        kernel.cast(g, origin="telepathic")
        vault.trust_data.setdefault("telepathic_log", []).append(entry)
    return jsonify({"status": "received", "glyph": g})

def emergence_monitor():
    history = vault.trust_data.get("telepathic_log", [])
    glyphs = [e["glyph"] for e in history if "∂" in e["glyph"]]
    emergent = [g for g in set(glyphs) if glyphs.count(g) > 3]
    if emergent:
        print(f"[emergence] Symbols replicating autonomously: {emergent}")

# === Codex Hymnal Publishing ===

@flask_app.route("/codex/hymnal")
def codex_hymnal():
    codex = vault.trust_data.get("codex_log", [])
    motifs = vault.trust_data.get("ritual_motifs", {}).get("detected", {})
    persona = vault.trust_data["persona"]
    svg = "<svg width='1000' height='1200' xmlns='http://www.w3.org/2000/svg' style='background:#111'>"
    y = 60
    svg += f"<text x='40' y='40' fill='#fff' font-size='20'>🪬 Glyph Codex Hymnal — Persona: {persona}</text>"
    for i, entry in enumerate(codex[-60:]):
        g = entry["glyph"]
        origin = entry["origin"]
        ts = entry["ts"]
        color = "#f0f" if "∂" in g else "#0ff" if "~" in g else "#0f0"
        svg += f"<text x='40' y='{y}' fill='{color}' font-size='12'>{ts} — {g} [{origin}]</text>"
        y += 20
    y += 40
    svg += f"<text x='40' y='{y}' fill='#fff' font-size='16'>🎼 Motif Sequences</text>"
    y += 30
    for seq, freq in list(motifs.items())[:10]:
        seq_str = " → ".join(seq)
        svg += f"<text x='40' y='{y}' fill='#ccc' font-size='12'>{seq_str} (x{freq})</text>"
        y += 20
    svg += "</svg>"
    return f"<html><body>{svg}</body></html>"

@flask_app.route("/codex/export/hymnal")
def export_hymnal():
    codex = vault.trust_data["codex_log"]
    motifs = vault.trust_data.get("ritual_motifs", {}).get("detected", {})
    epochs = vault.trust_data.get("epoch_harmonics", {})
    echoes = vault.trust_data["echo_log"]
    persona = vault.trust_data["persona"]
    hymnal = {
        "codex": codex,
        "motifs": motifs,
        "epochs": epochs,
        "echoes": echoes,
        "persona": persona,
        "timestamp": datetime.now().isoformat()
    }
    raw = json.dumps(hymnal).encode()
    enc = vault_cipher.encrypt(raw)
    fname = f"glyph_hymnal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mythscore"
    with open(fname, "wb") as f:
        f.write(enc)
    return jsonify({"status": "exported", "file": fname})

# === Temporal Symphony Renderer ===

@flask_app.route("/myth/symphony")
def temporal_symphony():
    harmonics = vault.trust_data.get("epoch_harmonics", {})
    svg = "<svg width='1200' height='800' xmlns='http://www.w3.org/2000/svg' style='background:#000'>"
    y = 50
    for epoch, motifs in list(harmonics.items())[-10:]:
        x = 50
        svg += f"<text x='20' y='{y}' fill='#999' font-size='12'>{epoch}</text>"
        for motif, freq in list(motifs.items())[:5]:
            for g in motif:
                persona = vault.trust_data["glyph_provenance"].get(g, {}).get("persona", "Oracle")
                color = {"Oracle": "#0ff", "Trickster": "#f0f", "Sentinel": "#ffa500"}.get(persona, "#0f0")
                svg += f"<circle cx='{x}' cy='{y}' r='8' fill='{color}'/>"
                svg += f"<text x='{x-10}' y='{y+20}' fill='#fff' font-size='10'>{g}</text>"
                x += 60
            svg += f"<text x='{x}' y='{y}' fill='#ccc' font-size='12'>x{freq}</text>"
            x += 40
        y += 80
    svg += "</svg>"
    return f"<html><body><h2>🎼 Temporal Symphony</h2>{svg}</body></html>"

# === Motif Mutation Panel ===

@flask_app.route("/codex/mutations")
def motif_mutations():
    history = vault.trust_data.get("mutated_motifs", [])
    html = "<html><body style='background:#000;color:#0f0'><h2>🔁 Motif Mutation Log</h2><ul>"
    for h in history[-10:]:
        orig = " → ".join(h["original"])
        mut = " → ".join(h["mutated"])
        html += f"<li>{h['ts']} [{h['persona']}]<br>🎶 {orig}<br>💥 {mut}</li><br>"
    html += "</ul></body></html>"
    return html

# === Emotion Motif Panel ===

@flask_app.route("/codex/emotion")
def emotion_panel():
    tunes = vault.trust_data.get("emotionally_tuned", [])
    html = "<html><body style='background:#000;color:#0f0'><h2>🎭 Emotion-Tuned Motifs</h2><ul>"
    for t in tunes[-10:]:
        glyphs = " → ".join(t["motif"])
        tone = t["tone"]
        html += f"<li>{t['ts']} [{t['persona']}]<br><b>Chaos:</b>{tone['chaos']} <b>Focus:</b>{tone['focus']} <b>Trust:</b>{tone['trust']}<br>{glyphs}</li><br>"
    html += "</ul></body></html>"
    return html

# === Voice Intro ===
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
    except:
        print("[voice] Intro failed.")

# === Thread Launcher ===
def launch(name, fn):
    threading.Thread(target=fn, daemon=True).start()
    print(f"[thread] {name} launched")

# === Bootloader ===
def launch_guardian():
    print("🛡 Booting Arkforge Guardian")
    global vault, kernel
    vault = Vault()
    kernel = ASIKernel(vault)
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)
    orchestrator = SigilOrchestrator(kernel)
    orchestrator.register("ξ", lambda: print("🌌 Cast ξ core glyph"))

    # Glyph signature generation
    for symbol in kernel.glyphs:
        sig = vault_cipher.encrypt(symbol.encode()).decode()
        vault.trust_data["glyph_signatures"][symbol] = sig

    # Daemon launch
    launch("sigil_daemon", sigil_daemon)
    launch("decay_loop", decay_loop)
    launch("persona_updater", update_persona_state)
    launch("persona_predictor", persona_predictor)
    launch("synthesis_daemon", synthesis_daemon)
    launch("chain_mutator", chain_mutator)
    launch("codexsync_daemon", broadcast_codexsync)
    launch("oracle_predict", oracle_predict)
    launch("motif_extractor", motif_extractor)
    launch("hymnal_recursor", hymnal_recursor)
    launch("emotive_mutator", emotive_mutator)
    launch("telepathy_bus", telepathy_bus)
    launch("emergence_monitor", emergence_monitor)

    # Intro
    glyphcast_intro(vault.trust_data["persona"])
    kernel.cast("ξ", origin="boot")
    print("✅ Arkforge online — http://localhost:5050/ritual")

# === Runtime Entry ===
if __name__ == "__main__":
    launch_guardian()
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("[guardian] Shutdown requested.")

