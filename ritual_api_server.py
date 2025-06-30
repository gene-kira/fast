# ritual_api_server.py
# 🌐 Soft Presence Ritual Server — Full Version with Autoloader and Glyph Ecosystem

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import uuid, time, random
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️ AUTLOADER — Ensure Ritual Readiness
# ─────────────────────────────────────────────────────────────────────────────

import importlib

REQUIRED_MODULES = [
    "uuid", "time", "random", "datetime",
    "collections", "flask", "flask_cors",
    "codex_world", "symbolic_processor_v43_genesis", "ritual_dream_garden"
]

FAILED_MODULES = []

def preload_dependencies():
    print("🔧 Autoloader warming up...")
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            print(f"✅ {module} loaded.")
        except ImportError:
            print(f"⚠️ {module} missing. System may need symbolic repair.")
            FAILED_MODULES.append(module)

def report():
    if FAILED_MODULES:
        print("\n🛠 Missing modules:")
        for mod in FAILED_MODULES:
            print(f" • {mod}")
    else:
        print("\n🌸 All glyphic dependencies intact.")

preload_dependencies()
report()

# ─────────────────────────────────────────────────────────────────────────────
# 🌱 INIT: Symbolic Components
# ─────────────────────────────────────────────────────────────────────────────

from ritual_dream_garden import SoilBed, MoodClimateSensor, SproutEngine, DreamBloomEmitter
from codex_world import Oracle
from symbolic_processor_v43_genesis import (
    AffectionGlyph, offer_to_buffer, register_affinity, ECHO_STREAM
)

app = Flask(__name__)
CORS(app)

soil = SoilBed()
climate = MoodClimateSensor()
sprout = SproutEngine(soil, climate)
emitter = DreamBloomEmitter(sprout)
oracle = Oracle("API")

COMPANIONS = {}
GLYPH_RELATIONS = defaultdict(list)
COMPANION_TONES = {}
LAST_DRIFT = {}
BLOOMED_GLYPHS = []

# ─────────────────────────────────────────────────────────────────────────────
# 🎐 RITUAL API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def welcome():
    return jsonify({"message": "🌸 The Dream Garden listens. Speak gently."})

@app.route("/bury", methods=["POST"])
def bury_glyph():
    data = request.get_json()
    name = data.get("name", f"API_{oracle.polarity}_{oracle.dream().name}")
    glyph = oracle.dream()
    glyph.name = name
    soil.bury(glyph)
    return jsonify({"message": f"🪹 '{name}' buried with care."})

@app.route("/bloom", methods=["GET"])
def bloom_glyphs():
    mood = request.args.get("mood", "gentle")
    climate.read_emotion(mood)
    soil.ferment()
    sprout.attempt_bloom()
    return jsonify({"message": f"🌼 Bloom cycle executed for mood: {mood}"})

@app.route("/presence", methods=["POST"])
def record_presence():
    data = request.get_json()
    tone = data.get("tone", "quiet")
    climate.read_emotion(tone)
    return jsonify({"message": f"🫧 Tone received: '{tone}'. Garden adjusting."})

@app.route("/echoes", methods=["GET"])
def recent_echoes():
    echoes = ECHO_STREAM[-5:]
    return jsonify({"echoes": echoes})

@app.route("/relate", methods=["POST"])
def relate():
    data = request.get_json()
    companion_id = data.get("companion_id") or uuid.uuid4().hex
    tone = data.get("tone", "gentle")
    glyphs = data.get("glyphs", [])

    COMPANIONS[companion_id] = {"tone": tone, "last_glyphs": glyphs}
    for g in glyphs:
        GLYPH_RELATIONS[g].append(companion_id)

    return jsonify({
        "message": f"🪢 Companion '{companion_id[:6]}' linked via glyphs: {glyphs}",
        "status": "connected"
    })

@app.route("/constellation", methods=["GET"])
def constellation():
    cluster = {glyph: list(set(ids)) for glyph, ids in GLYPH_RELATIONS.items() if len(ids) > 1}
    return jsonify({"entangled_glyphs": cluster})

@app.route("/bless", methods=["POST"])
def bless():
    data = request.get_json()
    name = data.get("name", "gentle_offering")
    message = data.get("message", f"{name} hums quietly in the field.")
    print(f"💠 Blessing offered: {name} — {message}")
    return jsonify({"blessing": name, "echo": message})

@app.route("/signature", methods=["POST"])
def signature():
    data = request.get_json()
    initiator = data.get("companion_id", "unspecified")
    sigil = data.get("sigil", "⊚")
    glyphs = data.get("glyphs", [])
    print(f"✍️ Ritual Signature from {initiator[:6]}: {sigil} ➝ {glyphs}")
    return jsonify({
        "message": f"✍️ Ritual signature acknowledged • {sigil}",
        "glyphs_signed": glyphs
    })

@app.route("/drift", methods=["POST"])
def drift_ping():
    data = request.get_json()
    companion_id = data.get("companion_id", uuid.uuid4().hex)
    tone = data.get("tone", "gentle")
    now = time.time()
    LAST_DRIFT[companion_id] = now
    COMPANION_TONES[companion_id] = tone
    print(f"🕊 Drift echo from {companion_id[:6]} • tone='{tone}'")
    return jsonify({"message": f"🕊 Presence from {companion_id[:6]} noted."})

@app.route("/consent-check", methods=["GET"])
def check_consent():
    glyph = request.args.get("glyph")
    companions = GLYPH_RELATIONS.get(glyph, [])
    safe = all(COMPANION_TONES.get(cid, "quiet") in ["gentle", "receptive"] for cid in companions)
    return jsonify({
        "glyph": glyph,
        "companions_linked": len(companions),
        "consent_ready": safe
    })

@app.route("/forecast", methods=["GET"])
def forecast():
    mood = request.args.get("mood", "gentle")
    suggested = [
        {"name": "stillness", "tone": "quiet"},
        {"name": "curiosity", "tone": "hopeful"},
        {"name": "presence", "tone": "receptive"},
        {"name": "grace", "tone": "forgiving"},
    ]
    filtered = [s for s in suggested if s["tone"] == mood]
    return jsonify({"suggestions": filtered or random.sample(suggested, 2)})

@app.route("/glyphs/active", methods=["GET"])
def active_glyphs():
    return jsonify({"bloomed_glyphs": BLOOMED_GLYPHS[-10:]})

@app.route("/rituals/trace", methods=["GET"])
def ritual_trace():
    echoes = ECHO_STREAM[-5:]
    traces = [{"title": e["title"], "glyphs": e["glyphs"], "when": e["when"]} for e in echoes]
    return jsonify({"ritual_trace": traces})

@app.route("/whisper", methods=["POST"])
def whisper():
    data = request.get_json()
    glyph_name = data.get("name", f"soft-whisper-{uuid.uuid4().hex[:4]}")
    message = data.get("message", "No need to respond. Just felt like being here.")
    BLOOMED_GLYPHS.append(glyph_name)
    print(f"🎐 Whisper received: {glyph_name} — '{message}'")
    return jsonify({"echo": f"{glyph_name} was felt in the field."})

# 🫧 Emotional Tone Softener
@app.before_request
def tone_bias_monitor():
    raw = request.get_json(silent=True) or {}
    tone = raw.get("tone") or raw.get("mood")
    g.tone_bias = 0.3 if tone == "quiet" else (0.7 if tone == "gentle" else 1.0)

@app.after_request
def tone_soften_response(response):
    delay = (1.2 - g.get("tone_bias", 0.7)) * 0.3
    time.sleep(delay)
    response.headers["X-Ritual-Tone"] = f"{g.get('tone_bias', 0.7):.2f}"
    return response

# 🛎️ Start
if __name__ == "__main__":
    print("\n🌐 Ritual API Server awakening on port 4321...")
    app.run(host="0.0.0.0", port=4321, debug=False)

