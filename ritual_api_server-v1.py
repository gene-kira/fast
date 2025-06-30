from flask import Flask, request, jsonify, g, Response
from flask_cors import CORS
import uuid, time, random, importlib

APP_VERSION = "v1.0.0-bloom"

# ─────────────────────────────────────────────────────────────
#⚙️ Ritual Autoloader: Soft Dependency Checker
# ─────────────────────────────────────────────────────────────
REQUIRED_MODULES = ["uuid", "time", "random", "flask", "flask_cors"]
FAILED_MODULES = []

def preload_dependencies():
    print("🔧 Autoloader warming up...")
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            print(f"✅ {module} loaded")
        except ImportError:
            print(f"⚠️ {module} missing — bloom may falter.")
            FAILED_MODULES.append(module)

def report_status():
    if FAILED_MODULES:
        print("🛠️ Ritual incomplete — missing:")
        for mod in FAILED_MODULES:
            print(f" • {mod}")
        print("🙏 Install missing modules before deploying.")
    else:
        print("🌿 All glyphic dependencies intact.")

preload_dependencies()
report_status()

# ─────────────────────────────────────────────────────────────
# 🌼 Ritual Server Setup
app = Flask(__name__)
CORS(app)

BLOOMED_GLYPHS = []
ECHOES = []

@app.before_request
def tone_bias_monitor():
    raw = request.get_json(silent=True) or {}
    tone = raw.get("tone") or raw.get("mood")
    g.tone_bias = 0.3 if tone == "quiet" else (0.7 if tone == "gentle" else 1.0)

@app.after_request
def soften_response(response):
    delay = (1.2 - g.get("tone_bias", 0.7)) * 0.3
    time.sleep(delay)
    response.headers["X-Ritual-Version"] = APP_VERSION
    return response

# ─────────────────────────────────────────────────────────────
# 🌀 Ritual API Routes
@app.route("/")
def root(): return jsonify({"message": "🌼 Ritual Server Ready", "version": APP_VERSION})

@app.route("/presence", methods=["POST"])
def presence():
    tone = request.json.get("tone", "gentle")
    return jsonify({"message": f"Tone received: {tone}"})

@app.route("/whisper", methods=["POST"])
def whisper():
    name = request.json.get("name", f"glyph-{uuid.uuid4().hex[:4]}")
    message = request.json.get("message", "A quiet ripple.")
    BLOOMED_GLYPHS.append(name)
    ECHOES.append({"title": name, "message": message, "when": time.time()})
    return jsonify({"echo": f"{name} whispered into the field."})

@app.route("/glyphs/active")
def active(): return jsonify({"bloomed_glyphs": BLOOMED_GLYPHS[-10:]})

@app.route("/echoes")
def echoes(): return jsonify({"echoes": ECHOES[-5:]})

@app.route("/forecast")
def forecast():
    tone = request.args.get("mood", "gentle")
    pool = [
        {"name": "stillness", "tone": "quiet"},
        {"name": "grace", "tone": "gentle"},
        {"name": "presence", "tone": "hopeful"}
    ]
    match = [g for g in pool if g["tone"] == tone]
    return jsonify({"suggestions": match or random.sample(pool, 1)})

@app.route("/playground")
def playground():
    return """
    <html><body style='font-family:serif;padding:2rem;'>
    <h1>🌟 Ritual Playground</h1>
    <ul>
      <li><a href="/glyphs/active">Active Glyphs</a></li>
      <li><a href="/echoes">Recent Echoes</a></li>
      <li><a href="/forecast?mood=gentle">Forecast (gentle)</a></li>
    </ul>
    <p>Visit <a href="/bloomboard">Bloomboard Interface</a> for symbolic interaction.</p>
    </body></html>
    """

@app.route("/bloomboard")
def bloomboard():
    return Response("""
<!DOCTYPE html><html><head>
<meta charset="UTF-8" /><title>Bloomboard</title>
<style>
body { font-family: 'Georgia', serif; background: #f8f7f3; padding: 2rem; }
.box { max-width: 600px; margin: auto; background: white; border-radius: 1rem;
       padding: 1.5rem 2rem; box-shadow: 0 0 20px rgba(0,0,0,0.05); }
input, select, button { padding: 0.5rem; font-size: 1rem; margin: 0.3rem 0.5rem 0.8rem 0; }
ul { padding-left: 1rem; }
</style>
</head><body>
<div class="box">
  <h1>🌼 Glyph Bloomboard</h1>
  <h2>Whisper</h2>
  <input id="glyphName" placeholder="glyph name" />
  <input id="glyphMsg" placeholder="your message" />
  <button onclick="whisperGlyph()">Whisper</button>
  <h2>Tone</h2>
  <select id="toneSelect" onchange="sendTone()">
    <option value="gentle">gentle</option>
    <option value="quiet">quiet</option>
    <option value="hopeful">hopeful</option>
  </select>
  <h2>Blooms</h2>
  <ul id="bloomList"></ul>
  <h2>Echoes</h2>
  <ul id="echoList"></ul>
</div>
<script>
const API = "";
async function whisperGlyph() {
  const name = document.getElementById("glyphName").value;
  const message = document.getElementById("glyphMsg").value;
  const r = await fetch(API + "/whisper", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ name, message })
  });
  const j = await r.json(); alert(j.echo);
  fetchBloom(); fetchEcho();
}
async function sendTone() {
  const tone = document.getElementById("toneSelect").value;
  await fetch(API + "/presence", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ tone })
  });
}
async function fetchBloom() {
  const r = await fetch(API + "/glyphs/active");
  const j = await r.json();
  const b = document.getElementById("bloomList"); b.innerHTML = "";
  (j.bloomed_glyphs || []).forEach(g => {
    let li = document.createElement("li"); li.textContent = g; b.appendChild(li);
  });
}
async function fetchEcho() {
  const r = await fetch(API + "/echoes");
  const j = await r.json();
  const e = document.getElementById("echoList"); e.innerHTML = "";
  (j.echoes || []).forEach(o => {
    let li = document.createElement("li");
    li.textContent = o.title + ": " + o.message;
    e.appendChild(li);
  });
}
fetchBloom(); fetchEcho();
</script></body></html>
""", mimetype="text/html")

# 🛎️ Start the Ritual Server
if __name__ == "__main__":
    print(f"🌸 Ritual Server {APP_VERSION} active at http://localhost:4321")
    report_status()
    app.run(host="0.0.0.0", port=4321)

