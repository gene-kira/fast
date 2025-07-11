# ==================== AUTOLOADER ====================
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_libs = {
    "flask": "Flask",
    "pyttsx3": "pyttsx3",
    "pydub": "pydub",
    "flask_socketio": "flask_socketio"
}

for lib, import_name in required_libs.items():
    try:
        if import_name:
            __import__(import_name)
    except ImportError:
        print(f"Installing missing library: {lib}")
        install(lib)

# ==================== IMPORTS ====================
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import re
import pyttsx3
from pydub.generators import Sine, Square
from pydub import AudioSegment
import random

# ==================== INITIALIZATION ====================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
engine = pyttsx3.init()
trust_history = {}
glyph_memory = {}
blacklisted_ips = set()

# ==================== TRUST ENGINE ====================
class TrustAgent:
    def __init__(self):
        self.trust_scores = {}

    def evaluate_trust(self, entity_id, event_type):
        score_change = self._trust_logic(event_type)
        current = self.trust_scores.get(entity_id, 50)
        updated = max(0, min(100, current + score_change))
        self.trust_scores[entity_id] = updated
        trust_history.setdefault(entity_id, []).append(updated)

    def _trust_logic(self, event_type):
        heuristics = {
            "positive_interaction": +10,
            "suspicious_command": -20,
            "failed_auth": -30,
            "glyph_resonance": +15,
            "emotional_disruption": -25,
            "ritual_chain_triggered": +5,
            "volatility_spike": -10
        }
        return heuristics.get(event_type, 0)

    def get_trust_score(self, entity_id):
        return self.trust_scores.get(entity_id, 50)

# ==================== VOLATILITY & MEMORY ====================
def calculate_volatility(history):
    diffs = [abs(history[i+1] - history[i]) for i in range(len(history)-1)]
    return sum(diffs)/len(diffs) if diffs else 0

def log_glyph_memory(entity_id, ritual):
    glyph_memory.setdefault(entity_id, []).append(ritual)

# ==================== RITUAL COMPOSER ====================
def generate_ritual(entity_id, trust, volatility):
    return {
        "entity": entity_id,
        "glyph": "rupture" if trust < 20 else "resonance",
        "shape": "spiral" if volatility > 40 else "circle",
        "aura": "panic" if trust < 20 else "calm",
        "intensity": min(volatility * 2, 100),
        "chain": ["guardian_sigil"] if trust < 20 and volatility > 50 else ["stabilizer"]
    }

# ==================== SOUND SYNTHESIS ====================
def generate_tone(volatility):
    duration = 1000
    base_freq = 440
    if volatility < 10:
        tone = Sine(base_freq).to_audio_segment(duration=duration).fade_in(300).fade_out(300)
    elif volatility < 30:
        tone = Sine(base_freq - 50).to_audio_segment(duration=duration).overlay(
               Sine(base_freq + 30).to_audio_segment(duration=duration))
    elif volatility < 50:
        tone = Square(base_freq + random.randint(20,60)).to_audio_segment(duration=duration)
    else:
        tone = Sine(base_freq + 100).to_audio_segment(duration=duration).apply_gain(+10).reverse()
    tone.export("glyph_tone.wav", format="wav")

# ==================== VOICE & EMISSION ====================
def speak(message):
    engine.say(message)
    engine.runAndWait()

def broadcast_glyph(entity_id, trust, volatility, ritual):
    socketio.emit('glyph_event', {
        "entity": entity_id,
        "trust": trust,
        "volatility": volatility,
        "glyph": ritual["glyph"],
        "aura": ritual["aura"],
        "intensity": ritual["intensity"]
    })

def alert_response(entity_id, trust_agent):
    trust = trust_agent.get_trust_score(entity_id)
    volatility = calculate_volatility(trust_history.get(entity_id, [50]))
    ritual = generate_ritual(entity_id, trust, volatility)
    log_glyph_memory(entity_id, ritual)
    broadcast_glyph(entity_id, trust, volatility, ritual)
    generate_tone(volatility)

    if trust <= 20:
        speak("ALERT: glyph integrity compromised.")
        return {"alert": "HIGH RISK", "ritual": ritual}
    elif trust <= 40:
        speak("Warning: trust destabilizing.")
        return {"alert": "Medium Risk", "ritual": ritual}
    else:
        speak("Trust resonance stable.")
        return {"alert": "Low Risk", "ritual": ritual}

# ==================== RCE SHIELD ====================
def is_safe_input(user_input):
    rce_patterns = [r';', r'\|\|', r'&&', r'\b(eval|exec|system|os\.system)\b', r'<script>']
    for pattern in rce_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return False
    return True

# ==================== IP HANDLER ====================
def get_client_ip():
    return request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)

def block_ip(ip):
    blacklisted_ips.add(ip)

@app.before_request
def check_ip():
    ip = get_client_ip()
    if ip in blacklisted_ips:
        speak("Host sandboxed. Signal terminated.")
        return jsonify({"error": "Access denied", "status": "sandboxed"}), 403

# ==================== SUBMIT ROUTE ====================
trust_agent = TrustAgent()

@app.route('/submit', methods=['POST'])
def handle_input():
    data = request.get_json()
    entity = data.get('user_id', 'anonymous')
    content = data.get('input', '')
    ip = get_client_ip()

    if not is_safe_input(content):
        trust_agent.evaluate_trust(entity, "suspicious_command")
        block_ip(ip)
        return jsonify({"error": "Unsafe input", **alert_response(entity, trust_agent)}), 403

    trust_agent.evaluate_trust(entity, "positive_interaction")
    trust = trust_agent.get_trust_score(entity)
    volatility = calculate_volatility(trust_history.get(entity, [50]))
    ritual = generate_ritual(entity, trust, volatility)
    log_glyph_memory(entity, ritual)
    broadcast_glyph(entity, trust, volatility, ritual)

    return jsonify({
        "message": "Signal harmonized.",
        "trust": trust,
        "volatility": volatility,
        "ritual": ritual,
        **alert_response(entity, trust_agent)
    }), 200

# ==================== RUN ====================
if __name__ == '__main__':
    socketio.run(app, debug=True)

