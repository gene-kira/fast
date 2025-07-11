import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_libs = {
    "flask": "Flask",
    "re": None,  # built-in
    "json": None,  # built-in
    "pyttsx3": "pyttsx3",
    "pydub": "pydub",
}

for lib, import_name in required_libs.items():
    try:
        if import_name:
            __import__(import_name)
    except ImportError:
        print(f"Installing missing library: {lib}")
        install(lib)

# ================= AUTOLOADER ===================
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_libs = {
    "flask": "Flask",
    "re": None,
    "json": None,
    "pyttsx3": "pyttsx3",
    "pydub": "pydub",
}

for lib, import_name in required_libs.items():
    try:
        if import_name:
            __import__(import_name)
    except ImportError:
        print(f"Installing missing library: {lib}")
        install(lib)

# ================= IMPORTS ===================
from flask import Flask, request, jsonify
import re
import json
import pyttsx3
from pydub.generators import Sine, Square
from pydub import AudioSegment
import random

# ================= INITIALIZATION ===================
app = Flask(__name__)
engine = pyttsx3.init()
blacklisted_ips = set()

# ================= SPEAK FUNCTION ===================
def speak(message):
    engine.say(message)
    engine.runAndWait()

# ================= TRUST ENGINE ===================
class TrustAgent:
    def __init__(self):
        self.trust_scores = {}

    def evaluate_trust(self, user_id, event_type, metadata=None):
        score_change = self._trust_logic(event_type)
        self.trust_scores[user_id] = self.trust_scores.get(user_id, 50) + score_change
        self.trust_scores[user_id] = max(0, min(100, self.trust_scores[user_id]))

    def _trust_logic(self, event_type):
        heuristics = {
            "positive_interaction": +10,
            "neutral_ping": 0,
            "suspicious_command": -20,
            "failed_auth": -30,
            "glyph_resonance": +15,
            "emotional_disruption": -25
        }
        return heuristics.get(event_type, 0)

    def get_trust_score(self, user_id):
        return self.trust_scores.get(user_id, 50)

# ================= TRUST VOLATILITY ===================
def calculate_volatility(history):
    diffs = [abs(history[i+1] - history[i]) for i in range(len(history)-1)]
    return sum(diffs) / len(diffs) if diffs else 0

# ================= ALERT LOGIC ===================
def check_alerts(user_id, trust_agent):
    score = trust_agent.get_trust_score(user_id)

    if score <= 20:
        message = "ALERT! High risk detected."
        speak(message)
        generate_tone(score)
        return {"alert": "HIGH RISK", "message": message}
    elif score <= 40:
        message = "Caution: Medium risk detected."
        speak(message)
        generate_tone(score)
        return {"alert": "Medium Risk", "message": message}
    else:
        message = "Trust levels stable."
        speak(message)
        generate_tone(score)
        return {"alert": "Low Risk", "message": message}

# ================= SONIC GLYPH ===================
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

    tone.export("trust_resonance.wav", format="wav")

# ================= IP LOGIC ===================
def get_client_ip():
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        ip = request.environ['HTTP_X_FORWARDED_FOR']
    else:
        ip = request.remote_addr
    return ip

def block_ip(ip):
    blacklisted_ips.add(ip)

@app.before_request
def deny_blacklisted_ips():
    ip = get_client_ip()
    if ip in blacklisted_ips:
        return jsonify({"error": "Access denied"}), 403

# ================= RCE SHIELD ===================
def is_safe_input(user_input):
    rce_patterns = [r';', r'\|\|', r'&&', r'\b(eval|exec|system|os\.system)\b', r'<script>']
    for pattern in rce_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return False
    return True

# ================= MAIN ROUTE ===================
trust_agent = TrustAgent()
trust_history = []

@app.route('/submit', methods=['POST'])
def handle_input():
    data = request.get_json()
    user_id = data.get('user_id', 'anonymous')
    user_input = data.get('input', '')
    ip_address = get_client_ip()

    if not is_safe_input(user_input):
        trust_agent.evaluate_trust(user_id, "suspicious_command")
        block_ip(ip_address)
        return jsonify({"error": "Unsafe input", **check_alerts(user_id, trust_agent)}), 403

    trust_agent.evaluate_trust(user_id, "positive_interaction")
    trust_score = trust_agent.get_trust_score(user_id)
    trust_history.append(trust_score)
    volatility = calculate_volatility(trust_history)
    return jsonify({"message": "Input accepted", "trust": trust_score, "volatility": volatility, **check_alerts(user_id, trust_agent)}), 200

# ================= RUN ===================
if __name__ == '__main__':
    app.run(debug=True)

