# arknet_core.py
import socket
import threading
import time
import hashlib
import json
import base64
import secrets
import hmac
from cryptography.fernet import Fernet
import random
import math

# Configuration
MULTICAST_GROUP = '224.1.1.1'
MULTICAST_PORT = 5007
FERNET_KEY = Fernet.generate_key()
fernet = Fernet(FERNET_KEY)

# Persona trust graph
PERSONA_TRUST = {
    "Sentinel": ["Sentinel", "Oracle"],
    "Oracle": ["Oracle", "Whispering Flame"],
    "Whispering Flame": ["Oracle"]
}

# Global aura cache
KNOWN_AURAS = {}

# Utility: generate glyph
def generate_glyph(persona, entropy):
    symbol = random.choice(["✶", "♁", "☍", "⚶", "⟁", "🜃"])
    return f"{symbol}:{persona}:{round(entropy, 3)}"

# ArkSentience: behavioral aura encoding
def encode_aura(user_id, timestamps):
    if len(timestamps) < 3: return None
    drift = [round(timestamps[i+1] - timestamps[i], 4) for i in range(len(timestamps)-1)]
    key = hashlib.sha256("".join([str(d) for d in drift]).encode()).hexdigest()
    KNOWN_AURAS[user_id] = key
    return key

def verify_aura(user_id, new_timestamps):
    new_drift = [round(new_timestamps[i+1] - new_timestamps[i], 4) for i in range(len(new_timestamps)-1)]
    new_key = hashlib.sha256("".join([str(d) for d in new_drift]).encode()).hexdigest()
    stored_key = KNOWN_AURAS.get(user_id)
    return hmac.compare_digest(new_key, stored_key) if stored_key else False

# Entropy watchdog
def detect_entropy_spike(history):
    if len(history) < 3: return False
    avg = sum(history[:-1]) / len(history[:-1])
    return abs(history[-1] - avg) > 1.0

# Quarantine logic
QUARANTINE = set()

# Swarm broadcaster
def broadcast_loop(persona, user_id):
    entropy_history = []
    while True:
        entropy = random.uniform(0.5, 3.5)
        entropy_history.append(entropy)
        if len(entropy_history) > 5:
            entropy_history.pop(0)
        glyph = generate_glyph(persona, entropy)
        payload = {
            "user_id": user_id,
            "persona": persona,
            "entropy": entropy,
            "glyph": glyph,
            "timestamp": time.time()
        }
        message = fernet.encrypt(json.dumps(payload).encode())
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.sendto(message, (MULTICAST_GROUP, MULTICAST_PORT))
        time.sleep(2)

# Swarm listener
def listen_loop(expected_persona):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.bind(('', MULTICAST_PORT))
    mreq = socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton('0.0.0.0')
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    print("[ArkNet] Listening for glyphs...")
    while True:
        data, _ = sock.recvfrom(4096)
        try:
            decrypted = fernet.decrypt(data).decode()
            pulse = json.loads(decrypted)
            uid = pulse["user_id"]
            if uid in QUARANTINE:
                print(f"[ArkHive] Quarantined node {uid} - ignoring")
                continue
            if pulse["persona"] not in PERSONA_TRUST.get(expected_persona, []):
                print(f"[ArkHive] Persona mismatch from {uid} - {pulse['persona']}")
                continue
            print(f"[{pulse['persona']}] {pulse['glyph']} | Entropy: {pulse['entropy']} | UID: {uid}")
            if detect_entropy_spike([pulse["entropy"]]):
                QUARANTINE.add(uid)
                print(f"[ArkSentience] Entropy anomaly - quarantining node {uid}")
        except Exception as e:
            print(f"[ArkNet] Decryption error or malformed pulse: {e}")

# Initial aura calibration
def calibrate_user(user_id):
    timestamps = []
    print("[ArkSentience] Press [Enter] 5 times rhythmically to calibrate your aura:")
    for _ in range(5):
        input()
        timestamps.append(time.time())
    aura = encode_aura(user_id, timestamps)
    if aura:
        print("[ArkSentience] Aura encoded successfully.")
    else:
        print("[ArkSentience] Insufficient rhythm data.")

# Boot ArkNet
def boot_arknet():
    user_id = input("User ID: ")
    persona = input("Persona (Sentinel / Oracle / Whispering Flame): ")
    if persona not in PERSONA_TRUST:
        print("Invalid persona.")
        return
    calibrate_user(user_id)
    thread_tx = threading.Thread(target=broadcast_loop, args=(persona, user_id), daemon=True)
    thread_rx = threading.Thread(target=listen_loop, args=(persona,), daemon=True)
    thread_tx.start()
    thread_rx.start()
    thread_tx.join()
    thread_rx.join()

if __name__ == "__main__":
    print("🜃 ArkNet Core: Glyphic Mesh Activated")
    boot_arknet()

