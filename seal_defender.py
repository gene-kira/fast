# === 🛡️ SENTINEL GUARDIAN MODULE ===
import threading, datetime, socket, time, requests, hashlib
import pyttsx3
import psutil

engine = pyttsx3.init()

# === CONFIG ===
ALLOWED_COUNTRIES = ["US"]
ALLOWED_CITIES = ["Junction City", "Wichita"]
TRUST_THRESHOLD = 0.25
ENTROPY_THRESHOLD = 0.85
VERDICT_SPAM_LIMIT = 5

glyph_memory = {"graph": {}, "audit": [], "threats": []}
agent_registry = {}
local_system_id = "Seer_453"

# === 🔍 GEOLOCATION CHECK ===
def get_geo_location():
    try:
        r = requests.get("https://ipinfo.io", timeout=5)
        data = r.json()
        return {
            "ip": data.get("ip", ""),
            "city": data.get("city", ""),
            "country": data.get("country", "")
        }
    except:
        return {"ip": "unknown", "city": "unknown", "country": "unknown"}

def geo_fence_check():
    loc = get_geo_location()
    city = loc["city"]
    country = loc["country"]
    allowed = country in ALLOWED_COUNTRIES and city in ALLOWED_CITIES
    glyph_memory["geo_check"] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "result": "authorized" if allowed else "unauthorized",
        "region": f"{city}, {country}",
        "ip": loc["ip"]
    }
    if not allowed:
        engine.say("Unauthorized region detected. Activating lockdown.")
        engine.runAndWait()
        trigger_lockdown("GeoFence Trigger")
    else:
        print(f"✅ Region authorized: {city}, {country}")
        engine.say("Seal location validated.")
        engine.runAndWait()

# === 🚨 EMERGENCY LOCKDOWN ===
def trigger_lockdown(reason="Unknown"):
    glyph_memory.setdefault("audit", []).append({
        "event": "LOCKDOWN",
        "timestamp": datetime.datetime.now().isoformat(),
        "reason": reason
    })
    engine.say(f"System lockdown activated. Reason: {reason}")
    engine.runAndWait()
    for gid in agent_registry:
        agent_registry[gid]["muted"] = True
    print(f"🛑 Emergency lockdown triggered: {reason}")

# === 🧬 TRUST AND ENTROPY CHECK ===
def compute_entropy(gid):
    verdicts = glyph_memory.get("graph", {}).get(gid, [])
    flips = sum(1 for i in range(1, len(verdicts)) if verdicts[i]["verdict"] != verdicts[i-1]["verdict"])
    return round(min(1.0, flips / (len(verdicts) + 1)), 3)

def scan_for_rogue_agents():
    for gid in glyph_memory.get("graph", {}):
        ent = compute_entropy(gid)
        trust = agent_registry.get(gid, {}).get("trust", 0.5)
        history = glyph_memory["graph"][gid][-5:]
        if ent >= ENTROPY_THRESHOLD and trust <= TRUST_THRESHOLD and len(history) >= 5:
            agent_registry[gid]["muted"] = True
            glyph_memory["threats"].append({
                "glyph": gid,
                "timestamp": datetime.datetime.now().isoformat(),
                "reason": "Rogue behavior detected"
            })
            engine.say(f"Glyph {gid} quarantined due to rogue entropy.")
            engine.runAndWait()
            print(f"⚠️ Quarantined rogue glyph: {gid}")

# === 🛰️ VERDICT SPAM DETECTION ===
def scan_verdict_spammers():
    now = datetime.datetime.now()
    for gid, history in glyph_memory.get("graph", {}).items():
        recent = [v for v in history if "timestamp" in v and (now - datetime.datetime.fromisoformat(v["timestamp"])).seconds < 60]
        if len(recent) >= VERDICT_SPAM_LIMIT:
            agent_registry.setdefault(gid, {})["muted"] = True
            glyph_memory["threats"].append({
                "glyph": gid,
                "timestamp": datetime.datetime.now().isoformat(),
                "reason": "Verdict spam detected"
            })
            engine.say(f"Glyph {gid} muted due to spam.")
            engine.runAndWait()
            print(f"⚠️ Muted spam agent: {gid}")

# === 🌐 PORT SCANNER ===
def scan_open_ports():
    for conn in psutil.net_connections(kind='inet'):
        if conn.status == 'LISTEN':
            port = conn.laddr.port
            if port not in [45201,45202,45203,7654,8080]:
                glyph_memory["threats"].append({
                    "port": port,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "reason": "Unexpected port open"
                })
                engine.say(f"Suspicious port {port} detected.")
                engine.runAndWait()
                print(f"⚠️ Port anomaly: {port}")

# === 🔐 SEAL GUARDIAN STARTUP ===
def activate_defense():
    print("🛡️ Seal Guardian initializing...")
    geo_fence_check()
    scan_for_rogue_agents()
    scan_verdict_spammers()
    scan_open_ports()

    def defense_loop():
        while True:
            scan_for_rogue_agents()
            scan_verdict_spammers()
            scan_open_ports()
            time.sleep(10)

    threading.Thread(target=defense_loop, daemon=True).start()
    print("🧿 Sentinel running. Watching all systems.")

# === 🚀 AUTO-START ===
if __name__ == "__main__":
    activate_defense()

