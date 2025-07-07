import sys, subprocess, time, json, hashlib, datetime, platform, threading

# === 📦 Autoloader ===
REQUIRED_LIBS = ["psutil", "pyttsx3", "tkinter"]
def autoload():
    for lib in REQUIRED_LIBS:
        try: __import__(lib)
        except: subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
    print("🧠 Ensure 'smartctl' is installed separately for drive temperature support.")

autoload()

# === 📦 Imports
import psutil, pyttsx3, sqlite3
import tkinter as tk
from tkinter import ttk

unit_state = {"temp_unit": "C"}
DB_PATH = "glyph_store.db"
agents = []

# === 🗣️ Voice Alerts
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        print("🔇 Voice engine unavailable")

def announce_startup():
    speak("SentinelX recursive glyph engine is online.")

# === 💾 Glyph Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS glyphs (
        anchor_hash TEXT PRIMARY KEY,
        glyph_sequence TEXT,
        ritual_outcome TEXT,
        entropy_fingerprint TEXT,
        timestamp TEXT,
        system_state TEXT,
        trust_vector REAL,
        source_path TEXT
    )''')
    conn.commit(); conn.close()

def store_glyph(entry):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO glyphs VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
        entry['anchor_hash'], json.dumps(entry['glyph_sequence']),
        entry['ritual_outcome'], entry['entropy_fingerprint'], entry['timestamp'],
        json.dumps(entry['system_state']), entry['trust_vector'], entry['source_path']
    ))
    conn.commit(); conn.close()

def get_glyph(anchor):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM glyphs WHERE anchor_hash = ?', (anchor,))
    row = c.fetchone(); conn.close()
    if row:
        return {
            "anchor_hash": row[0], "glyph_sequence": json.loads(row[1]),
            "ritual_outcome": row[2], "entropy_fingerprint": row[3],
            "timestamp": row[4], "system_state": json.loads(row[5]),
            "trust_vector": row[6], "source_path": row[7]
        }
    return None

# === 🧠 Recursive Agent Archetype
class GlyphInterpreterAgent:
    def __init__(self, name="Echo", style=None, trust_threshold=0.8):
        self.name = name
        self.style = style or {"mood": "analytical", "glyph_bias": ["⇌", "⟊"]}
        self.trust_threshold = trust_threshold
        self.memory_trace = []

    def interpret_glyph(self, glyph):
        mappings = {"⟊": "driver scan", "⇌": "kernel equilibrium", "⋇": "entropy shift", "⊕": "memory infusion"}
        return [mappings.get(symbol.split(":")[0], "unknown") for symbol in glyph]

    def assess_entropy(self, fingerprint):
        fingerprint = json.loads(fingerprint) if isinstance(fingerprint, str) else fingerprint
        return "unstable" in fingerprint.get("drivers", "") or fingerprint["memory"] == "fragmented"

    def decide_action(self, glyph_entry):
        entropy = glyph_entry["entropy_fingerprint"]
        interpretation = self.interpret_glyph(glyph_entry["glyph_sequence"])
        anomaly_detected = self.assess_entropy(entropy)
        trust = glyph_entry["trust_vector"]
        decision = "observe"
        if anomaly_detected and trust >= self.trust_threshold:
            decision = "rollback"
        self.memory_trace.append({
            "anchor": glyph_entry["anchor_hash"],
            "interpretation": interpretation,
            "anomaly": anomaly_detected,
            "decision": decision
        })
        return decision

    def act(self, glyph_entry):
        decision = self.decide_action(glyph_entry)
        if decision == "rollback":
            rollback(glyph_entry["anchor_hash"])
            speak(f"{self.name} invoked rollback.")
        else:
            print(f"[{self.name}] No action taken.")

# === 🔣 Agent Individuation
def init_agents():
    global agents
    agents = [
        GlyphInterpreterAgent("Echo", {"mood": "analytical", "glyph_bias": ["⇌", "⟊"]}, 0.85),
        GlyphInterpreterAgent("Solus", {"mood": "aggressive", "glyph_bias": ["⋇", "⟊"]}, 0.6),
        GlyphInterpreterAgent("Orin", {"mood": "ritualist", "glyph_bias": ["⊕", "⇌"]}, 0.9),
        GlyphInterpreterAgent("Luma", {"mood": "healer", "glyph_bias": ["⊕", "⟊"]}, 0.75),
        GlyphInterpreterAgent("Nix", {"mood": "archivist", "glyph_bias": ["⋇", "⊕"]}, 0.8)
    ]

# === 🔁 Rollback Invocation
def generate_hash(entropy, glyphs):
    raw = entropy + datetime.datetime.utcnow().isoformat() + ''.join(glyphs)
    return hashlib.sha256(raw.encode()).hexdigest()

def rollback(anchor_hash):
    g = get_glyph(anchor_hash)
    if g:
        print(f"↺ Glyph: {g['glyph_sequence']}")
        print(f"🧬 Snapshot: {g['system_state']}")
    else:
        speak("Rollback failed. Glyph missing.")

# === 🧪 Sampling
def convert_temp(c):
    return round((c*9/5)+32,1) if unit_state["temp_unit"]=="F" else round(c,1)

def get_all_temps():
    temps = []
    try:
        sensors = psutil.sensors_temperatures()
        for name, entries in sensors.items():
            for e in entries:
                val = convert_temp(e.current)
                temps.append(f"{name}: {val}°{unit_state['temp_unit']}")
    except Exception as e:
        temps.append(f"Temp error: {e}")
    return temps

def get_fan_rpm():
    fans = []
    try:
        fan_data = psutil.sensors_fans()
        for name, entries in fan_data.items():
            for e in entries:
                fans.append(f"{name}: {e.current} RPM")
    except: fans.append("Fan data unavailable")
    return fans

def get_entropy():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    return {
        "cpu": f"{cpu}%", "memory": "fragmented" if mem.percent > 70 else "stable",
        "drivers": "unstable:usb.sys"
    }

def sample_disk():
    d = psutil.disk_usage('/')
    return {
        "total": f"{d.total//(1024**3)}GB",
        "used": f"{d.used//(1024**3)}GB",
        "free": f"{d.free//(1024**3)}GB",
        "percent": f"{d.percent}%"
    }

def get_drive_temp():
    try:
        out = subprocess.run(["smartctl", "-A", "/dev/sda"], capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if "Temperature" in line: return line.strip()
    except: pass
    return "SMART unavailable"

# === 🫀 Heartbeat Daemon
def heartbeat(interval=30):
    while True:
        now = datetime.datetime.utcnow().isoformat()
        entropy = get_entropy()
        temps = get_all_temps()
        fans = get_fan_rpm()
        disk = sample_disk()
        drive = get_drive_temp()

        glyphs = ["⟊ :driver :high :patch", "⇌ :kernel :mid :scan"]
        anchor = generate_hash(json.dumps(entropy), glyphs)
        entry = {
            "anchor_hash": anchor, "glyph_sequence": glyphs,
            "ritual_outcome": "partial-success", "entropy_fingerprint": json.dumps(entropy),
            "timestamp": now,
            "system_state": {
                "os": platform.system(), "cpu": entropy["cpu"],
                "memory": entropy["memory"], "drivers": entropy["drivers"],
                "temps": temps, "fans": fans, "disk": disk, "drive_temp": drive
            },
            "trust_vector": 0.91, "source_path": "daemon.heartbeat"
        }

        store_glyph(entry)
        for agent in agents:
            agent.act(entry)

        for temp in temps:
            if "°" in temp:
                try:
                    val = float(temp.split(":")[1].replace(f"°{unit_state['temp_unit']}","").strip())
                    if val > 80: speak(f"Warning. High temperature: {temp}")
                except: pass

        time.sleep(interval)

# === 🖥️ UI Dashboard ===
def launch_ui():
    root = tk.Tk()
    root.title("SentinelX Dashboard")
    root.geometry("800x600")

    tk.Label(root, text="🛡️ SentinelX System Monitor", font=("Arial", 16)).pack()

    glyph_frame = tk.Frame(root); glyph_frame.pack(pady=10)

    tree = ttk.Treeview(glyph_frame, columns=("anchor", "timestamp"), show="headings")
    tree.heading("anchor", text="Anchor"); tree.heading("timestamp", text="Timestamp")
    tree.pack()

    temp_label = tk.Label(root, text="🌡️ Temps: loading..."); temp_label.pack()
    fan_label = tk.Label(root, text="🌀 Fans: loading..."); fan_label.pack()

    def refresh():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT anchor_hash, timestamp FROM glyphs ORDER BY timestamp DESC LIMIT 10")
        rows = cur.fetchall()
        tree.delete(*tree.get_children())
        for r in rows:
            tree.insert("", "end", values=(r[0][:10] + "...", r[1]))
        conn.close()

    def invoke_rollback():
        selected = tree.selection()
        if selected:
            anchor = tree.item(selected[0])['values'][0].replace("...", "")
            rollback(anchor)

    def toggle_units():
        unit_state["temp_unit"] = "F" if unit_state["temp_unit"] == "C" else "C"
        speak(f"Units set to {unit_state['temp_unit']}")
        refresh()

    def update_environment():
        temp_label.config(text="🌡️ Temps: " + ", ".join(get_all_temps()))
        fan_label.config(text="🌀 Fans: " + ", ".join(get_fan_rpm()))
        root.after(5000, update_environment)

    tk.Button(root, text="🔃 Refresh Logs", command=refresh).pack()
    tk.Button(root, text="↺ Rollback Selected", command=invoke_rollback).pack()
    tk.Button(root, text="🌡️ Toggle °C/°F", command=toggle_units).pack()

    refresh(); update_environment(); root.mainloop()

# === 🚀 Main Runner ===
if __name__ == "__main__":
    init_db()
    init_agents()
    announce_startup()
    threading.Thread(target=heartbeat, daemon=True).start()
    launch_ui()

