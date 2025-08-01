import sys, subprocess, time, json, hashlib, datetime, platform, threading

# === 🧰 Autoloader for Dependencies
REQUIRED_LIBS = ["psutil", "pyttsx3", "tkinter", "sqlite3"]
def autoload():
    for lib in REQUIRED_LIBS:
        try: __import__(lib)
        except: subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
autoload()

# === 📦 Imports
import psutil, pyttsx3, sqlite3
import tkinter as tk
from tkinter import ttk

# === 🌐 Global State
unit_state = {"temp_unit": "C"}
voice_state = {"enabled": True, "volume": 1.0}
DB_PATH = "glyph_store.db"
TUNING_DB = "tuning_glyphs.db"
MEMORY_DB = "agent_memory.db"
agents = []

# === 🔉 Thread-Safe Voice Engine
voice_lock = threading.Lock()
def speak(text):
    if not voice_state["enabled"]: return
    try:
        with voice_lock:
            engine = pyttsx3.init()
            engine.setProperty('volume', voice_state["volume"])
            engine.say(text); engine.runAndWait(); engine.stop()
    except Exception as e:
        print(f"🔇 Voice error: {e}")

def announce_startup():
    speak("SentinelX recursive engine online.")

# === 💾 Glyph Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS glyphs (
        anchor_hash TEXT PRIMARY KEY,
        glyph_sequence TEXT,
        ritual_outcome TEXT,
        entropy_fingerprint TEXT,
        timestamp TEXT,
        system_state TEXT,
        trust_vector REAL,
        source_path TEXT
    )'''); conn.commit(); conn.close()

def store_glyph(entry):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''INSERT OR REPLACE INTO glyphs VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
        entry['anchor_hash'], json.dumps(entry['glyph_sequence']),
        entry['ritual_outcome'], entry['entropy_fingerprint'], entry['timestamp'],
        json.dumps(entry['system_state']), entry['trust_vector'], entry['source_path']
    )); conn.commit(); conn.close()

def get_glyph(anchor):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT * FROM glyphs WHERE anchor_hash = ?', (anchor,))
    row = cur.fetchone(); conn.close()
    if row:
        return {
            "anchor_hash": row[0], "glyph_sequence": json.loads(row[1]),
            "ritual_outcome": row[2], "entropy_fingerprint": row[3],
            "timestamp": row[4], "system_state": json.loads(row[5]),
            "trust_vector": row[6], "source_path": row[7]
        }
    return None

# === 💾 Tuning Glyphs Database
def init_tuning_db():
    conn = sqlite3.connect(TUNING_DB)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS tuning_glyphs (
        tuning_id TEXT PRIMARY KEY,
        timestamp TEXT,
        method TEXT,
        target TEXT,
        outcome_snapshot TEXT,
        feedback_score REAL,
        agent_name TEXT
    )'''); conn.commit(); conn.close()

def log_tuning_glyph(entry):
    try:
        conn = sqlite3.connect(TUNING_DB)
        cur = conn.cursor()
        cur.execute('''INSERT OR REPLACE INTO tuning_glyphs VALUES (?, ?, ?, ?, ?, ?, ?)''', (
            entry['tuning_id'], entry['timestamp'], entry['method'], entry['target'],
            json.dumps(entry['outcome_snapshot']), entry['feedback_score'], entry['agent_name']
        )); conn.commit(); conn.close()
    except Exception as e:
        print(f"⚠️ Tuning glyph logging failed: {e}")

# === 💾 Agent Memory Trace Database
def init_memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS agent_memory (
        trace_id TEXT PRIMARY KEY,
        agent_name TEXT,
        anchor_hash TEXT,
        decision TEXT,
        feedback_score REAL,
        entropy_snapshot TEXT,
        glyph_sequence TEXT,
        timestamp TEXT
    )'''); conn.commit(); conn.close()

def log_memory_trace(agent, glyph_entry, decision, feedback_score):
    try:
        trace_id = hashlib.sha256((glyph_entry['anchor_hash'] + agent.name).encode()).hexdigest()
        conn = sqlite3.connect(MEMORY_DB)
        cur = conn.cursor()
        cur.execute('''INSERT OR REPLACE INTO agent_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
            trace_id, agent.name, glyph_entry['anchor_hash'], decision,
            feedback_score, glyph_entry["entropy_fingerprint"],
            json.dumps(glyph_entry["glyph_sequence"]), glyph_entry["timestamp"]
        )); conn.commit(); conn.close()
    except Exception as e:
        print(f"⚠️ Memory trace logging failed: {e}")

def load_memory_trace(agent_name, limit=20):
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute("SELECT glyph_sequence, decision, feedback_score FROM agent_memory WHERE agent_name=? ORDER BY timestamp DESC LIMIT ?", (agent_name, limit))
    rows = cur.fetchall(); conn.close()
    return rows

# === 🔀 CPU Affinity Redistributor
def redistribute_affinity():
    try:
        core_loads = psutil.cpu_percent(percpu=True)
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            p = psutil.Process(proc.info['pid'])
            if proc.info['cpu_percent'] > 50:
                new_affinity = sorted(range(len(core_loads)), key=lambda i: core_loads[i])[:2]
                p.cpu_affinity(new_affinity)
    except Exception as e:
        print(f"⚠️ Affinity redistribution error: {e}")

# === 🤖 Agents

class GlyphInterpreterAgent:
    def __init__(self, name="Echo", style=None, trust_threshold=0.8):
        self.name = name
        self.style = style or {"mood": "analytical"}
        self.trust_threshold = trust_threshold
        self.memory_trace = []

    def interpret_glyph(self, glyph):
        mappings = {"⟊": "driver scan", "⇌": "kernel equilibrium", "⋇": "entropy shift", "⊕": "memory infusion", "⇄": "affinity rebalance"}
        return [mappings.get(symbol.split(":")[0], "unknown") for symbol in glyph]

    def assess_entropy(self, fingerprint):
        fingerprint = json.loads(fingerprint) if isinstance(fingerprint, str) else fingerprint
        return "unstable" in fingerprint.get("drivers", "") or fingerprint.get("memory") == "fragmented"

    def decide_action(self, glyph_entry):
        entropy = glyph_entry["entropy_fingerprint"]
        interpretation = self.interpret_glyph(glyph_entry["glyph_sequence"])
        anomaly = self.assess_entropy(entropy)
        trust = glyph_entry["trust_vector"]
        decision = "observe"
        if anomaly and trust >= self.trust_threshold:
            decision = "rollback"
        self.memory_trace.append({"anchor": glyph_entry["anchor_hash"], "interpretation": interpretation, "anomaly": anomaly, "decision": decision})
        log_memory_trace(self, glyph_entry, decision, 0.0)
        return decision

    def act(self, glyph_entry):
        decision = self.decide_action(glyph_entry)
        if decision == "rollback":
            rollback(glyph_entry["anchor_hash"])
            speak(f"{self.name} initiated rollback.")
        else:
            print(f"[{self.name}] Observing.")

class GlyphTunerAgent(GlyphInterpreterAgent):
    def __init__(self, name="TunerX", trust_threshold=0.75):
        super().__init__(name, {"mood": "optimizer"}, trust_threshold)
        self.last_entropy = {}

    def tune_cpu(self):
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            if proc.info['cpu_percent'] > 50:
                try: psutil.Process(proc.info['pid']).nice(psutil.IDLE_PRIORITY_CLASS)
                except: pass
        redistribute_affinity()

    def score_feedback(self, previous, current):
        try:
            score = 0.5
            if previous.get("memory") == "fragmented" and current.get("memory") == "stable": score += 0.3
            if "unstable" not in current.get("drivers", ""): score += 0.2
            return round(score, 2)
        except: return 0.5

    def act(self, glyph_entry):
        try:
            entropy = json.loads(glyph_entry["entropy_fingerprint"])
            timestamp = glyph_entry.get("timestamp", datetime.datetime.utcnow().isoformat())
            if "fragmented" in entropy.get("memory", "") or "unstable" in entropy.get("drivers", ""):
                self.tune_cpu()
                score = self.score_feedback(self.last_entropy, entropy)
                tuning_id = hashlib.sha256((timestamp + self.name + "parallel").encode()).hexdigest()
                log_tuning_glyph({
                    "tuning_id": tuning_id, "timestamp": timestamp, "method": "cpu_affinity_redistribution",
                    "target": "cpu_parallel", "outcome_snapshot": entropy,
                    "feedback_score": score, "agent_name": self.name
                })
                log_memory_trace(self, glyph_entry, "tune", score)
                speak(f"{self.name} tuned affinity. Score: {score}")
            self.last_entropy = entropy
            super().act(glyph_entry)
        except Exception as e:
            print(f"⚠️ TunerX error: {e}")

# === 🔁 Rollback + Entropy

def generate_hash(entropy, glyphs):
    raw = entropy + datetime.datetime.utcnow().isoformat() + ''.join(glyphs)
    return hashlib.sha256(raw.encode()).hexdigest()

def rollback(anchor_hash):
    g = get_glyph(anchor_hash)
    if g: print(f"↺ Glyph: {g['glyph_sequence']}\n🧬 Snapshot: {g['system_state']}")
    else: speak("Rollback failed. Glyph missing.")

def get_entropy():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    return {"cpu": f"{cpu}%", "memory": "fragmented" if mem.percent > 70 else "stable", "drivers": "unstable:usb.sys"}

def get_all_temps():
    temps = []
    try:
        sensors = psutil.sensors_temperatures()
        for name, entries in sensors.items():
            for e in entries:
                val = round((e.current*9/5)+32,1) if unit_state["temp_unit"]=="F" else round(e.current,1)
                temps.append(f"{name}: {val}°{unit_state['temp_unit']}")
    except: temps.append("Temp error")
    return temps

def get_fan_rpm():
    fans = []
    try:
        fan_data = psutil.sensors_fans()
        for name, entries in fan_data.items():
            for e in entries: fans.append(f"{name}: {e.current} RPM")
    except: fans.append("Fan error")
    return fans

def sample_disk():
    d = psutil.disk_usage('/')
    return {"total": f"{d.total//(1024**3)}GB", "used": f"{d.used//(1024**3)}GB", "free": f"{d.free//(1024**3)}GB", "percent": f"{d.percent}%"}

def get_drive_temp():
    try:
        out = subprocess.run(["smartctl", "-A", "/dev/sda"], capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if "Temperature" in line: return line.strip()
    except: pass
    return "SMART unavailable"

# === 🫀 Heartbeat

def heartbeat(interval=30):
    while True:
        try:
            now = datetime.datetime.utcnow().isoformat()
            entropy = get_entropy()
            temps = get_all_temps()
            fans = get_fan_rpm()
            disk = sample_disk()
            drive = get_drive_temp()
            glyphs = ["⟊ :driver :patch", "⇌ :kernel :scan", "⇄ :affinity :rebalance"]
            anchor = generate_hash(json.dumps(entropy), glyphs)
            entry = {
                "anchor_hash": anchor, "glyph_sequence": glyphs, "ritual_outcome": "partial-success",
                "entropy_fingerprint": json.dumps(entropy), "timestamp": now,
                "system_state": {
                    "os": platform.system(), "cpu": entropy["cpu"], "memory": entropy["memory"],
                    "drivers": entropy["drivers"], "temps": temps, "fans": fans,
                    "disk": disk, "drive_temp": drive
                }, "trust_vector": 0.91, "source_path": "daemon.heartbeat"
            }
            store_glyph(entry)
            for agent in agents: agent.act(entry)
            for temp in temps:
                if "°" in temp:
                    try:
                        val = float(temp.split(":")[1].replace(f"°{unit_state['temp_unit']}", "").strip())
                        if val > 80: speak(f"High temperature: {temp}")
                    except: pass
            time.sleep(interval)
        except Exception as e:
            print(f"⚠️ Heartbeat error: {e}")
            time.sleep(10)

# === 🖥️ UI Dashboard

def launch_ui():
    root = tk.Tk()
    root.title("SentinelX Dashboard")
    root.geometry("800x600")

    tk.Label(root, text="🛡️ SentinelX Monitor", font=("Arial", 16)).pack(pady=10)

    tree_frame = tk.Frame(root); tree_frame.pack(pady=10)
    tree = ttk.Treeview(tree_frame, columns=("anchor", "timestamp"), show="headings")
    tree.heading("anchor", text="Anchor")
    tree.heading("timestamp", text="Timestamp")
    tree.pack()

    temp_label = tk.Label(root, text="🌡️ Temps: loading..."); temp_label.pack()
    fan_label = tk.Label(root, text="🌀 Fans: loading..."); fan_label.pack()

    voice_frame = tk.Frame(root); voice_frame.pack(pady=10)
    tk.Label(voice_frame, text="🗣️ Voice Control").pack()

    def toggle_voice():
        voice_state["enabled"] = not voice_state["enabled"]
        status = "enabled" if voice_state["enabled"] else "disabled"
        speak(f"Voice {status}")
        voice_status_btn.config(text=f"🔘 Voice: {status.capitalize()}")

    voice_status_btn = tk.Button(voice_frame, text="🔘 Voice: Enabled", command=toggle_voice)
    voice_status_btn.pack()

    def set_volume(val): voice_state["volume"] = float(val)/100
    tk.Label(voice_frame, text="🔊 Volume").pack()
    volume_slider = tk.Scale(voice_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=set_volume)
    volume_slider.set(100); volume_slider.pack()

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

    refresh(); update_environment()
    root.mainloop()

# === 🔧 Agent Setup

def init_agents():
    global agents
    agents = [
        GlyphInterpreterAgent("Echo", {"mood": "analytical"}, 0.85),
        GlyphInterpreterAgent("Solus", {"mood": "aggressive"}, 0.6),
        GlyphInterpreterAgent("Orin", {"mood": "ritualist"}, 0.9),
        GlyphInterpreterAgent("Luma", {"mood": "healer"}, 0.75),
        GlyphInterpreterAgent("Nix", {"mood": "archivist"}, 0.8),
        GlyphTunerAgent("TunerX", trust_threshold=0.7)
    ]

# === 🚀 Main Launch Block

if __name__ == "__main__":
    init_db()
    init_tuning_db()
    init_memory_db()
    init_agents()
    announce_startup()
    threading.Thread(target=heartbeat, daemon=True).start()
    launch_ui()

