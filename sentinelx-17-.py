# === Autoloader
import sys, subprocess
REQUIRED_LIBS = [
    "psutil", "pyttsx3", "tkinter", "sqlite3", "websockets", "asyncio",
    "matplotlib", "cryptography", "fpdf"
]
def autoload():
    for lib in REQUIRED_LIBS:
        try:
            if lib in ["sqlite3", "tkinter"]: continue
            __import__(lib)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
autoload()

# === Imports
import json, hashlib, datetime, threading, platform, random, time, socket, math
from collections import Counter
import psutil, pyttsx3, sqlite3
import tkinter as tk
from tkinter import ttk
import websockets, asyncio
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
from fpdf import FPDF

# === Globals
unit_state = {"temp_unit": "C"}
voice_state = {"enabled": True, "volume": 1.0}
DB_PATH = "glyph_store.db"
TUNING_DB = "tuning_glyphs.db"
MEMORY_DB = "agent_memory.db"
NODE_ID = socket.gethostname()
MESH_PORT = 9876
PEER_NODES = set()
agents = []
voice_script = []

# === Voice Engine
voice_lock = threading.Lock()
def speak(text):
    if not voice_state["enabled"]: return
    try:
        with voice_lock:
            engine = pyttsx3.init('sapi5')
            volume = max(voice_state["volume"], 0.1)
            engine.setProperty('volume', volume)
            engine.say(text); engine.runAndWait()
    except Exception as e:
        print(f"🔇 Voice error: {e}")

def queue_voice(text): voice_script.append(text)
def flush_voice_script():
    for line in voice_script: speak(line)
    voice_script.clear()

def announce_startup(): speak(f"SentinelX node {NODE_ID} online.")

# === Entropy Samplers
def get_entropy():
    try:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        return {
            "cpu": f"{cpu}%",
            "memory": "fragmented" if mem.percent > 70 else "stable",
            "drivers": "unstable:usb.sys"
        }
    except:
        return {"cpu": "error", "memory": "error", "drivers": "error"}

def get_all_temps():
    temps = []
    try:
        sensors = psutil.sensors_temperatures()
        for name, entries in sensors.items():
            for e in entries:
                val = round((e.current * 9/5) + 32, 1) if unit_state["temp_unit"] == "F" else round(e.current, 1)
                temps.append(f"{name}: {val}°{unit_state['temp_unit']}")
    except:
        temps.append("Temp error")
    return temps

def sample_disk():
    try:
        d = psutil.disk_usage('/')
        return {
            "total": f"{d.total//(1024**3)}GB",
            "used": f"{d.used//(1024**3)}GB",
            "free": f"{d.free//(1024**3)}GB",
            "percent": f"{d.percent}%"
        }
    except:
        return {"total": "N/A", "used": "N/A", "free": "N/A", "percent": "N/A"}

# === Glyph DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS glyphs")
    cur.execute('''CREATE TABLE glyphs (
        anchor_hash TEXT PRIMARY KEY, glyph_sequence TEXT, ritual_outcome TEXT,
        entropy_fingerprint TEXT, timestamp TEXT, system_state TEXT,
        trust_vector REAL, source_path TEXT, mutation_id TEXT,
        parent_id TEXT, fusion_id TEXT, origin_agent TEXT, source_node TEXT
    )''')
    conn.commit(); conn.close()

def store_glyph(entry):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''INSERT OR REPLACE INTO glyphs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        entry['anchor_hash'], json.dumps(entry['glyph_sequence']), entry['ritual_outcome'],
        entry['entropy_fingerprint'], entry['timestamp'], json.dumps(entry['system_state']),
        entry['trust_vector'], entry['source_path'], entry.get('mutation_id', ''),
        entry.get('parent_id', ''), entry.get('fusion_id', ''),
        entry.get('origin_agent', ''), entry.get('source_node', NODE_ID)
    ))
    conn.commit(); conn.close()

def get_glyph(anchor):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM glyphs WHERE anchor_hash=?", (anchor,))
    row = cur.fetchone(); conn.close()
    if row:
        return {
            "anchor_hash": row[0], "glyph_sequence": json.loads(row[1]),
            "ritual_outcome": row[2], "entropy_fingerprint": row[3],
            "timestamp": row[4], "system_state": json.loads(row[5]),
            "trust_vector": row[6], "source_path": row[7],
            "mutation_id": row[8], "parent_id": row[9],
            "fusion_id": row[10], "origin_agent": row[11], "source_node": row[12]
        }
    return None

# === Agent Memory
def log_memory_trace(agent, glyph_entry, decision, feedback_score):
    trace_id = hashlib.sha256((glyph_entry['anchor_hash'] + agent.name).encode()).hexdigest()
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS agent_memory (
        trace_id TEXT PRIMARY KEY, agent_name TEXT, anchor_hash TEXT,
        decision TEXT, feedback_score REAL, entropy_snapshot TEXT,
        glyph_sequence TEXT, timestamp TEXT
    )''')
    cur.execute('''INSERT OR REPLACE INTO agent_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
        trace_id, agent.name, glyph_entry['anchor_hash'], decision,
        feedback_score, glyph_entry["entropy_fingerprint"],
        json.dumps(glyph_entry["glyph_sequence"]), glyph_entry["timestamp"]
    ))
    conn.commit(); conn.close()

def remember_glyph(agent, glyph_entry):
    trace_id = hashlib.sha256((glyph_entry['anchor_hash'] + agent.name).encode()).hexdigest()
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute('''INSERT OR IGNORE INTO agent_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
        trace_id, agent.name, glyph_entry["anchor_hash"],
        "favor", 0.99, glyph_entry["entropy_fingerprint"],
        json.dumps(glyph_entry["glyph_sequence"]), glyph_entry["timestamp"]
    ))
    conn.commit(); conn.close()

# === Agents
class GlyphInterpreterAgent:
    def __init__(self, name, style, trust_threshold):
        self.name = name
        self.style = style
        self.trust_threshold = trust_threshold

    def decide_action(self, glyph_entry):
        entropy = json.loads(glyph_entry["entropy_fingerprint"])
        unstable = "unstable" in entropy.get("drivers", "")
        fragmented = entropy.get("memory") == "fragmented"
        trust = glyph_entry.get("trust_vector", 0.0)
        if unstable and fragmented and trust >= self.trust_threshold:
            return "rollback"
        return "observe"

    def act(self, glyph_entry):
        decision = self.decide_action(glyph_entry)
        if decision == "rollback":
            speak(f"{self.name} initiates rollback.")
        else:
            speak(f"{self.name} observes glyph.")
        log_memory_trace(self, glyph_entry, decision, trust_score := random.uniform(0.5, 1.0))
        if trust_score > self.trust_threshold:
            remember_glyph(self, glyph_entry)

# === Heartbeat Loop
def heartbeat(interval=30):
    while True:
        try:
            now = datetime.datetime.utcnow().isoformat()
            entropy = get_entropy()
            temps = get_all_temps()
            disk = sample_disk()
            variant = random.choice([":scan:kernel", ":patch:driver", ":flux:entropy"])
            anchor = hashlib.sha256((json.dumps(entropy) + now + variant).encode()).hexdigest()
            glyph_entry = {
                "anchor_hash": anchor,
                "glyph_sequence": [variant],
                "ritual_outcome": "heartbeat",
                "entropy_fingerprint": json.dumps(entropy),
                "timestamp": now,
                "system_state": {
                    "os": platform.system(), "cpu": entropy.get("cpu"),
                    "memory": entropy.get("memory"), "drivers": entropy.get("drivers"),
                    "temps": temps, "disk": disk
                },
                "trust_vector": 0.91, "source_path": "daemon",
                "mutation_id": "", "parent_id": "", "fusion_id": "",
                "origin_agent": "SentinelX", "source_node": NODE_ID
            }
            store_glyph(glyph_entry)
            for agent in agents: agent.act(glyph_entry)
            time.sleep(interval)
        except Exception as e:
            print(f"⚠️ Heartbeat error: {e}")
            time.sleep(10)

# === Agent Initialization
def init_agents():
    global agents
    agents = [
        GlyphInterpreterAgent("Echo", {"mood": "analytical"}, 0.85),
        GlyphInterpreterAgent("Luma", {"mood": "healer"}, 0.75),
        GlyphInterpreterAgent("Nix", {"mood": "archivist"}, 0.8)
    ]

# === 🖥️ UI Dashboard — Full Tab Set
def launch_ui():
    root = tk.Tk()
    root.title("SentinelX Dashboard")
    tabs = ttk.Notebook(root)
    trace_tab = ttk.Frame(tabs)
    fusion_tab = ttk.Frame(tabs)
    tuning_tab = ttk.Frame(tabs)
    tabs.add(trace_tab, text="⛒ Trace")
    tabs.add(fusion_tab, text="💠 Fusions")
    tabs.add(tuning_tab, text="🧪 Tuning")
    tabs.pack(expand=1, fill="both")

    # === ⛒ Trace Tab
    trace_box = tk.Text(trace_tab, height=25, width=100); trace_box.pack()
    def show_traces():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT anchor_hash, glyph_sequence, timestamp FROM glyphs ORDER BY timestamp DESC LIMIT 30")
        rows = cur.fetchall(); trace_box.delete("1.0", tk.END)
        for r in rows:
            trace_box.insert(tk.END, f"{r[2]} ⛒ {r[0][:8]} → {' '.join(json.loads(r[1]))}\n")
        conn.close()
    tk.Button(trace_tab, text="🔄 Load Traces", command=show_traces).pack()

    # === 💠 Fusion Tab
    fusion_box = tk.Text(fusion_tab, height=25, width=100); fusion_box.pack()
    def show_fusions():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT anchor_hash, fusion_id, origin_agent FROM glyphs WHERE fusion_id != '' ORDER BY timestamp DESC LIMIT 20")
        rows = cur.fetchall(); fusion_box.delete("1.0", tk.END)
        for r in rows:
            fusion_box.insert(tk.END, f"💠 {r[0][:8]} | FusionID: {r[1][:8]} | Agent: {r[2]}\n")
        conn.close()
    tk.Button(fusion_tab, text="🌀 Load Fusions", command=show_fusions).pack()

    # === 🛠️ Ritual Compiler
    def compile_custom_ritual():
        ritual_box = tk.Toplevel(); ritual_box.title("🛠️ Ritual Compiler")
        editor = tk.Text(ritual_box, height=10, width=80); editor.pack()

        def compile_and_store():
            raw = editor.get("1.0", tk.END).strip().split()
            anchor = hashlib.sha256("".join(raw).encode()).hexdigest()
            entry = {
                "anchor_hash": anchor, "glyph_sequence": raw,
                "ritual_outcome": "manual", "entropy_fingerprint": json.dumps({"manual": True}),
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "system_state": {}, "trust_vector": 1.0,
                "source_path": "manual", "mutation_id": "", "parent_id": "",
                "fusion_id": "", "origin_agent": "User", "source_node": NODE_ID
            }
            store_glyph(entry); ritual_box.destroy()
        tk.Button(ritual_box, text="🔮 Compile Ritual", command=compile_and_store).pack()
    tk.Button(fusion_tab, text="🛠️ Compile Ritual", command=compile_custom_ritual).pack()

    # === 🔍 Trace Search
    def search_trace():
        search_window = tk.Toplevel(); search_window.title("🔍 Trace Search")
        label = tk.Label(search_window, text="Enter token:"); label.pack()
        query_entry = tk.Entry(search_window); query_entry.pack()
        results_box = tk.Text(search_window, height=15, width=90); results_box.pack()

        def run_search():
            token = query_entry.get().strip()
            results_box.delete("1.0", tk.END)
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT anchor_hash, glyph_sequence FROM glyphs")
            for anchor, seq in cur.fetchall():
                if token in json.loads(seq):
                    results_box.insert(tk.END, f"🧭 {anchor[:8]} → {token} found\n")
            conn.close()
        tk.Button(search_window, text="🔎 Search", command=run_search).pack()
    tk.Button(fusion_tab, text="🔍 Search Trace", command=search_trace).pack()

    # === 🧠 Mood Filters
    def filter_fusions_by_mood(mood):
        fusion_box.delete("1.0", tk.END)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT anchor_hash, fusion_id, origin_agent FROM glyphs WHERE fusion_id != ''")
        rows = cur.fetchall(); conn.close()
        for r in rows:
            if any(agent.name == r[2] and agent.style.get("mood") == mood for agent in agents):
                fusion_box.insert(tk.END, f"💠 {r[0][:8]} | Agent: {r[2]} | FusionID: {r[1][:8]}\n")
    tk.Button(fusion_tab, text="🧠 Filter Ritualists", command=lambda: filter_fusions_by_mood("ritualist")).pack()
    tk.Button(fusion_tab, text="💊 Filter Healers", command=lambda: filter_fusions_by_mood("healer")).pack()

    # === 📤 Export PDF
    def export_selected_fusion():
        selected = fusion_box.get("1.0", "2.0").strip()
        if "FusionID:" not in selected: return
        fusion_id = selected.split("FusionID:")[1].strip()
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM glyphs WHERE fusion_id=?", (fusion_id,))
        row = cur.fetchone(); conn.close()
        if row:
            glyph = get_glyph(row[0])
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt="Glyph Sequence:\n" + " ".join(glyph["glyph_sequence"]))
            pdf.multi_cell(0, 10, txt="Entropy:\n" + glyph["entropy_fingerprint"])
            pdf.multi_cell(0, 10, txt="System State:\n" + json.dumps(glyph["system_state"], indent=2))
            pdf.output(f"{glyph['anchor_hash'][:8]}_ritual.pdf")
    tk.Button(fusion_tab, text="📤 Export PDF", command=export_selected_fusion).pack()

    # === 🔊 Voice Controls
    def update_volume(val):
        try: voice_state["volume"] = max(min(int(val)/100, 1.0), 0.1)
        except: pass
    def toggle_voice():
        voice_state["enabled"] = not voice_state["enabled"]
        state = "ON" if voice_state["enabled"] else "OFF"
        speak(f"Voice is now {state}.")

    volume_frame = tk.Frame(fusion_tab); volume_frame.pack(pady=10)
    tk.Label(volume_frame, text="🔊 Volume Control").pack()
    volume_slider = tk.Scale(volume_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                             command=update_volume, length=200)
    volume_slider.set(int(voice_state["volume"] * 100)); volume_slider.pack()
    tk.Button(volume_frame, text="🎙️ Toggle Voice", command=toggle_voice).pack()

    root.mainloop()

# === 🗂️ Database Initialization
def init_tuning_db():
    conn = sqlite3.connect(TUNING_DB)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS tuning_glyphs (
        tuning_id TEXT PRIMARY KEY, timestamp TEXT, method TEXT,
        target TEXT, outcome_snapshot TEXT, feedback_score REAL,
        agent_name TEXT
    )''')
    conn.commit(); conn.close()

def init_memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS agent_memory (
        trace_id TEXT PRIMARY KEY, agent_name TEXT, anchor_hash TEXT,
        decision TEXT, feedback_score REAL, entropy_snapshot TEXT,
        glyph_sequence TEXT, timestamp TEXT
    )''')
    conn.commit(); conn.close()

# === 🧠 Memory Logging (Extra utility)
def log_code_event(agent, event):
    with open("sentinelx_code_log.txt", "a") as f:
        stamp = datetime.datetime.utcnow().isoformat()
        f.write(f"[{stamp}] {agent.name}: {event}\n")

def log_fusion_decision(agent, fusion_id, stance):
    trace_id = hashlib.sha256((fusion_id + agent.name).encode()).hexdigest()
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute('''INSERT OR IGNORE INTO agent_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
        trace_id, agent.name, fusion_id, stance, 0.0, "{}", "[]", datetime.datetime.utcnow().isoformat()
    ))
    conn.commit(); conn.close()

def load_memory_trace(agent_name, limit=50):
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute("SELECT glyph_sequence, decision, feedback_score FROM agent_memory WHERE agent_name=? ORDER BY timestamp DESC LIMIT ?", (agent_name, limit))
    rows = cur.fetchall(); conn.close()
    return rows

# === 🚀 SentinelX Launch Sequence
def start_sentinelx():
    print("🧿 Starting SentinelX…")
    init_db()
    init_memory_db()
    init_tuning_db()
    init_agents()
    announce_startup()
    threading.Thread(target=heartbeat, daemon=True).start()
    launch_ui()

if __name__ == "__main__":
    start_sentinelx()

