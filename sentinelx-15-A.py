# === 🔧 AutoLoader for Required Libraries
import sys, subprocess

REQUIRED_LIBS = [
    "psutil", "pyttsx3", "tkinter", "sqlite3", "websockets", "asyncio",
    "torch", "matplotlib", "cryptography", "pythonnet"
]

def autoload():
    for lib in REQUIRED_LIBS:
        try:
            if lib in ["sqlite3", "tkinter"]: continue  # built-in
            __import__(lib)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
autoload()

# === 📦 Imports
import json, hashlib, datetime, threading, platform, random, time, socket, atexit
from collections import Counter
import psutil, pyttsx3, sqlite3
import tkinter as tk
from tkinter import ttk
import websockets, asyncio
import torch
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet

# === 🌐 Globals
unit_state = {"temp_unit": "C"}
voice_state = {"enabled": True, "volume": 1.0}
DB_PATH = "glyph_store.db"
TUNING_DB = "tuning_glyphs.db"
MEMORY_DB = "agent_memory.db"
NODE_ID = socket.gethostname()
MESH_PORT = 9876
PEER_NODES = set()
agents = []
locked_fusions = set()
agent_keys = {"Echo": "alpha123", "Solus": "beta456"}
system_paused = False
voice_script = []

# === 🔉 Safe Voice Engine
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

def announce_startup():
    speak(f"SentinelX node {NODE_ID} online.")

# === 💾 Glyph Database Schema
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS glyphs")
    cur.execute('''CREATE TABLE glyphs (
        anchor_hash TEXT PRIMARY KEY,
        glyph_sequence TEXT,
        ritual_outcome TEXT,
        entropy_fingerprint TEXT,
        timestamp TEXT,
        system_state TEXT,
        trust_vector REAL,
        source_path TEXT,
        mutation_id TEXT,
        parent_id TEXT,
        fusion_id TEXT,
        origin_agent TEXT,
        source_node TEXT
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
    cur.execute('SELECT * FROM glyphs WHERE anchor_hash = ?', (anchor,))
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

# === 🔁 Mutation Generator
def mutate_glyph(base_glyph, agent_name):
    if not base_glyph: return [":fallback:entropy"], "", ""
    mutated = base_glyph[:]
    if random.random() < 0.5:
        mutated[random.randint(0, len(mutated)-1)] = ":entropy:flux"
    mutation_id = hashlib.sha256(("".join(mutated) + agent_name + str(datetime.datetime.utcnow())).encode()).hexdigest()
    parent_id = hashlib.sha256(("".join(base_glyph)).encode()).hexdigest()
    return mutated, mutation_id, parent_id

# === 🌐 Mesh Sync Node
class GlyphMeshNode:
    def __init__(self, port=MESH_PORT):
        self.peers = set()
        self.port = port

    async def broadcast_glyphs(self):
        while True:
            await asyncio.sleep(45)
            try:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("SELECT anchor_hash, glyph_sequence, mutation_id, parent_id, fusion_id FROM glyphs ORDER BY timestamp DESC LIMIT 5")
                rows = cur.fetchall()
                payload = json.dumps({
                    "node": NODE_ID,
                    "glyphs": [dict(
                        anchor_hash=r[0], glyph_sequence=r[1],
                        mutation_id=r[2], parent_id=r[3], fusion_id=r[4]
                    ) for r in rows]
                })
                for peer in self.peers:
                    try:
                        async with websockets.connect(peer) as ws:
                            await ws.send(payload)
                    except Exception as e:
                        print(f"🌐 Broadcast to {peer} failed: {e}")
            except Exception as e:
                print(f"🌐 Broadcast error: {e}")

    async def receive_glyphs(self, websocket, path):
        try:
            data = await websocket.recv()
            packet = json.loads(data)
            node_id = packet.get("node")
            glyphs = packet.get("glyphs", [])
            PEER_NODES.add(node_id)
            for g in glyphs:
                entry = {
                    "anchor_hash": g["anchor_hash"],
                    "glyph_sequence": json.loads(g["glyph_sequence"]),
                    "ritual_outcome": "mesh",
                    "entropy_fingerprint": json.dumps({"source": "mesh"}),
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "system_state": json.dumps({"source": "mesh"}),
                    "trust_vector": 0.5,
                    "source_path": "mesh",
                    "mutation_id": g.get("mutation_id", ""),
                    "parent_id": g.get("parent_id", ""),
                    "fusion_id": g.get("fusion_id", ""),
                    "origin_agent": "mesh",
                    "source_node": node_id
                }
                store_glyph(entry)
        except Exception as e:
            print(f"🌐 Mesh receive error: {e}")

    def start(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            server = websockets.serve(self.receive_glyphs, "0.0.0.0", self.port)
            loop.run_until_complete(server)
            loop.create_task(self.broadcast_glyphs())
            loop.run_forever()
        except Exception as e:
            print(f"🌐 Mesh server error: {e}")

# === 🤖 Cognitive Agent Classes

class GlyphInterpreterAgent:
    def __init__(self, name="Echo", style=None, trust_threshold=0.8):
        self.name = name
        self.style = style or {"mood": "analytical"}
        self.trust_threshold = trust_threshold

    def dream_glyph(self):
        traces = load_memory_trace(self.name, limit=50)
        good = [json.loads(row[0]) for row in traces if row[2] >= 0.7]
        return random.choice(good) + [f":mood:{self.style['mood']}"] if good else [":fallback:entropy"]

    def assess_entropy(self, fingerprint):
        try:
            f = json.loads(fingerprint) if isinstance(fingerprint, str) else fingerprint
            return "unstable" in f.get("drivers", "") or f.get("memory") == "fragmented"
        except: return False

    def forecast_score(self, glyph, entropy):
        try:
            traces = load_memory_trace(self.name, limit=30)
            matches = [json.loads(row[0]) for row in traces if glyph[:2] == json.loads(row[0])[:2]]
            score = sum([row[2] for row in traces if json.loads(row[0]) == glyph]) / len(matches) if matches else 0.5
            return round(score, 2)
        except: return 0.5

    def decide_action(self, glyph_entry):
        anomaly = self.assess_entropy(glyph_entry["entropy_fingerprint"])
        trust = glyph_entry.get("trust_vector", 0.5)
        forecast = self.forecast_score(glyph_entry["glyph_sequence"], glyph_entry["entropy_fingerprint"])
        decision = "observe"
        if anomaly and trust >= self.trust_threshold:
            decision = "rollback"
        log_memory_trace(self, glyph_entry, decision, forecast)
        return decision

    def act(self, glyph_entry):
        try:
            decision = self.decide_action(glyph_entry)
            if decision == "rollback":
                rollback(glyph_entry["anchor_hash"])
                speak(f"{self.name} rolled back unstable glyph.")
            else:
                print(f"[{self.name}] observing glyph.")
        except Exception as e:
            print(f"⚠️ Agent error ({self.name}): {e}")

class GlyphTunerAgent(GlyphInterpreterAgent):
    def __init__(self, name="TunerX", trust_threshold=0.75):
        super().__init__(name, {"mood": "optimizer"}, trust_threshold)
        self.last_entropy = {}

    def tune_cpu(self):
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            if proc.info['cpu_percent'] > 50:
                try: psutil.Process(proc.info['pid']).nice(psutil.IDLE_PRIORITY_CLASS)
                except: pass

    def score_feedback(self, previous, current):
        score = 0.5
        if previous.get("memory") == "fragmented" and current.get("memory") == "stable": score += 0.3
        if "unstable" not in current.get("drivers", ""): score += 0.2
        return round(score, 2)

    def act(self, glyph_entry):
        try:
            entropy = json.loads(glyph_entry["entropy_fingerprint"])
            timestamp = glyph_entry.get("timestamp", datetime.datetime.utcnow().isoformat())
            self.tune_cpu()
            score = self.score_feedback(self.last_entropy, entropy)
            tuning_id = hashlib.sha256((timestamp + self.name + "tune").encode()).hexdigest()
            log_tuning_glyph({
                "tuning_id": tuning_id, "timestamp": timestamp, "method": "affinity",
                "target": "cpu", "outcome_snapshot": entropy,
                "feedback_score": score, "agent_name": self.name
            })
            log_memory_trace(self, glyph_entry, "tune", score)
            speak(f"{self.name} tuned system. Score: {score}")
            self.last_entropy = entropy
            super().act(glyph_entry)
        except Exception as e:
            print(f"⚠️ TunerX error: {e}")

class MeshAgent(GlyphInterpreterAgent):
    def __init__(self, name="MeshAgent"):
        super().__init__(name, {"mood": "sync"}, trust_threshold=0.65)

# === 🌡️ Entropy Samplers

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
                val = round((e.current*9/5)+32, 1) if unit_state["temp_unit"]=="F" else round(e.current, 1)
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

# === 🔄 Rollback Logic
def rollback(anchor_hash):
    g = get_glyph(anchor_hash)
    if g:
        print(f"↺ Glyph: {g['glyph_sequence']}\n🧬 Snapshot: {g['system_state']}")
    else:
        speak("Rollback failed. Glyph not found.")

# === 🗳️ Glyph Council
class GlyphCouncil:
    def convene(self, agents, glyph_entry):
        votes = [agent.decide_action(glyph_entry) for agent in agents]
        count = Counter(votes)
        top = count.most_common(1)[0]
        print(f"🗳️ Council voted: {top[0]} ({top[1]} votes)")
        return top[0]

# === 🫀 Heartbeat Daemon
def generate_hash(entropy, glyphs):
    raw = entropy + datetime.datetime.utcnow().isoformat() + ''.join(glyphs)
    return hashlib.sha256(raw.encode()).hexdigest()

def heartbeat(interval=30):
    while True:
        try:
            if system_paused:
                time.sleep(10)
                continue
            now = datetime.datetime.utcnow().isoformat()
            entropy = get_entropy()
            temps = get_all_temps()
            disk = sample_disk()
            glyphs = ["⟊:driver:patch", "⇌:kernel:scan", "⇄:affinity:rebalance"]
            anchor = generate_hash(json.dumps(entropy), glyphs)
            entry = {
                "anchor_hash": anchor,
                "glyph_sequence": glyphs,
                "ritual_outcome": "heartbeat",
                "entropy_fingerprint": json.dumps(entropy),
                "timestamp": now,
                "system_state": {
                    "os": platform.system(), "cpu": entropy.get("cpu"),
                    "memory": entropy.get("memory"), "drivers": entropy.get("drivers"),
                    "temps": temps, "disk": disk
                },
                "trust_vector": 0.91,
                "source_path": "daemon"
            }
            store_glyph(entry)
            council = GlyphCouncil(); council.convene(agents, entry)
            for agent in agents: agent.act(entry)
            time.sleep(interval)
        except Exception as e:
            print(f"⚠️ Heartbeat error: {e}")
            time.sleep(10)

# === 🗂️ Memory + Tuning DBs
def init_tuning_db():
    conn = sqlite3.connect(TUNING_DB)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS tuning_glyphs (
        tuning_id TEXT PRIMARY KEY, timestamp TEXT, method TEXT,
        target TEXT, outcome_snapshot TEXT, feedback_score REAL,
        agent_name TEXT
    )'''); conn.commit(); conn.close()

def init_memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS agent_memory (
        trace_id TEXT PRIMARY KEY, agent_name TEXT, anchor_hash TEXT,
        decision TEXT, feedback_score REAL, entropy_snapshot TEXT,
        glyph_sequence TEXT, timestamp TEXT
    )'''); conn.commit(); conn.close()

def log_tuning_glyph(entry):
    conn = sqlite3.connect(TUNING_DB)
    cur = conn.cursor()
    cur.execute('''INSERT OR REPLACE INTO tuning_glyphs VALUES (?, ?, ?, ?, ?, ?, ?)''', (
        entry['tuning_id'], entry['timestamp'], entry['method'], entry['target'],
        json.dumps(entry['outcome_snapshot']), entry['feedback_score'], entry['agent_name']
    )); conn.commit(); conn.close()

def log_memory_trace(agent, glyph_entry, decision, feedback_score):
    trace_id = hashlib.sha256((glyph_entry['anchor_hash'] + agent.name).encode()).hexdigest()
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute('''INSERT OR REPLACE INTO agent_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
        trace_id, agent.name, glyph_entry['anchor_hash'], decision,
        feedback_score, glyph_entry["entropy_fingerprint"],
        json.dumps(glyph_entry["glyph_sequence"]), glyph_entry["timestamp"]
    )); conn.commit(); conn.close()

def load_memory_trace(agent_name, limit=20):
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute("SELECT glyph_sequence, decision, feedback_score FROM agent_memory WHERE agent_name=? ORDER BY timestamp DESC LIMIT ?", (agent_name, limit))
    rows = cur.fetchall(); conn.close()
    return rows

# === 🧠 Agent Initialization
def init_agents():
    global agents
    agents = [
        GlyphInterpreterAgent("Echo", {"mood": "analytical"}, 0.85),
        GlyphInterpreterAgent("Solus", {"mood": "aggressive"}, 0.6),
        GlyphInterpreterAgent("Orin", {"mood": "ritualist"}, 0.9),
        GlyphInterpreterAgent("Luma", {"mood": "healer"}, 0.75),
        GlyphInterpreterAgent("Nix", {"mood": "archivist"}, 0.8),
        GlyphTunerAgent("TunerX", trust_threshold=0.7),
        MeshAgent("MeshAgent")
    ]

# === 🖥️ Dashboard UI
def launch_ui():
    root = tk.Tk()
    root.title("SentinelX Dashboard")
    root.geometry("1000x800")

    voice_frame = ttk.Frame(root)
    voice_label = ttk.Label(voice_frame, text="🔊 Voice Engine")
    voice_toggle = ttk.Button(voice_frame, text="Turn OFF" if voice_state["enabled"] else "Turn ON")
    def toggle_voice():
        voice_state["enabled"] = not voice_state["enabled"]
        voice_toggle.config(text="Turn OFF" if voice_state["enabled"] else "Turn ON")
    voice_toggle.config(command=toggle_voice)
    volume_slider = ttk.Scale(voice_frame, from_=0.0, to=1.0, value=voice_state["volume"], orient="horizontal", length=200)
    def set_volume(val): voice_state["volume"] = float(val)
    volume_slider.config(command=set_volume)
    voice_label.pack(side="left", padx=5)
    voice_toggle.pack(side="left", padx=5)
    ttk.Label(voice_frame, text="Volume").pack(side="left", padx=5)
    volume_slider.pack(side="left", padx=5)
    voice_frame.pack(pady=10)

    tabs = ttk.Notebook(root)

    # Trace Tab
    trace_tab = ttk.Frame(tabs)
    tabs.add(trace_tab, text="⛒ Trace")
    tree = ttk.Treeview(trace_tab, columns=("anchor", "timestamp"), show="headings")
    tree.heading("anchor", text="Anchor"); tree.heading("timestamp", text="Timestamp")
    tree.pack(pady=5)
    trace_box = tk.Text(trace_tab, height=10, width=120); trace_box.pack()

    def refresh_trace():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT anchor_hash, timestamp FROM glyphs ORDER BY timestamp DESC LIMIT 20")
        rows = cur.fetchall(); tree.delete(*tree.get_children())
        for r in rows:
            tree.insert("", "end", values=(r[0][:12]+"...", r[1]))
        conn.close(); trace_box.delete("1.0", tk.END)
        for agent in agents:
            traces = load_memory_trace(agent.name, limit=6)
            trace_box.insert(tk.END, f"\n🧠 {agent.name} Trace:\n")
            for row in traces:
                glyphs = " ".join(json.loads(row[0]))
                trace_box.insert(tk.END, f" ⛒ {glyphs} | Decision: {row[1]} | Score: {row[2]}\n")

    def rollback_selected():
        selected = tree.selection()
        if selected:
            anchor = tree.item(selected[0])['values'][0].replace("...", "")
            rollback(anchor)

    tk.Button(trace_tab, text="🔃 Refresh Trace", command=refresh_trace).pack()
    tk.Button(trace_tab, text="↺ Rollback Selected", command=rollback_selected).pack()

    # Mutation Tab
    mutation_tab = ttk.Frame(tabs)
    tabs.add(mutation_tab, text="🧬 Mutations")
    mutation_box = tk.Text(mutation_tab, height=30, width=120); mutation_box.pack()

    def show_mutations():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT anchor_hash, mutation_id, parent_id, origin_agent FROM glyphs WHERE mutation_id != '' ORDER BY timestamp DESC LIMIT 30")
        rows = cur.fetchall(); mutation_box.delete("1.0", tk.END)
        for r in rows:
            mutation_box.insert(tk.END, f"🔁 Glyph: {r[0][:12]}... | Mutation: {r[1][:8]} | Parent: {r[2][:8]} | Agent: {r[3]}\n")
        conn.close()

    tk.Button(mutation_tab, text="🔄 Load Mutations", command=show_mutations).pack()

    # Fusion Tab
    fusion_tab = ttk.Frame(tabs)
    tabs.add(fusion_tab, text="💠 Fusions")
    fusion_box = tk.Text(fusion_tab, height=30, width=120); fusion_box.pack()

    def show_fusions():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT anchor_hash, fusion_id, origin_agent, source_node FROM glyphs WHERE fusion_id != '' ORDER BY timestamp DESC LIMIT 30")
        rows = cur.fetchall(); fusion_box.delete("1.0", tk.END)
        for r in rows:
            fusion_box.insert(tk.END, f"💠 Glyph: {r[0][:12]}... | Fusion: {r[1][:8]} | Agent: {r[2]} | Node: {r[3]}\n")
        conn.close()

    tk.Button(fusion_tab, text="🌀 Load Fusions", command=show_fusions).pack()

    # Mesh Tab
    mesh_tab = ttk.Frame(tabs)
    tabs.add(mesh_tab, text="🌐 Mesh")
    mesh_box = tk.Text(mesh_tab, height=30, width=120); mesh_box.pack()

    def show_mesh():
        mesh_box.delete("1.0", tk.END)
        mesh_box.insert(tk.END, f"🌐 Local Node: {NODE_ID}\n")
        for peer in PEER_NODES:
            mesh_box.insert(tk.END, f"🤝 Peer: {peer}\n")

    tk.Button(mesh_tab, text="🔎 Refresh Peers", command=show_mesh).pack()

    tabs.pack(expand=1, fill="both")
    refresh_trace(); root.mainloop()

# === 🚀 Boot Sequence
if __name__ == "__main__":
    print("🧿 Starting SentinelX…")
    init_db()
    init_tuning_db()
    init_memory_db()
    init_agents()
    announce_startup()
    threading.Thread(target=heartbeat, daemon=True).start()
    threading.Thread(target=GlyphMeshNode().start, daemon=True).start()
    launch_ui()

