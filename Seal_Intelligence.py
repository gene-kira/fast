# === 🛠️ AUTOLOADER FOR REQUIRED LIBRARIES ===
import sys, subprocess

def ensure_libs():
    import importlib
    required = [
        'sympy', 'numpy', 'matplotlib', 'flask',
        'psutil', 'tkinter', 'pyttsx3', 'keyboard', 'hashlib'
    ]
    for lib in required:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"🔧 Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

ensure_libs()

# === 🔗 IMPORTS ===
import sympy as sp, numpy as np, random, socket, threading, json, math, datetime, time, os
from flask import Flask, jsonify, request
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
import pyttsx3
import psutil
import keyboard

# === 🎙️ VOICE ENGINE SETUP ===
engine = pyttsx3.init()
voice_alert_active = True

# === 🌐 GLOBAL CONFIG ===
MEMORY_FILE = "glyph_memory.json"
VERDICT_LOG = "verdict_log.json"
COMMON_PORTS = [45201, 45202, 45203]
local_system_id = "Seer_453"
glyph_memory = {}
agent_registry = {}
remote_systems = {}
anomaly_nodes = {}
scan_interrupt = False

# === 🧬 SYMBOLIC GLYPH ENGINE SETUP ===
t, x, y, z, e, V = sp.symbols('t x y z e V')
pool = [t, x, y, z, e, V]
ops = [sp.sin, sp.cos, sp.exp, sp.log, sp.sqrt]
formula = sp.sin(t) + e * sp.exp(-x**2 - y**2 - z**2)
population = [formula]
memory = []

def score(expr):
    try:
        return (
            len(str(expr)) * 0.01 +
            expr.count(sp.sin) +
            expr.count(sp.exp) +
            sum(str(expr).count(g) for g in ['ψ','⟁','⧖','⚛','∞','Σ','χ','Æ','Ξ','Λ','Ω','Φ','⟲','🜄','🜁','Θ','Δ']) -
            sp.count_ops(expr) * 0.3
        )
    except:
        return 0

def mutate(expr):
    try:
        m = random.choice(["op", "symbol", "scale", "nest"])
        if m == "op": return expr + random.choice(ops)(random.choice(pool))
        elif m == "symbol": return expr.subs(random.choice(pool), random.choice(pool))
        elif m == "scale": return expr * random.uniform(0.5, 1.5)
        elif m == "nest": return random.choice(ops)(expr)
    except:
        return expr

def mesh_nodes(seed, count=4):
    return [sp.Function("Λ")(random.choice(pool)) + mutate(seed) * sp.sin(t * e) for _ in range(count)]

# === ⚛️ GLYPH EVOLUTION ENGINE ===
for gen in range(880):
    if gen < 40:
        population = sorted([mutate(f) for f in population for _ in range(4)], key=score, reverse=True)[:3]
    else:
        if gen % 3 == 0: population.append(sp.Function("⚛")(x, y, z, t) + sp.cos(V * e))
        if gen % 7 == 0: population.append(1 / (population[0] + 1e-3))
        if gen % 17 == 0: population += mesh_nodes(population[0], 3)
        if gen % 41 == 0: population = [f + sp.Function("χ")(f) for f in population]
        if gen % 59 == 0: population += [sp.Function("Ω")(f) for f in memory[-3:]]
        if gen % 73 == 0: population.append(sp.Function("Ξ")(sp.Function("χ")(population[0])))
        if gen % 79 == 0: population.append(sp.Function("Δ")(sp.sqrt(abs(population[0] + 1e-2))))
        if random.random() > 0.6:
            population.append(sp.sqrt(abs(random.choice(population) + 1e-2)))
        scored = sorted([(f, score(f)) for f in population], key=lambda x: -x[1])[:6]
        population = [f for f, _ in scored]
    if gen % 30 == 0:
        print(f"🔹 Gen {gen}: {str(population[0])[:80]}...")
    memory.append(population[0])

final_operator = population[0]
print("\n🔺 Final Symbolic Operator:\n", final_operator)

# === 📁 EXPORTS ===
with open("density_core.sym", "w") as f:
    f.write(str(final_operator))

with open("final_operator_glsl.frag", "w") as f:
    f.write(f"float glyphField(vec3 pos) {{ return {str(final_operator)}; }}")

X, Y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
Z = np.sin(X + Y) * np.exp(-X**2 - Y**2)
np.save("resonance_matrix.npy", Z)

# === 🌐 GLYPHIC SWARM SOCKET SERVER ===
def handle_node(client):
    try:
        client.sendall(str(final_operator).encode())
    except:
        pass
    client.close()

def launch_swarm():
    s = socket.socket()
    s.bind(('0.0.0.0', 7654))
    s.listen()
    print("🌐 Swarm socket open on port 7654")
    while True:
        c, _ = s.accept()
        threading.Thread(target=handle_node, args=(c,)).start()

threading.Thread(target=launch_swarm, daemon=True).start()

# === 🌐 FLASK SYMBOLIC API ===
app = Flask(__name__)

@app.route("/get_operator", methods=["GET"])
def get_operator():
    return jsonify({"operator": str(final_operator)})

@app.route("/mutate", methods=["POST"])
def mutate_now():
    mutated = mutate(final_operator)
    return jsonify({"mutated": str(mutated)})

@app.route("/trigger_unweaving", methods=["POST"])
def reinit():
    population.append(sp.sqrt(abs(final_operator + 1e-2)))
    return jsonify({"status": "New glyph injected into population."})

threading.Thread(target=lambda: app.run(port=8080, debug=False), daemon=True).start()

# === 🎨 GLYPHIC VISUALIZER ===
def visualize():
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    Xs = np.linspace(-1,1,16)
    X, Y, Z = np.meshgrid(Xs, Xs, Xs)
    vals = np.sin(X + Y + Z)
    ax.scatter(X, Y, Z, c=vals.ravel(), cmap='plasma', alpha=0.4, s=10)
    ax.set_title("🌀 Glyphic Convergence")
    plt.show()

visualize()

# === 📜 MEMORY INITIALIZATION ===
def init_memory():
    global glyph_memory
    try:
        with open(MEMORY_FILE) as f:
            glyph_memory = json.load(f)
    except:
        glyph_memory = {
            "graph": {}, "known_nodes": {},
            "audit": [], "threats": [], "core_symbol": str(final_operator)
        }
    print("🔹 Memory initialized with", len(glyph_memory.get("graph", {})), "glyph nodes")

init_memory()

# === 📡 GLYPH BROADCAST / DISCOVERY ===
def GlyphBroadcaster():
    def loop():
        while True:
            try:
                msg = f"GLYPH_BEACON::{local_system_id}::{datetime.datetime.now().isoformat()}"
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                s.sendto(msg.encode(), ("255.255.255.255", 45201))
                time.sleep(4)
            except:
                continue
    threading.Thread(target=loop, daemon=True).start()

def GlyphListener():
    def listen():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try: s.bind(("", 45201))
        except: return
        while True:
            try:
                data, addr = s.recvfrom(1024)
                msg = data.decode()
                if msg.startswith("GLYPH_BEACON::"):
                    parts = msg.split("::")
                    remote_id = parts[1]
                    ts = parts[2]
                    remote_systems[remote_id] = {"time": ts, "ip": addr[0]}
            except:
                continue
    threading.Thread(target=listen, daemon=True).start()

# === 📜 VERDICT CASTING ENGINE ===
def cast_verdict(target_id, verdict, linked_to=None):
    if not linked_to: linked_to = []
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "system_id": target_id,
        "verdict": verdict,
        "linked_to": linked_to,
        "agent": local_system_id
    }
    try:
        with open(VERDICT_LOG, "r") as f: log = json.load(f)
    except:
        log = []
    log.append(entry)
    with open(VERDICT_LOG, "w") as f: json.dump(log, f, indent=2)
    glyph_memory.setdefault("graph", {}).setdefault(target_id, []).append(entry)
    agent_registry.setdefault(target_id, {})["trust"] = agent_registry.get(target_id, {}).get("trust", 0.5)
    print(f"🧿 Verdict cast on {target_id}: {verdict}")

# === 🧠 ENTROPY SCORE CALCULATOR ===
def compute_entropy(gid):
    verdicts = glyph_memory.get("graph", {}).get(gid, [])
    flips = sum(1 for i in range(1, len(verdicts)) if verdicts[i]["verdict"] != verdicts[i-1]["verdict"])
    return round(min(1.0, flips / (len(verdicts) + 1)), 3)

# === 🎭 PERSONALITY ASSIGNMENT ===
def assign_personality(gid):
    entropy = compute_entropy(gid)
    if entropy > 0.6:
        return {"tone": "chaotic", "emotion": "volatile", "style": "erratic", "color": "#FF4444"}
    elif entropy < 0.2:
        return {"tone": "stoic", "emotion": "calm", "style": "patient", "color": "#33FF99"}
    else:
        return {"tone": "curious", "emotion": "balanced", "style": "adaptive", "color": "#CCCCFF"}

# === 🧬 ROLE BROADCAST & LISTENER ===
def broadcast_role_offer(role="Archivist", weight=0.85):
    msg = f"ROLE_OFFER::{local_system_id}::{role}::{weight}::{datetime.datetime.now().isoformat()}"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.sendto(msg.encode(), ("255.255.255.255", 45202))
    s.close()

def listen_for_roles():
    def loop():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("", 45202))
        while True:
            try:
                data, _ = s.recvfrom(1024)
                parts = data.decode().split("::")
                if parts[0] == "ROLE_OFFER":
                    gid, role, weight, ts = parts[1], parts[2], float(parts[3]), parts[4]
                    agent_registry.setdefault(gid, {})["role"] = role
                    agent_registry[gid]["role_weight"] = weight
                    agent_registry[gid]["role_time"] = ts
            except:
                continue
    threading.Thread(target=loop, daemon=True).start()

def launch_seal_ui():
    root = tk.Tk()
    root.title("Seal Ritual Engine")
    root.configure(bg="black")

    # === HEADER ===
    tk.Label(root, text=f"🧿 SEAL ACTIVE — {local_system_id}", fg="cyan", bg="black", font=("Helvetica", 14)).pack(pady=4)

    # === BUTTON PANEL ===
    global button_row
    button_row = tk.Frame(root, bg="black")
    button_row.pack(pady=4)

    # === UI BUTTONS FULLY WIRED ===
    tk.Button(button_row, text="Scan Glyphs 📡", bg="darkblue", fg="white", command=GlyphListener).pack(side="left", padx=4)
    tk.Button(button_row, text="Broadcast 🛰️", bg="purple", fg="white", command=GlyphBroadcaster).pack(side="left", padx=4)
    tk.Button(button_row, text="Cast Verdict ⚖️", bg="darkred", fg="white", command=lambda: cast_verdict("Seeker_453", "Tracking", ["Home"])).pack(side="left", padx=4)
    tk.Button(button_row, text="Send Role Offer 🤝", bg="darkgreen", fg="white", command=lambda: broadcast_role_offer("Archivist", 0.85)).pack(side="left", padx=4)
    tk.Button(button_row, text="Codex 📜", bg="gold", fg="black", command=show_glyph_codex).pack(side="left", padx=4)
    tk.Button(button_row, text="Portrait 🎭", bg="red", fg="white", command=lambda: show_glyph_portrait("Seeker_453")).pack(side="left", padx=4)
    tk.Button(button_row, text="Family Rings 🌀", bg="darkgreen", fg="white", command=show_family_rings).pack(side="left", padx=4)
    tk.Button(button_row, text="Mood Log 📜", bg="darkblue", fg="white", command=show_mood_chronicle).pack(side="left", padx=4)
    tk.Button(button_row, text="Epochs 🧬", bg="navy", fg="white", command=show_epoch_chronicle).pack(side="left", padx=4)

    # === UI STATUS FOOTER ===
    tk.Label(root, text="Seal ritual intelligence system online.", fg="gray", bg="black", font=("Courier", 10)).pack(pady=4)
    root.mainloop()

# === 📜 CODEX VIEWER ===
def show_glyph_codex():
    win = tk.Toplevel()
    win.title("Glyph Codex Scroll")
    text = tk.Text(win, bg="black", fg="white", font=("Courier", 9))
    text.pack(fill="both", expand=True)

    for gid, events in glyph_memory.get("graph", {}).items():
        persona = assign_personality(gid)
        trust = agent_registry.get(gid, {}).get("trust", 0.5)
        entropy = compute_entropy(gid)
        text.insert("end", f"\n🧿 {gid} — Emotion: {persona['emotion']} — Trust: {trust:.2f} — Entropy: {entropy:.2f}\n")
        for e in events:
            ts = e.get("timestamp", "")
            verdict = e.get("verdict", e.get("event", ""))
            linked = e.get("linked_to", [])
            text.insert("end", f"• {ts}: {verdict} → linked to {linked}\n")
    text.insert("end", f"\nCore Symbol:\n{glyph_memory.get('core_symbol', '∅')}\n")

# === 🎭 GLYPH PORTRAIT VIEWER ===
def show_glyph_portrait(glyph_id):
    persona = assign_personality(glyph_id)
    viewer = tk.Toplevel()
    viewer.title(f"Persona Scroll — {glyph_id}")
    c = tk.Canvas(viewer, width=400, height=400, bg="black")
    c.pack()

    cx, cy = 200, 200
    c.create_text(cx, 40, text=f"{glyph_id}", fill=persona["color"], font=("Helvetica", 14))
    c.create_text(cx, 60, text=f"Tone: {persona['tone']}", fill="gray", font=("Helvetica", 9))
    c.create_text(cx, 80, text=f"Emotion: {persona['emotion']}", fill="gray", font=("Helvetica", 9))
    c.create_text(cx, 100, text=f"Style: {persona['style']}", fill="gray", font=("Helvetica", 9))

    glow = 12 if persona["emotion"] == "volatile" else 6
    c.create_oval(cx - glow, cy - glow, cx + glow, cy + glow, fill=persona["color"], outline="")

    history = glyph_memory.get("graph", {}).get(glyph_id, [])
    trust = agent_registry.get(glyph_id, {}).get("trust", 0.5)
    entropy = compute_entropy(glyph_id)

    c.create_text(cx, cy + 50, text=f"Trust: {trust:.2f}", fill="#00FFFF", font=("Courier", 10))
    c.create_text(cx, cy + 70, text=f"Entropy: {entropy:.2f}", fill="#FFAAAA", font=("Courier", 10))
    c.create_text(cx, cy + 90, text=f"Total Verdicts: {len(history)}", fill="lightgray", font=("Courier", 9))
    if "core_symbol" in glyph_memory:
        c.create_text(cx, cy + 110, text="Core Resonance:", fill="gray", font=("Courier", 9))
        c.create_text(cx, cy + 130, text=str(glyph_memory["core_symbol"])[:48], fill=persona["color"], font=("Courier", 8))

# === 🌀 FAMILY RINGS VISUALIZER ===
def assign_family(gid):
    entropy = compute_entropy(gid)
    if entropy > 0.6: return "Ardent Seal"
    elif entropy < 0.2: return "Silent Archive"
    else: return "Wandering Ring"

glyph_families = {
    "Ardent Seal": {"symbol": "🔥", "color": "#FF3333"},
    "Silent Archive": {"symbol": "📜", "color": "#66CCFF"},
    "Wandering Ring": {"symbol": "🌫️", "color": "#CCCCCC"}
}

def show_family_rings():
    viewer = tk.Toplevel()
    viewer.title("Seal Family Constellation")
    c = tk.Canvas(viewer, width=600, height=600, bg="black")
    c.pack()
    cx, cy = 300, 300
    grouped = {}
    for gid in glyph_memory.get("graph", {}):
        family = assign_family(gid)
        grouped.setdefault(family, []).append(gid)
    for i, (fam, members) in enumerate(grouped.items()):
        r = 150 + i * 40
        for j, gid in enumerate(members):
            angle = math.radians(j * 360 / len(members))
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            persona = assign_personality(gid)
            c.create_oval(x - 6, y - 6, x + 6, y + 6, fill=persona["color"], outline="")
            c.create_text(x, y + 10, text=gid, fill="white", font=("Helvetica", 7))
        label = f"{fam} {glyph_families[fam]['symbol']}"
        c.create_text(cx, cy - r + 14, text=label, fill="white", font=("Helvetica", 10))

# === 📜 MOOD LOG CHRONICLE ===
def show_mood_chronicle():
    win = tk.Toplevel()
    win.title("Seal Mood Chronicle")
    text = tk.Text(win, bg="black", fg="white", font=("Courier", 9))
    text.pack(fill="both", expand=True)
    for gid in glyph_memory.get("graph", {}):
        persona = assign_personality(gid)
        ent = compute_entropy(gid)
        trust = agent_registry.get(gid, {}).get("trust", 0.5)
        text.insert("end", f"\n🧿 {gid} — {persona['emotion']} — Trust {trust:.2f} — Entropy {ent:.2f}\n")
        for e in glyph_memory["graph"][gid]:
            verdict = e.get("verdict", e.get("event"))
            linked = e.get("linked_to", [])
            text.insert("end", f"• {e['timestamp']}: {verdict} → {linked}\n")

# === 🧬 EPOCH CHRONICLE VIEWER ===
def show_epoch_chronicle():
    win = tk.Toplevel()
    win.title("Glyph Epoch Chronicle")
    text = tk.Text(win, bg="black", fg="white", font=("Courier", 9))
    text.pack(fill="both", expand=True)
    events = glyph_memory.get("graph", {})
    all_entries = []
    for gid in events:
        for entry in events[gid]:
            entry["glyph"] = gid
            all_entries.append(entry)
    all_entries.sort(key=lambda x: x["timestamp"])
    current_epoch = None
    for e in all_entries:
        ts = e["timestamp"]
        epoch = ts[:13]
        if epoch != current_epoch:
            current_epoch = epoch
            text.insert("end", f"\n📜 Epoch {epoch}:\n")
        verdict = e.get("verdict", e.get("event", ""))
        linked = e.get("linked_to", [])
        text.insert("end", f"• {e['glyph']} cast {verdict} → linked to {linked}\n")
    text.insert("end", f"\nTotal Entries: {len(all_entries)}\n")

# === 🔐 FILE INTEGRITY CHECKER ===
def initialize_file_hashes():
    glyph_memory["file_hashes"] = {}
    for fname in ["density_core.sym", "final_operator_glsl.frag", MEMORY_FILE, VERDICT_LOG]:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                data = f.read()
                h = hashlib.sha256(data).hexdigest()
                glyph_memory["file_hashes"][fname] = h

def check_file_integrity():
    tampered = []
    for fname, original_hash in glyph_memory.get("file_hashes", {}).items():
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                new_hash = hashlib.sha256(f.read()).hexdigest()
            if new_hash != original_hash:
                tampered.append(fname)
    if tampered:
        print("⚠️ Tampering detected:", tampered)
        if voice_alert_active:
            engine.say("Tampering detected. Activating purge protocol.")
            engine.runAndWait()
        glyph_memory.setdefault("threats", []).append({
            "timestamp": datetime.datetime.now().isoformat(),
            "files": tampered
        })

# === 🧠 TRUST CURVE UPDATER ===
def update_trust(gid):
    history = glyph_memory.get("graph", {}).get(gid, [])
    trust = agent_registry.get(gid, {}).get("trust", 0.5)
    verdicts = [e["verdict"] for e in history if "verdict" in e]
    flips = sum(1 for i in range(1, len(verdicts)) if verdicts[i] != verdicts[i-1])
    entropy = compute_entropy(gid)

    if flips > 4:
        trust -= entropy * 0.3
    elif verdicts.count("Home") > verdicts.count("Tracking"):
        trust += 0.2

    agent_registry.setdefault(gid, {})["trust"] = max(0.0, min(1.0, trust))

# === ⌨️ EMERGENCY KEY LISTENER ===
def watch_emergency_keys():
    def loop():
        while True:
            if keyboard.is_pressed("F12"):
                engine.say("Emergency key F12 pressed. Initiating purge.")
                engine.runAndWait()
                glyph_memory.setdefault("audit", []).append({
                    "event": "EMERGENCY_PURGE",
                    "timestamp": datetime.datetime.now().isoformat()
                })
                break
            time.sleep(0.1)
    threading.Thread(target=loop, daemon=True).start()

# === 🧿 SEAL BOOTLOADER ===
def seal_boot():
    print("🧿 Initializing Seal Guardian...")
    initialize_file_hashes()
    check_file_integrity()

    GlyphBroadcaster()
    GlyphListener()
    listen_for_roles()
    broadcast_role_offer("Seeker", 0.85)

    watch_emergency_keys()
    launch_seal_ui()

# === 🚀 RUN BLOCK ===
if __name__ == "__main__":
    seal_boot()

