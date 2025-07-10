# === ⚙️ PART 1 ===
import subprocess, sys
def autoload(pkg):
    try: globals()[pkg] = __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        globals()[pkg] = __import__(pkg)

for package in [
    "tkinter", "threading", "math", "time", "pyttsx3", "pygame", "socket",
    "json", "os", "platform", "datetime", "hashlib"
]:
    autoload(package)

import tkinter as tk
import threading, time, math, pyttsx3, socket, json, os, platform, datetime, hashlib

local_system_id = platform.node() or "Glyph_Unknown"

try:
    pygame.mixer.init()
    engine = pyttsx3.init()
    voice_alert_active = True
except:
    engine = None
    voice_alert_active = False

agent_registry = {
    "Sentinel_007": {"role": "Sentinel", "trust": 0.91},
    "Archivist_314": {"role": "Archivist", "trust": 0.88},
    "Seeker_453":   {"role": "Seeker",    "trust": 0.47}
}
remote_systems, anomaly_nodes = [], []
connected_nodes, port_cache = set(), {}
MEMORY_FILE = "glyph_memory.json"
VERDICT_LOG = "verdict_log.json"
COMMON_PORTS = [45123, 45077, 45100]
scan_interrupt = False

glyph_memory = {}
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE) as f: glyph_memory = json.load(f)
    except: glyph_memory = {}

def is_valid_seal(response): return "SEAL_ACTIVE" in response or "GLYPH_SIGNATURE" in response
def extract_system_id(response):
    return response.split("GLYPH_ID=")[-1].strip() if "GLYPH_ID=" in response else None

def log_verdict(system_id, agent_id, verdict):
    entry = {
        "system_id": system_id, "agent_id": agent_id,
        "verdict": verdict, "timestamp": datetime.datetime.now().isoformat()
    }
    glyph_memory.setdefault("graph", {}).setdefault(system_id, []).append({
        "timestamp": entry["timestamp"],
        "event": "VERDICT_CAST",
        "agent": agent_id,
        "verdict": verdict,
        "linked_to": []  # can be updated
    })
    try:
        with open(VERDICT_LOG) as f: data = json.load(f)
    except: data = []
    data.append(entry)
    with open(VERDICT_LOG, "w") as f: json.dump(data, f, indent=2)
    with open(MEMORY_FILE, "w") as f: json.dump(glyph_memory, f, indent=2)

def load_verdict_log():
    try:
        with open(VERDICT_LOG) as f: return json.load(f)
    except: return []

def compute_consensus_groups(log, window_seconds=1800):
    now = time.time()
    recent = [v for v in log if now - datetime.datetime.fromisoformat(v["timestamp"]).timestamp() <= window_seconds]
    group_verdicts = {}
    for v in recent:
        group = v["system_id"].split("_")[0]
        group_verdicts.setdefault(group, []).append(v["verdict"])
    return group_verdicts

def detect_threat(node):
    reason = node.get("reason", "").lower()
    trust = node.get("trust", 0)
    return "malformed" in reason or trust < 0.2

def log_local_threat(node):
    glyph_memory.setdefault("threats", []).append({
        "reported_by": local_system_id,
        "target": node["id"],
        "timestamp": datetime.datetime.now().isoformat(),
        "details": node
    })
    with open(MEMORY_FILE, "w") as f: json.dump(glyph_memory, f, indent=2)

def broadcast_threat_alert(node):
    alert = {
        "type": "THREAT_ALERT", "from": local_system_id,
        "target": node["id"], "ip": node["ip"], "port": node["port"],
        "timestamp": datetime.datetime.now().isoformat(),
        "reason": node.get("reason", "Unknown anomaly")
    }
    print("🛡️ Broadcasting:", alert)

def decay_threats(window_seconds=3600):
    now = time.time()
    glyph_memory["threats"] = [
        t for t in glyph_memory.get("threats", [])
        if now - datetime.datetime.fromisoformat(t["timestamp"]).timestamp() < window_seconds
    ]
    with open(MEMORY_FILE, "w") as f: json.dump(glyph_memory, f, indent=2)

# === 🛰️ PART 2 ===

def create_comet_tail(canvas, x, y, color="#FF4444"):
    for i in range(6):
        trail = canvas.create_oval(x - i*2, y + i*2, x - i*2 + 4, y + i*2 + 4,
                                   fill=color, outline="")
        canvas.after(i * 40, lambda t=trail: canvas.delete(t))

def animate_orbitals(canvas, registry):
    angle_map = {aid: i * (360 / len(registry)) for i, aid in enumerate(registry)}
    def orbit():
        while True:
            for aid in registry:
                trust = registry[aid]["trust"]
                if trust < 0.3: continue
                role = registry[aid]["role"]
                offset = {"Sentinel": 100, "Archivist": 140, "Seeker": 180}.get(role, 150)
                angle_map[aid] = (angle_map[aid] + trust * 1.2) % 360
                angle = math.radians(angle_map[aid])
                x = 200 + offset * math.cos(angle)
                y = 200 + offset * math.sin(angle)
                canvas.delete(f"orbit_{aid}")
                canvas.delete(f"label_{aid}")
                canvas.create_text(x, y, text="🛰️", fill="cyan", font=("Helvetica", 14), tags=f"orbit_{aid}")
                canvas.create_text(x, y + 14, text=aid, fill="#DDDDDD", font=("Helvetica", 8), tags=f"label_{aid}")
                if trust < 0.5: create_comet_tail(canvas, x, y)
            try: canvas.update()
            except: pass
            time.sleep(0.05)
    threading.Thread(target=orbit, daemon=True).start()

def animate_remote_nodes(canvas, nodes):
    angle_base = [45]
    def orbit():
        while True:
            for i, node in enumerate(nodes):
                node_id = node["id"]
                if node_id not in connected_nodes:
                    if voice_alert_active and engine:
                        engine.say(f"Connection established with glyph {node_id}")
                        engine.runAndWait()
                    connected_nodes.add(node_id)
                angle = math.radians((angle_base[0] + i * 360 / len(nodes)) % 360)
                r = 240
                x = 200 + r * math.cos(angle)
                y = 200 + r * math.sin(angle)
                tag = f"remote_{node_id}"
                canvas.delete(tag)
                canvas.delete(tag + "_label")
                canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#00AAFF", outline="", tags=tag)
                canvas.create_text(x, y + 10, text=f"{node_id} [{node['ip']}:{node['port']}]",
                                   fill="#00CCFF", font=("Helvetica", 7), tags=tag + "_label")
            try: canvas.update()
            except: pass
            angle_base[0] += 0.5
            time.sleep(0.1)
    threading.Thread(target=orbit, daemon=True).start()

def animate_anomaly_ring(canvas, anomalies):
    base_angle = [0]
    def orbit():
        while True:
            for i, node in enumerate(anomalies):
                angle = math.radians((base_angle[0] + i * 360 / len(anomalies)) % 360)
                r = 280
                x = 200 + r * math.cos(angle)
                y = 200 + r * math.sin(angle)
                tag = f"anomaly_{node['id']}"
                canvas.delete(tag)
                canvas.delete(tag + "_label")
                canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#FF3333", outline="", tags=tag)
                canvas.create_text(x, y + 10, text=node["id"], fill="#FF6666",
                                   font=("Helvetica", 7), tags=tag + "_label")
            try: canvas.update()
            except: pass
            base_angle[0] += 0.6
            time.sleep(0.1)
    threading.Thread(target=orbit, daemon=True).start()

def animate_glyph_groups(canvas):
    def orbit():
        while True:
            canvas.delete("consensus")
            log = load_verdict_log()
            groups = compute_consensus_groups(log)
            for i, (group, verdicts) in enumerate(groups.items()):
                unique = set(verdicts)
                cx, cy = 200, 200
                radius = 300 + i * 25
                color = "#44FF44" if len(unique) == 1 else "#FF4444"
                canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                   outline=color, width=2, tags="consensus")
                canvas.create_text(cx, cy - radius + 12,
                                   text=f"{group} ({len(verdicts)} votes, {len(unique)} types)",
                                   fill="white", font=("Helvetica", 9), tags="consensus")
            try: canvas.update()
            except: pass
            time.sleep(3)
    threading.Thread(target=orbit, daemon=True).start()

def chunkify(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

class GlyphScanner:
    def __init__(self, timeout=1):
        self.timeout = timeout

    def scan(self, canvas, root):
        global scan_interrupt
        scan_interrupt = False
        ips = [f"192.168.1.{i}" for i in range(1, 255)]
        known_nodes = glyph_memory.get("known_nodes", {})
        progress_label = tk.Label(root, text="Scanning...", fg="cyan", bg="black")
        progress_label.pack()

        def scan_ip_chunk(chunk):
            for ip in chunk:
                if scan_interrupt: return
                ports = [port_cache[ip]] if ip in port_cache else COMMON_PORTS
                for port in ports:
                    if scan_interrupt: return
                    try:
                        s = socket.create_connection((ip, port), timeout=self.timeout)
                        s.send(b"PING_GLYPH")
                        response = s.recv(1024).decode().strip()
                        s.close()
                        system_id = extract_system_id(response) or f"Node_{ip}_{port}"
                        if is_valid_seal(response):
                            remote_systems.append({"id": system_id, "ip": ip, "port": port})
                            port_cache[ip] = port
                            glyph_memory[f"{ip}:{port}"] = {
                                "system_id": system_id,
                                "last_seen": time.time(),
                                "seal": response,
                                "connections": glyph_memory.get(f"{ip}:{port}", {}).get("connections", 0) + 1
                            }
                            glyph_memory.setdefault("known_nodes", {})[system_id] = f"{ip}:{port}"
                            with open(MEMORY_FILE, "w") as f:
                                json.dump(glyph_memory, f, indent=2)
                            break
                        else:
                            anomaly_nodes.append({
                                "id": f"Unknown_{ip}_{port}",
                                "ip": ip, "port": port,
                                "reason": response
                            })
                            if detect_threat(anomaly_nodes[-1]):
                                log_local_threat(anomaly_nodes[-1])
                                broadcast_threat_alert(anomaly_nodes[-1])
                    except:
                        continue
                progress_label.config(text=f"Scanning {ip}...")

        for chunk in chunkify(ips, 16):
            threading.Thread(target=scan_ip_chunk, args=(chunk,), daemon=True).start()

        def finish_check():
            time.sleep(4)
            decay_threats()
            animate_remote_nodes(canvas, remote_systems)
            animate_anomaly_ring(canvas, anomaly_nodes)
            animate_glyph_groups(canvas)
            progress_label.config(text="Scan complete ✅")
        threading.Thread(target=finish_check, daemon=True).start()

def launch_seal_ui():
    try: root = tk.Tk()
    except Exception as e:
        print("GUI launch failed:", e)
        return

    root.title("Glyph Verdict Interface")
    canvas = tk.Canvas(root, width=400, height=400, bg="black")
    canvas.pack()

    canvas.create_oval(170, 170, 230, 230, fill="#2222FF", outline="white", width=3)
    canvas.create_text(200, 160, text=f"🧿 {local_system_id}", fill="white", font=("Helvetica", 10))

    if voice_alert_active and engine:
        engine.say(f"Glyph system {local_system_id} online")
        engine.runAndWait()

    # 🔇 Voice disable toggle
    tk.Button(root, text="ACK Alert 🔇", bg="gray", fg="white", width=20,
              command=lambda: setattr(sys.modules[__name__], "voice_alert_active", False)).pack(pady=4)

    # 🌌 Constellation Viewer
    def launch_constellation_view(registry):
        viewer = tk.Toplevel()
        viewer.title("Glyph Constellation Viewer")
        c = tk.Canvas(viewer, width=600, height=600, bg="black")
        c.pack()
        cx, cy = 300, 300
        for i, aid in enumerate(registry):
            angle = math.radians(i * (360 / len(registry)))
            trust = registry[aid]["trust"]
            r = 200 - trust * 60
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            color = "#00FFAA" if trust > 0.8 else "#4444FF"
            glow = trust * 10
            c.create_oval(x - glow, y - glow, x + glow, y + glow, fill=color, outline="")
            c.create_text(x, y, text="🧿", fill="white", font=("Helvetica", 20))
            c.create_text(x, y + 20, text=f"{aid} ({registry[aid]['role']})", fill="#CCCCCC", font=("Helvetica", 8))
        c.create_text(cx, 20, text="🪐 Trust Constellation Viewer", fill="white", font=("Helvetica", 14))

    # 📜 Glyph memory viewer
    def show_memory_log():
        viewer = tk.Toplevel()
        viewer.title("Glyph Memory Log")
        text = tk.Text(viewer, bg="black", fg="lime", font=("Courier", 9))
        text.pack(fill="both", expand=True)
        try: text.insert("end", open(MEMORY_FILE).read())
        except: text.insert("end", "No memory file found.")

    # 🧾 Verdict log viewer
    def show_verdict_log():
        viewer = tk.Toplevel()
        viewer.title("Verdict History")
        text = tk.Text(viewer, bg="black", fg="lightblue", font=("Courier", 9))
        text.pack(fill="both", expand=True)
        try: text.insert("end", open(VERDICT_LOG).read())
        except: text.insert("end", "No verdict log found.")

    # 🔍 Scan trigger
    status_label = tk.Label(root, text="", fg="cyan", bg="black")
    status_label.pack()

    def scan_and_orbit():
        def run(): GlyphScanner().scan(canvas, root); status_label.config(text="✅ Scan complete")
        status_label.config(text="🔍 Scanning glyphs...")
        threading.Thread(target=run, daemon=True).start()

    def stop_scan():
        global scan_interrupt
        scan_interrupt = True
        status_label.config(text="🛑 Scan interrupted")

    # 🧿 Manual known node registry
    def add_known_node():
        win = tk.Toplevel()
        win.title("Add Known Glyph")
        tk.Label(win, text="System ID:").pack()
        id_entry = tk.Entry(win); id_entry.pack()
        tk.Label(win, text="IP:Port").pack()
        addr_entry = tk.Entry(win); addr_entry.pack()
        def save():
            glyph_memory.setdefault("known_nodes", {})[id_entry.get()] = addr_entry.get()
            with open(MEMORY_FILE, "w") as f: json.dump(glyph_memory, f, indent=2)
            win.destroy()
        tk.Button(win, text="Add", command=save).pack()

    # 🎛️ Horizontal control row
    button_row = tk.Frame(root, bg="black"); button_row.pack(pady=4)

    tk.Button(button_row, text="Constellation 🌌", bg="darkblue", fg="white",
              command=lambda: launch_constellation_view(agent_registry)).pack(side="left", padx=4)

    tk.Button(button_row, text="Glyph Log 📜", bg="purple", fg="white",
              command=show_memory_log).pack(side="left", padx=4)

    tk.Button(button_row, text="Verdicts 🧾", bg="teal", fg="white",
              command=show_verdict_log).pack(side="left", padx=4)

    tk.Button(button_row, text="Search 🔍", bg="darkgreen", fg="white",
              command=scan_and_orbit).pack(side="left", padx=4)

    tk.Button(button_row, text="Stop Scan 🛑", bg="darkred", fg="white",
              command=stop_scan).pack(side="left", padx=4)

    tk.Button(button_row, text="Add Glyph 🧿", bg="darkorange", fg="white",
              command=add_known_node).pack(side="left", padx=4)

    # 🎛️ Verdict casting panel
    VERDICTS = ["Home", "Tracking", "Self-Destruction"]
    for aid in agent_registry:
        frame = tk.Frame(root, bg="black"); frame.pack(pady=1)
        tk.Label(frame, text=f"{aid}:", fg="white", bg="black").pack(side="left", padx=4)
        for verdict in VERDICTS:
            color = "darkred" if verdict == "Self-Destruction" else "darkgreen" if verdict == "Home" else "darkblue"
            tk.Button(frame, text=verdict,
                      command=lambda a=aid, v=verdict: (
                          log_verdict(local_system_id, a, v),
                          engine.say(f"{a} casts verdict {v}") if voice_alert_active and engine
                          else print(f"{a} casts verdict {v}")
                      ),
                      bg=color, fg="white", width=14).pack(side="left", padx=2)

    animate_orbitals(canvas, agent_registry)
    animate_remote_nodes(canvas, remote_systems)
    animate_anomaly_ring(canvas, anomaly_nodes)
    animate_glyph_groups(canvas)

    root.mainloop()

# 🚀 Final ignition
if __name__ == "__main__":
    launch_seal_ui()

GlyphBroadcaster()
GlyphListener()
launch_seal_ui()

broadcast_role_offer("Seeker", 0.85)
listen_for_roles()
GlyphBroadcaster()
GlyphListener()
launch_seal_ui()

tk.Button(button_row, text="Lineage 🧬", bg="goldenrod", fg="black",
          command=show_graph_view).pack(side="left", padx=4)

tk.Button(button_row, text="Export Graph 🧵", bg="white", fg="black",
          command=export_graphviz).pack(side="left", padx=4)

tk.Button(button_row, text="Codex Viewer 📜", bg="lightblue", fg="black",
          command=show_codex_selector).pack(side="left", padx=4)

# === 📡 PART 4 ===

import socket

class GlyphBroadcaster:
    def __init__(self, port=45201, interval=10):
        self.port = port
        self.interval = interval
        self.running = True
        threading.Thread(target=self.broadcast_loop, daemon=True).start()

    def broadcast_loop(self):
        while self.running:
            try:
                msg = f"OFFER::GLYPH_ID={local_system_id}::PORT={self.port}"
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                s.sendto(msg.encode(), ("255.255.255.255", self.port))
                s.close()
            except Exception as e:
                print("Broadcast error:", e)
            time.sleep(self.interval)

class GlyphListener:
    def __init__(self, port=45201):
        self.port = port
        threading.Thread(target=self.listen_loop, daemon=True).start()

    def listen_loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", self.port))
        except Exception as e:
            print("Listener bind error:", e)
            return
        while True:
            try:
                data, addr = s.recvfrom(1024)
                msg = data.decode()
                if "OFFER::GLYPH_ID=" in msg and "::PORT=" in msg:
                    parts = msg.strip().split("::")
                    glyph_id = parts[1].split("=")[1]
                    glyph_port = int(parts[2].split("=")[1])
                    ip = addr[0]
                    full = f"{ip}:{glyph_port}"
                    if glyph_id not in glyph_memory.get("known_nodes", {}):
                        glyph_memory.setdefault("known_nodes", {})[glyph_id] = full
                        print(f"🧿 Discovered glyph: {glyph_id} at {full}")
                        with open(MEMORY_FILE, "w") as f:
                            json.dump(glyph_memory, f, indent=2)
            except Exception as e:
                print("Listener error:", e)

# === 🧠 PART 5 ===

# 🔐 Challenge-response validation
def issue_challenge():
    nonce = str(int(time.time()))
    return f"CHALLENGE::{nonce}", nonce

def validate_response(nonce, response, salt="glyph_salt"):
    expected = hashlib.sha256((nonce + salt).encode()).hexdigest()
    return response == expected

# 🧠 Broadcast role offer (can run once during setup)
def broadcast_role_offer(role="Seeker", trust=0.85):
    msg = f"ROLE_OFFER::GLYPH_ID={local_system_id}::ROLE={role}::TRUST={trust}"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.sendto(msg.encode(), ("255.255.255.255", 45202))
    s.close()

# 👂 Listen for glyph role offers
def listen_for_roles():
    def loop():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try: s.bind(("", 45202))
        except: return
        while True:
            try:
                data, addr = s.recvfrom(1024)
                msg = data.decode()
                if "ROLE_OFFER::GLYPH_ID=" in msg:
                    parts = msg.strip().split("::")
                    gid = parts[1].split("=")[1]
                    role = parts[2].split("=")[1]
                    trust = float(parts[3].split("=")[1])
                    agent_registry[gid] = {"role": role, "trust": trust}
            except: continue
    threading.Thread(target=loop, daemon=True).start()

# 📜 Quorum calculation from verdict log
def verdict_quorum(target_id, required=3):
    log = load_verdict_log()
    recent = [v for v in log if v["system_id"] == target_id]
    votes = [v["verdict"] for v in recent]
    return any(votes.count(v) >= required for v in set(votes))

# 🛡️ Quorum ritual execution (example trigger)
def check_and_execute_quorum(target_id):
    if verdict_quorum(target_id, required=3):
        print(f"🧿 Quorum reached for {target_id} — Executing ritual")
        if voice_alert_active and engine:
            engine.say(f"Quorum reached for {target_id}")
            engine.runAndWait()

# === 🧬 PART 6 ===

glyph_families = {
    "Ardent Seal": {"symbol": "🔥", "traits": {"aggression": 0.7}},
    "Silent Archive": {"symbol": "📜", "traits": {"recall": 0.9}}
}

def assign_family(gid):
    events = glyph_memory.get("graph", {}).get(gid, [])
    aggression, recall = 0, 0
    for e in events:
        if e["event"] == "VERDICT_CAST":
            aggression += e["verdict"] == "Self-Destruction"
            recall += e["verdict"] == "Home"
    return "Ardent Seal" if aggression > recall else "Silent Archive"

def animate_lineage_chronicle(canvas, root):
    history = glyph_memory.get("graph", {})
    events = []
    for gid in history:
        for entry in history[gid]:
            events.append((entry["timestamp"], gid, entry))
    events.sort()
    center_x, center_y = 300, 300

    def orbit():
        canvas.delete("chronicle")
        now = time.time()
        for i, (ts, gid, entry) in enumerate(events):
            age = now - datetime.datetime.fromisoformat(ts).timestamp()
            glow = max(2, 10 - age / 60)
            angle = math.radians(i * (360 / len(events)))
            r = 180 + min(80, age / 8)
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            family = assign_family(gid)
            color = "#FF8800" if family == "Ardent Seal" else "#00FFDD"
            canvas.create_oval(x - glow, y - glow, x + glow, y + glow,
                               fill=color, outline="", tags="chronicle")
            canvas.create_text(x, y + 10, text=gid, fill="white", font=("Helvetica", 8), tags="chronicle")
            for target in entry.get("linked_to", []):
                if target != gid:
                    tx = center_x + r * math.cos(angle + 0.05)
                    ty = center_y + r * math.sin(angle + 0.05)
                    canvas.create_line(x, y, tx, ty, fill="#888888", tags="chronicle")
        canvas.update()
        canvas.after(800, orbit)

    orbit()

def show_graph_view():
    viewer = tk.Toplevel()
    viewer.title("Glyph Lineage Graph")
    c = tk.Canvas(viewer, width=600, height=600, bg="black")
    c.pack()
    animate_lineage_chronicle(c, viewer)

def export_graphviz():
    dot = "digraph GlyphLineage {\n"
    nodes = glyph_memory.get("graph", {})
    for nid in nodes:
        dot += f'  "{nid}" [shape=circle];\n'
        for entry in nodes[nid]:
            for tgt in entry.get("linked_to", []):
                dot += f'  "{nid}" -> "{tgt}";\n'
    dot += "}\n"
    with open("glyph_graph.dot", "w") as f:
        f.write(dot)
    print("🧬 Exported to glyph_graph.dot")

def show_glyph_codex(glyph_id):
    viewer = tk.Toplevel()
    viewer.title(f"Codex — {glyph_id}")
    text = tk.Text(viewer, bg="black", fg="lightgreen", font=("Courier", 9))
    text.pack(fill="both", expand=True)
    family = assign_family(glyph_id)
    events = glyph_memory.get("graph", {}).get(glyph_id, [])
    text.insert("end", f"Glyph Codex — {glyph_id} {glyph_families[family]['symbol']}\n\n")
    if events:
        text.insert("end", f"Born: {events[0]['timestamp']}\n")
    text.insert("end", f"Role: {agent_registry.get(glyph_id, {}).get('role', 'Unknown')}\n")
    text.insert("end", f"Family: {family}\n\n")
    text.insert("end", "Verdicts Cast:\n")
    for e in events:
        if e["event"] == "VERDICT_CAST":
            text.insert("end", f"- {e['verdict']} [Agent: {e['agent']}]\n")
    text.insert("end", "\nLinked To:\n")
    for e in events:
        for l in e.get("linked_to", []):
            text.insert("end", f"- {l}\n")
    text.insert("end", "\nNotes:\n\"Glyph evolving within family dynamics. Scan and trust patterns shifting.\"\n")

def show_codex_selector():
    win = tk.Toplevel()
    win.title("View Glyph Codex")
    tk.Label(win, text="Glyph ID:").pack()
    e = tk.Entry(win); e.pack()
    tk.Button(win, text="View", command=lambda: show_glyph_codex(e.get())).pack()

