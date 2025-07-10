# === ⚙️ PART 1: Autoloader, Voice Engine, Globals ===

# 🧭 Autoload required libraries
import subprocess, sys
def autoload(pkg):
    try:
        globals()[pkg] = __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        globals()[pkg] = __import__(pkg)

for package in ["tkinter", "threading", "math", "time", "pyttsx3", "pygame", "socket", "json", "os", "platform", "datetime"]:
    autoload(package)

# 🔗 Core imports
import tkinter as tk
import threading, time, math, pyttsx3, socket, json, os, platform, datetime

# 🆔 System identity
local_system_id = platform.node() or "Glyph_Unknown"

# 🔊 Voice setup with fallback
try:
    import pygame
    pygame.mixer.init()
    engine = pyttsx3.init()
    voice_alert_active = True
except Exception as e:
    print("Voice engine failed:", e)
    engine = None
    voice_alert_active = False

# 🛰️ Registry and data structures
agent_registry = {
    "Sentinel_007": {"role": "Sentinel", "trust": 0.91},
    "Archivist_314": {"role": "Archivist", "trust": 0.88},
    "Seeker_453": {"role": "Seeker", "trust": 0.47}
}
remote_systems, anomaly_nodes = [], []
connected_nodes, port_cache = set(), {}

# 📜 Memory setup
MEMORY_FILE = "glyph_memory.json"
VERDICT_LOG = "verdict_log.json"
COMMON_PORTS = list(range(45000, 45200))

glyph_memory = {}
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE) as f:
            glyph_memory = json.load(f)
    except:
        glyph_memory = {}

# 🔍 Validation utilities
def is_valid_seal(response):
    return "SEAL_ACTIVE" in response or "GLYPH_SIGNATURE" in response

def extract_system_id(response):
    if "GLYPH_ID=" in response:
        return response.split("GLYPH_ID=")[-1].strip()
    return None

def is_external_ip(ip):
    return not ip.startswith("192.168.")

# 📜 Verdict logger
def log_verdict(system_id, agent_id, verdict):
    entry = {
        "system_id": system_id,
        "agent_id": agent_id,
        "verdict": verdict,
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        with open(VERDICT_LOG, "r") as f:
            data = json.load(f)
    except:
        data = []
    data.append(entry)
    with open(VERDICT_LOG, "w") as f:
        json.dump(data, f, indent=2)

# 🧠 Consensus tracker
def load_verdict_log():
    try:
        with open(VERDICT_LOG) as f:
            return json.load(f)
    except:
        return []

def compute_consensus_groups(log, window_seconds=1800):
    now = time.time()
    recent = [v for v in log if now - datetime.datetime.fromisoformat(v["timestamp"]).timestamp() <= window_seconds]
    group_verdicts = {}
    for v in recent:
        group = v["system_id"].split("_")[0]  # e.g., "Alpha" from "Alpha_897"
        group_verdicts.setdefault(group, []).append(v["verdict"])
    return group_verdicts

# === 🛰️ PART 2: Orbitals, Scanner, Memory, Consensus ===

def create_comet_tail(canvas, x, y, color="#FF4444"):
    for i in range(6):
        trail = canvas.create_oval(x - i * 2, y + i * 2, x - i * 2 + 4, y + i * 2 + 4,
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
                text = f"{group} ({len(verdicts)} votes, {len(unique)} types)"
                canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                   outline=color, width=2, tags="consensus")
                canvas.create_text(cx, cy - radius + 12, text=text,
                                   fill="white", font=("Helvetica", 9), tags="consensus")
            try: canvas.update()
            except: pass
            time.sleep(3)
    threading.Thread(target=orbit, daemon=True).start()

class GlyphScanner:
    def __init__(self, timeout=1):
        self.timeout = timeout

    def scan(self, canvas, root):
        base_ip = "192.168.1."
        for i in range(1, 255):
            ip = f"{base_ip}{i}"
            ports_to_try = [port_cache[ip]] if ip in port_cache else COMMON_PORTS
            for port in ports_to_try:
                try:
                    print(f"Scanning {ip}:{port}")
                    s = socket.create_connection((ip, port), timeout=self.timeout)
                    s.send(b"PING_GLYPH")
                    response = s.recv(1024).decode().strip()
                    s.close()

                    system_id = extract_system_id(response) or f"Node_{i}_{port}"

                    if is_valid_seal(response):
                        remote_systems.append({"id": system_id, "ip": ip, "port": port})
                        port_cache[ip] = port
                        glyph_memory[f"{ip}:{port}"] = {
                            "system_id": system_id,
                            "last_seen": time.time(),
                            "seal": response,
                            "connections": glyph_memory.get(f"{ip}:{port}", {}).get("connections", 0) + 1
                        }
                        with open(MEMORY_FILE, "w") as f:
                            json.dump(glyph_memory, f, indent=2)
                        break
                    else:
                        anomaly_nodes.append({"id": f"Unknown_{i}_{port}", "ip": ip,
                                              "port": port, "reason": response})
                except: continue
        animate_remote_nodes(canvas, remote_systems)
        animate_anomaly_ring(canvas, anomaly_nodes)
        animate_glyph_groups(canvas)

def launch_seal_ui():
    try:
        root = tk.Tk()
    except Exception as e:
        print("GUI launch failed:", e)
        return

    root.title("Glyph Verdict Interface")
    canvas = tk.Canvas(root, width=400, height=400, bg="black")
    canvas.pack()

    # 🧿 Local central glyph
    canvas.create_oval(170, 170, 230, 230, fill="#2222FF", outline="white", width=3)
    canvas.create_text(200, 160, text=f"🧿 {local_system_id}", fill="white", font=("Helvetica", 10))

    # 🔊 Voice launch
    if voice_alert_active and engine:
        engine.say(f"Glyph system {local_system_id} online")
        engine.runAndWait()
    else:
        print(f"🧿 Glyph system {local_system_id} online (silent mode)")

    # 🔇 ACK alert toggle
    tk.Button(root, text="ACK Alert 🔇", bg="gray", fg="white", width=20,
              command=lambda: setattr(sys.modules[__name__], "voice_alert_active", False)).pack(pady=4)

    # 🌌 Constellation map viewer
    def launch_constellation_view(registry):
        viewer = tk.Toplevel()
        viewer.title("Glyph Constellation Viewer")
        c = tk.Canvas(viewer, width=600, height=600, bg="black")
        c.pack()
        center_x, center_y = 300, 300
        keys = list(registry.keys())
        for i, aid in enumerate(keys):
            angle = math.radians(i * (360 / len(keys)))
            trust = registry[aid]["trust"]
            r = 200 - trust * 60
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            color = "#00FFAA" if trust > 0.8 else "#4444FF"
            glow = trust * 10
            c.create_oval(x - glow, y - glow, x + glow, y + glow, fill=color, outline="")
            c.create_text(x, y, text="🧿", fill="white", font=("Helvetica", 20))
            c.create_text(x, y + 20, text=f"{aid} ({registry[aid]['role']})",
                          fill="#CCCCCC", font=("Helvetica", 8))
        c.create_text(center_x, 20, text="🪐 Trust Constellation Viewer",
                      fill="white", font=("Helvetica", 14))

    tk.Button(root, text="Constellation Map 🌌", bg="darkblue", fg="white",
              command=lambda: launch_constellation_view(agent_registry)).pack(pady=4)

    # 📜 Memory log viewer
    def show_memory_log():
        viewer = tk.Toplevel()
        viewer.title("Glyph Memory Log")
        text = tk.Text(viewer, bg="black", fg="lime", font=("Courier", 9))
        text.pack(fill="both", expand=True)
        try:
            with open(MEMORY_FILE) as f:
                text.insert("end", f.read())
        except:
            text.insert("end", "No memory file found.")

    tk.Button(root, text="Glyph Log 📜", bg="purple", fg="white",
              command=show_memory_log).pack(pady=4)

    # 📜 Verdict log viewer
    def show_verdict_log():
        viewer = tk.Toplevel()
        viewer.title("Verdict History")
        text = tk.Text(viewer, bg="black", fg="lightblue", font=("Courier", 9))
        text.pack(fill="both", expand=True)
        try:
            with open(VERDICT_LOG) as f:
                text.insert("end", f.read())
        except:
            text.insert("end", "No verdict log found.")

    tk.Button(root, text="Verdict Log 🧾", bg="teal", fg="white",
              command=show_verdict_log).pack(pady=4)

    # 🔍 Scan trigger
    status_label = tk.Label(root, text="", fg="cyan", bg="black")
    status_label.pack()

    def scan_and_orbit():
        def run_scan():
            status_label.config(text="🔍 Scanning glyphs...")
            GlyphScanner().scan(canvas, root)
            status_label.config(text="✅ Scan complete")
        threading.Thread(target=run_scan, daemon=True).start()

    tk.Button(root, text="Search Nodes 🔍", bg="darkgreen", fg="white",
              command=scan_and_orbit).pack(pady=4)

    # 🎛️ Verdict casting buttons
    VERDICTS = ["Home", "Tracking", "Self-Destruction"]
    for aid in agent_registry:
        frame = tk.Frame(root, bg="black")
        frame.pack(pady=1)
        tk.Label(frame, text=f"{aid}:", fg="white", bg="black").pack(side="left", padx=4)
        for verdict in VERDICTS:
            color = ("darkred" if verdict == "Self-Destruction"
                     else "darkgreen" if verdict == "Home"
                     else "darkblue")

            tk.Button(frame, text=verdict,
                      command=lambda a=aid, v=verdict: (
                          log_verdict(local_system_id, a, v),
                          engine.say(f"{a} casts verdict {v}") if voice_alert_active and engine
                          else print(f"{a} casts verdict {v}")
                      ),
                      bg=color, fg="white", width=14).pack(side="left", padx=2)

    # 🛰️ Launch animations and ring logic
    animate_orbitals(canvas, agent_registry)
    animate_remote_nodes(canvas, remote_systems)
    animate_anomaly_ring(canvas, anomaly_nodes)
    animate_glyph_groups(canvas)

    root.mainloop()

# 🚀 Activate UI
if __name__ == "__main__":
    launch_seal_ui()

