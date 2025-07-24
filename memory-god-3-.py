# ===== AUTO LIBRARY LOADER =====
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    import time, hashlib, random, threading, requests
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    import os
    os.system("pip install requests matplotlib")
    import tkinter as tk
    from tkinter import ttk, messagebox
    import time, hashlib, random, threading, requests
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===== CORE MEMORY CELL MODEL =====
class MemoryCell:
    def __init__(self, identifier, content="", location="", shadow=False, bridge_id=None, weight=1.0):
        self.identifier = identifier
        self.content = content
        self.location = location
        self.shadow = shadow
        self.bridge_id = bridge_id
        self.weight = weight
        self.certificate = self.generate_certificate()

    def generate_certificate(self):
        raw = f"{self.identifier}{self.content}{self.location}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def verify_certificate(self):
        return self.certificate == self.generate_certificate()

    def clone(self):
        return MemoryCell(
            identifier=f"{self.identifier}_clone",
            content=self.content,
            location=self.location,
            shadow=self.shadow,
            bridge_id=self.bridge_id,
            weight=self.weight
        )

# ===== MEMORY BANK =====
class MemoryBank:
    def __init__(self):
        self.cells = {}

    def read(self, identifier):
        return self.cells.get(identifier)

    def write(self, identifier, content, location="", shadow=False, bridge_id=None):
        cell = MemoryCell(identifier, content, location, shadow, bridge_id)
        self.cells[identifier] = cell

    def clone_cell(self, identifier):
        original = self.read(identifier)
        if original:
            clone = original.clone()
            self.cells[clone.identifier] = clone
            return clone.identifier
        return None

    def quarantine_cell(self, identifier):
        cell = self.read(identifier)
        if cell:
            cell.shadow = True
            cell.location += " [QUARANTINED]"
            cell.weight = 0.2

# ===== GLYPH HANDLER =====
class GlyphHandler:
    MODE = "emoji"
    GLYPHS = {
        "emoji": {
            "normal": "🟢", "shadow": "🟣", "anomaly": "⚠️", "bridge": "🔗",
            "clone": "📄", "custom": "🌟", "lockdown": "🔴"
        }
    }
    CUSTOM_TRIGGERS = {"MAGIC", "ELEVATE"}

    @staticmethod
    def get_glyph(cell):
        mode = GlyphHandler.MODE
        g = GlyphHandler.GLYPHS[mode]
        if "!!" in cell.content: return g["lockdown"]
        if any(t in cell.content for t in GlyphHandler.CUSTOM_TRIGGERS): return g["custom"]
        if "!" in cell.content: return g["anomaly"]
        if cell.shadow: return g["shadow"]
        if "_clone" in cell.identifier: return g["clone"]
        if cell.bridge_id: return g["bridge"]
        return g["normal"]

# ===== MEMORY TRACKER =====
class MemoryTracker:
    def __init__(self):
        self.history = []

    def log(self, action, cell_id, details=""):
        self.history.append({
            "action": action, "cell_id": cell_id,
            "details": details, "timestamp": time.strftime("%H:%M:%S")
        })

    def get_trace(self):
        return [f"{t['timestamp']} | {t['action']} → {t['cell_id']}: {t['details']}" for t in self.history]

    def analyze_severity(self, cells):
        count = {"normal":0,"warning":0,"lockdown":0}
        for c in cells.values():
            if "!!" in c.content: count["lockdown"]+=1
            elif "!" in c.content: count["warning"]+=1
            else: count["normal"]+=1
        return count

# ===== GOVERNANCE PROTOCOL =====
class Governance:
    def __init__(self):
        self.thresholds = {"quarantine": 0.3, "sync_trust": 0.8, "regenerate": 0.5}

    def approve_action(self, action_type, value):
        th = self.thresholds.get(action_type, 0.5)
        return value <= th if action_type == "quarantine" else value >= th

# ===== ASI AGENT LOGIC =====
class ASIAgent:
    def __init__(self, memory_bank, tracker):
        self.memory_bank = memory_bank
        self.tracker = tracker
        self.guardian_state = True
        self.governance = Governance()

    def fetch_bridge_data(self, bridge_id):
        try:
            if bridge_id.startswith("WX_"):
                loc = bridge_id[3:]
                r = requests.get(f"https://wttr.in/{loc}?format=3", timeout=3)
                return f"🌤️ {r.text}" if r.ok else "⚠️ Weather fetch failed"
            elif bridge_id.startswith("GH_"):
                repo = bridge_id[3:]
                url = f"https://api.github.com/repos/{repo}"
                headers = {"Accept":"application/vnd.github.v3+json"}
                r = requests.get(url, headers=headers, timeout=3)
                if r.ok:
                    d=r.json()
                    return f"🪐 {d['full_name']} ⭐{d['stargazers_count']} 🧾{d['open_issues_count']}"
                return f"⚠️ Repo not found: {repo}"
            elif bridge_id.startswith("AI_"):
                return f"📡 {bridge_id} → Latency: {random.randint(20,80)}ms | Model: Stable"
            return f"🔗 {bridge_id} → Unknown or static bridge"
        except Exception as e:
            return f"🚫 {bridge_id} → Error: {str(e)}"

    def score_risk(self, cell):
        r=0
        if "!!" in cell.content: r+=3
        elif "!" in cell.content: r+=2
        if cell.shadow: r+=1
        if cell.bridge_id and "Unreachable" in self.fetch_bridge_data(cell.bridge_id): r+=2
        return r

    def update_weight(self, cell):
        r = self.score_risk(cell)
        cell.weight = max(0.0, 1.0 - (r * 0.1))

    def execute_write(self, identifier, content, location="", shadow=False, bridge_id=None):
        if not self.guardian_state:
            self.tracker.log("ASI_BLOCKED", identifier, "Write Denied")
            return

        anomaly = "none"
        if "!!" in content: anomaly = "lockdown"
        elif "!" in content: anomaly = "warning"

        self.memory_bank.write(identifier, content, location, shadow, bridge_id)
        cell = self.memory_bank.read(identifier)
        self.update_weight(cell)

        self.tracker.log("ASI_WRITE", identifier, content)
        if anomaly == "warning":
            self.tracker.log("ASI_ALERT", identifier, "⚠️ Soft Warning Triggered")
        elif anomaly == "lockdown":
            self.memory_bank.quarantine_cell(identifier)
            self.guardian_state = False
            self.tracker.log("ASI_LOCKDOWN", identifier, "🚫 Guardian Lockdown + Quarantine")

    def toggle_guardian(self):
        self.guardian_state = not self.guardian_state

# ===== GUI MAGICBOX CORE =====
class MagicBoxGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧠 MagicBox Memory Sentinel")
        self.geometry("1200x800")
        self.configure(bg="#1e1e2f")

        self.memory_bank = MemoryBank()
        self.tracker = MemoryTracker()
        self.asi_agent = ASIAgent(self.memory_bank, self.tracker)
        self.probe_filters = {"WX": True, "GH": True, "AI": True}

        self.create_widgets()
        self.init_probe_chart()
        self.refresh_monitor()
        self.refresh_diagnostics()
        self.refresh_probe_tab()

    def create_widgets(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TLabel", background="#1e1e2f", foreground="#fff", font=("Segoe UI", 12))
        self.style.configure("TButton", background="#5c5c8a", foreground="#fff", padding=6)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(expand=True, fill='both')

        self.reader_tab = ttk.Frame(self.tabs)
        self.writer_tab = ttk.Frame(self.tabs)
        self.cloner_tab = ttk.Frame(self.tabs)
        self.monitor_tab = ttk.Frame(self.tabs)
        self.diagnostic_tab = ttk.Frame(self.tabs)
        self.probe_tab = ttk.Frame(self.tabs)

        self.tabs.add(self.reader_tab, text="🧩 Reader")
        self.tabs.add(self.writer_tab, text="📝 Writer")
        self.tabs.add(self.cloner_tab, text="📄 Cloner")
        self.tabs.add(self.monitor_tab, text="📡 Monitor")
        self.tabs.add(self.diagnostic_tab, text="🚨 Diagnostics")
        self.tabs.add(self.probe_tab, text="🧠 Probe Manager")

        self.output_var = tk.StringVar()
        self.monitor_var = tk.StringVar()
        self.diagnostic_var = tk.StringVar()
        self.probe_var = tk.StringVar()

        # Reader
        ttk.Label(self.reader_tab, text="Memory Snapshot").pack(pady=10)
        ttk.Label(self.reader_tab, textvariable=self.output_var).pack()

        # Writer
        self.id_entry = ttk.Entry(self.writer_tab)
        self.content_entry = ttk.Entry(self.writer_tab)
        self.location_entry = ttk.Entry(self.writer_tab)
        ttk.Label(self.writer_tab, text="ID:").pack()
        self.id_entry.pack()
        ttk.Label(self.writer_tab, text="Content:").pack()
        self.content_entry.pack()
        ttk.Label(self.writer_tab, text="Location:").pack()
        self.location_entry.pack()
        ttk.Button(self.writer_tab, text="🖊️ One-Click Write", command=self.on_write_click).pack(pady=10)
        ttk.Button(self.writer_tab, text="🔄 Toggle ASI Guardian", command=self.toggle_asi).pack()

        # Cloner
        self.clone_entry = ttk.Entry(self.cloner_tab)
        ttk.Label(self.cloner_tab, text="Enter ID to Clone:").pack()
        self.clone_entry.pack()
        ttk.Button(self.cloner_tab, text="📄 One-Click Clone", command=self.on_clone_click).pack(pady=10)

        # Monitor
        ttk.Label(self.monitor_tab, text="Live Feed").pack()
        ttk.Label(self.monitor_tab, textvariable=self.monitor_var).pack()

        # Diagnostics
        ttk.Label(self.diagnostic_tab, text="Diagnostic Console").pack()
        ttk.Label(self.diagnostic_tab, textvariable=self.diagnostic_var, foreground="#ffcc00").pack()
        ttk.Button(self.diagnostic_tab, text="📤 Export Logs", command=self.export_logs).pack()

        # Probe Manager
        ttk.Label(self.probe_tab, text="Live Intelligence Feed").pack()
        ttk.Label(self.probe_tab, textvariable=self.probe_var).pack()
        ttk.Button(self.probe_tab, text="📈 Export Threat Snapshot", command=self.export_snapshot).pack()

    def on_write_click(self):
        id_ = self.id_entry.get()
        content = self.content_entry.get()
        location = self.location_entry.get()
        self.asi_agent.execute_write(id_, content, location)
        self.update_reader_view()

    def on_clone_click(self):
        id_ = self.clone_entry.get()
        clone_id = self.memory_bank.clone_cell(id_)
        self.tracker.log("CLONE", id_, f"New ID: {clone_id}")
        self.update_reader_view()

    def toggle_asi(self):
        self.asi_agent.toggle_guardian()
        status = "ENABLED" if self.asi_agent.guardian_state else "DISABLED"
        messagebox.showinfo("ASI Guardian", f"Agent is now {status}")
        self.tracker.log("ASI_TOGGLE", "Agent", status)

    def update_reader_view(self):
        display = []
        for cell in self.memory_bank.cells.values():
            glyph = GlyphHandler.get_glyph(cell)
            cert = "✅" if cell.verify_certificate() else "❌"
            display.append(f"{glyph} {cell.identifier} → {cell.content} @ {cell.location} | {cert} Weight: {cell.weight:.2f}")
        self.output_var.set("\n".join(display))

    def refresh_monitor(self):
        trace = self.tracker.get_trace()
        self.monitor_var.set("\n".join(trace[-12:]))
        self.after(1000, self.refresh_monitor)

    def refresh_diagnostics(self):
        logs = []
        severity = self.tracker.analyze_severity(self.memory_bank.cells)
        logs.append(f"Severity Totals → 🟢:{severity['normal']} 🟠:{severity['warning']} 🔴:{severity['lockdown']}")
        for cell in self.memory_bank.cells.values():
            tag = cell.identifier
            if "!!" in cell.content:
                logs.append(f"[{tag}] 🔴 LOCKDOWN - Critical anomaly")
            elif "!" in cell.content:
                logs.append(f"[{tag}] 🟠 WARNING - Soft anomaly")
            elif cell.shadow:
                logs.append(f"[{tag}] 🟣 Shadow Cell Active")
            elif any(trigger in cell.content for trigger in GlyphHandler.CUSTOM_TRIGGERS):
                logs.append(f"[{tag}] 🌟 Custom Trigger Active")
            elif cell.bridge_id:
                bridge_response = self.asi_agent.fetch_bridge_data(cell.bridge_id)
                logs.append(f"[{tag}] 🔗 Bridge → {bridge_response}")
        self.diagnostic_var.set("\n".join(logs[-12:]))
        self.after(3000, self.refresh_diagnostics)

    def init_probe_chart(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.probe_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        threading.Thread(target=self.start_live_probe_chart, daemon=True).start()

    def start_live_probe_chart(self):
        while True:
            temps = []
            for cell in self.memory_bank.cells.values():
                if cell.bridge_id and cell.bridge_id.startswith("WX_"):
                    data = self.asi_agent.fetch_bridge_data(cell.bridge_id)
                    try:
                        temp = int("".join(filter(str.isdigit, data)))
                        temps.append(temp)
                    except:
                        continue
            self.ax.clear()
            self.ax.plot(temps, label="🌡️ WX Probe Temp")
            self.ax.legend()
            self.canvas.draw()
            time.sleep(10)

    def refresh_probe_tab(self):
        logs = []
        for cell in self.memory_bank.cells.values():
            if cell.bridge_id:
                data = self.asi_agent.fetch_bridge_data(cell.bridge_id)
                logs.append(f"{cell.identifier} → {data}")
        self.probe_var.set("\n".join(logs[-12:]))
        self.after(5000, self.refresh_probe_tab)

    def export_logs(self):
        trace = self.tracker.get_trace()
        with open("MagicBox_TraceLog.txt", "w") as f:
            f.write("\n".join(trace))
        messagebox.showinfo("Export Complete", "Trace log saved as MagicBox_TraceLog.txt")

    def export_snapshot(self):
        with open("ThreatSnapshot.txt", "w") as f:
            f.write("=== Threat Graph ===\n")
            for cell in self.memory_bank.cells.values():
                if cell.bridge_id:
                    bridge_data = self.asi_agent.fetch_bridge_data(cell.bridge_id)
                    f.write(f"{cell.identifier} → {bridge_data}\n")
            f.write("\n=== Quarantined Cells ===\n")
            for cell in self.memory_bank.cells.values():
                if "QUARANTINED" in cell.location:
                    f.write(f"{cell.identifier} → {cell.content}\n")
        messagebox.showinfo("Snapshot Saved", "ThreatSnapshot.txt exported successfully.")

# ===== LAUNCH SENTINEL DASHBOARD =====
if __name__ == "__main__":
    app = MagicBoxGUI()
    app.mainloop()

