import os, sys, threading, time, uuid
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import psutil
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil requests"])
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import psutil
    import requests

# 🎨 Theme Settings
COLOR_SCHEMES = {
    "Void Mode":    {"bg": "#0c0c14", "fg": "#eeeeee", "hl": "#00d4ff"},
    "Cyber Violet": {"bg": "#2a003f", "fg": "#eeeeff", "hl": "#c92c92"},
    "Solarize":     {"bg": "#002b36", "fg": "#fdf6e3", "hl": "#b58900"},
    "High Contrast":{"bg": "#000000", "fg": "#ffffff", "hl": "#ff0000"}
}

CURRENT_SCHEME = "Void Mode"
BG = COLOR_SCHEMES[CURRENT_SCHEME]["bg"]
FG = COLOR_SCHEMES[CURRENT_SCHEME]["fg"]
HL = COLOR_SCHEMES[CURRENT_SCHEME]["hl"]
TTL = 60
REFRESH_INTERVAL = 30

class MagicBoxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧿 MagicBox Network Guardian")
        self.root.configure(bg=BG)
        self.root.geometry("1200x700")

        self.file_tracker = {}
        self.connection_memory = []
        self.mode = tk.StringVar(value="stealth")
        self.live_mode = tk.BooleanVar(value=True)

        self.build_gui()
        self.refresh_connections()
        self.start_file_tracker()
        self.start_live_refresh()

    def apply_color_scheme(self, scheme_name):
        global BG, FG, HL
        if scheme_name in COLOR_SCHEMES:
            BG = COLOR_SCHEMES[scheme_name]["bg"]
            FG = COLOR_SCHEMES[scheme_name]["fg"]
            HL = COLOR_SCHEMES[scheme_name]["hl"]
            self.root.configure(bg=BG)
            
            style = ttk.Style()
            style.theme_use("clam")
            style.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
            style.map(".", foreground=[('selected', HL)], background=[('selected', BG)])

            # Update treeview style
            style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])
            style.configure("Treeview", background=BG, foreground=FG, fieldbackground=BG)
            style.map('Treeview', background=[('selected', HL)])
            
            self.status_label.config(foreground=HL)

            # Update all widgets
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame) or isinstance(widget, ttk.LabelFrame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Label) or isinstance(child, ttk.Radiobutton) or isinstance(child, ttk.Checkbutton) or isinstance(child, ttk.Button):
                            child.configure(background=BG, foreground=FG)
                elif isinstance(widget, ttk.Combobox):
                    widget.configure(readonlybackground=BG, selectbackground=HL, fieldbackground=BG)
            
            self.scheme_box.current(list(COLOR_SCHEMES.keys()).index(scheme_name))

    def build_gui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))

        ttk.Label(self.root, text="MagicBox Network & Vaporization Monitor", font=("Segoe UI", 14)).pack(pady=10)

        mode_frame = ttk.LabelFrame(self.root, text="💣 Self-Destruct Style")
        mode_frame.pack(fill="x", padx=10)
        ttk.Radiobutton(mode_frame, text="🕶️ Stealth", variable=self.mode, value="stealth").pack(side="left", padx=10)
        ttk.Radiobutton(mode_frame, text="💥 Dramatic", variable=self.mode, value="dramatic").pack(side="left", padx=10)

        live_frame = ttk.Frame(self.root)
        live_frame.pack(pady=5)
        ttk.Checkbutton(live_frame, text="Enable Live Scan", variable=self.live_mode).pack()

        scheme_frame = ttk.LabelFrame(self.root, text="🎨 Color Theme")
        scheme_frame.pack(fill="x", padx=10)
        self.scheme_box = ttk.Combobox(scheme_frame, values=list(COLOR_SCHEMES.keys()), state="readonly")
        self.scheme_box.set(CURRENT_SCHEME)
        self.scheme_box.pack()
        self.scheme_box.bind("<<ComboboxSelected>>", lambda e: self.apply_color_scheme(self.scheme_box.get()))

        tree_frame = ttk.Frame(self.root)
        tree_frame.pack(expand=True, fill="both", padx=10)
        self.tree_normal = self.create_tree(tree_frame, "✅ Normal Connections")
        self.tree_abnormal = self.create_tree(tree_frame, "❗ Abnormal Connections")

        mem_frame = ttk.LabelFrame(self.root, text="🧠 Connection History")
        mem_frame.pack(fill="both", expand=True, padx=10, pady=5)
        cols = ("Time", "Direction", "IP", "MAC", "Country", "Info")
        self.tree_memory = ttk.Treeview(mem_frame, columns=cols, show="headings", height=6)
        for col in cols:
            self.tree_memory.heading(col, text=col)
            self.tree_memory.column(col, width=180)
        self.tree_memory.pack(fill="both", expand=True)

        ttk.Button(self.root, text="📁 Monitor Outbound File", command=self.choose_file).pack(pady=5)
        self.status_label = ttk.Label(self.root, text="🟢 System Ready", font=("Segoe UI", 10), foreground=HL)
        self.status_label.pack(pady=5)

    def create_tree(self, parent, label):
        frame = ttk.LabelFrame(parent, text=label)
        frame.pack(side="left", expand=True, fill="both", padx=5)
        cols = ("IP", "MAC", "Country", "Info")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=10)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=180)
        tree.pack(expand=True, fill="both")
        return tree

    def get_mac(self):
        try:
            mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
            return ":".join([mac[i:i+2] for i in range(0, 12, 2)]).upper()
        except:
            return "Unavailable"

    def get_country(self, ip):
        try:
            r = requests.get(f"https://ipinfo.io/{ip}/json", timeout=5)
            return r.json().get("country", "Unknown")
        except:
            return "Unknown"

    def refresh_connections(self):
        self.tree_normal.delete(*self.tree_normal.get_children())
        self.tree_abnormal.delete(*self.tree_abnormal.get_children())

        connections = psutil.net_connections(kind='inet')
        mac = self.get_mac()

        for conn in connections:
            if conn.status != "ESTABLISHED":
                continue
            try:
                ip = conn.raddr.ip if conn.raddr else conn.laddr.ip
                port = conn.raddr.port if conn.raddr else conn.laddr.port
                country = self.get_country(ip)
                info = f"{conn.status} | Port: {port}"
                direction = "Outgoing" if conn.raddr else "Incoming"
                timestamp = time.strftime("%H:%M:%S")

                record = {
                    "time": timestamp,
                    "direction": direction,
                    "ip": ip,
                    "mac": mac,
                    "country": country,
                    "info": info
                }
                self.connection_memory.append(record)

                values = (ip, mac, country, info)
                tree_target = self.tree_normal if country not in ("Unknown", "Private") else self.tree_abnormal
                tree_target.insert("", "end", values=values)
                self.tree_memory.insert("", "end", values=(timestamp, direction, ip, mac, country, info))
            except:
                continue

        self.status_label.config(text="✅ Connections refreshed")

    def choose_file(self):
        path = filedialog.askopenfilename()
        if path:
            fid = os.path.basename(path)
            self.file_tracker[fid] = {
                "path": path,
                "timestamp": time.time()
            }
            self.status_label.config(text=f"🕑 Tracking: {fid}")

    def start_file_tracker(self):
        def loop():
            while True:
                expired = []
                for fid, data in list(self.file_tracker.items()):
                    if time.time() - data["timestamp"] > TTL:
                        try:
                            if os.path.exists(data["path"]):
                                os.remove(data["path"])
                                self.status_label.config(text=f"💣 {fid} vaporized")
                                if self.mode.get() == "dramatic":
                                    messagebox.showinfo("💥 Vaporized", f"{fid} vanished into ether.")
                        except:
                            self.status_label.config(text=f"❌ Failed to vaporize: {fid}")
                        expired.append(fid)
                for fid in expired:
                    del self.file_tracker[fid]
                time.sleep(1)
        threading.Thread(target=loop, daemon=True).start()

    def start_live_refresh(self):
        def loop():
            while True:
                if self.live_mode.get():
                    self.refresh_connections()
                time.sleep(REFRESH_INTERVAL)
        threading.Thread(target=loop, daemon=True).start()

# 🧿 Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxApp(root)
    root.mainloop()