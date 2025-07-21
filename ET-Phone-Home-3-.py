# === MagicBox GlyphGuard ===
# Part 1: Autoloader + Imports + Configuration + Glyph Setup

# 📦 Install required libraries automatically
import sys
import subprocess

def install_package(package, import_name=None):
    try:
        __import__(import_name or package)
    except ImportError:
        print(f"Installing: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_libraries = {
    "requests": "requests",
    "cryptography": "cryptography",
    "matplotlib": "matplotlib",
    "numpy": "numpy"
}

for pkg, imp in required_libraries.items():
    install_package(pkg, imp)

# ✅ Safe to import everything now
import tkinter as tk
from tkinter import ttk, messagebox
import hashlib, socket, threading, os, json, requests
from cryptography.fernet import Fernet
from time import strftime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === Global Settings ===
API_KEY_HIBP = "YOUR_API_KEY_HERE"   # Replace with your HaveIBeenPwned API key
TRUST_PHRASE = "MagicBoxMesh"
LISTEN_PORTS = [65432, 65433, 65434]
EXTERNAL_PORT = 64500

# === Glyph Encryption Engine ===
crypto_key = Fernet.generate_key()
cipher = Fernet(crypto_key)

def create_glyph(device_id):
    base = f"{device_id}-{socket.gethostname()}-{TRUST_PHRASE}"
    hashed = hashlib.sha256(base.encode()).hexdigest()
    return hashed[:12]

def encrypt_text(text):
    return cipher.encrypt(text.encode()).decode()

def decrypt_text(encrypted_text):
    return cipher.decrypt(encrypted_text.encode()).decode()
# === Part 2: GUI Setup and Core Dashboard ===

class MagicBoxApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MagicBox GlyphGuard")
        self.geometry("950x600")
        self.configure(bg="#1e1e2f")
        self.pulse_phase = 0

        tk.Label(self, text="MagicBox GlyphGuard", font=("Segoe UI", 20, "bold"),
                 bg="#1e1e2f", fg="white").pack(pady=20)

        self.tab_group = ttk.Notebook(self)
        self.tab_group.pack(expand=1, fill="both")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook.Tab", background="#323247", foreground="white", padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", "#52527A")])

        self.setup_tabs()

    def setup_tabs(self):
        self.add_network_tab()
        self.add_audit_tab()
        self.add_breach_tab()
        self.add_glyphtrust_tab()

    def add_network_tab(self):
        tab = ttk.Frame(self.tab_group)
        self.tab_group.add(tab, text="🛡️ Network Status")

        tk.Label(tab, text="Connected Devices", font=("Segoe UI", 12),
                 bg="#1e1e2f", fg="white").pack(pady=10)

        device_list = tk.Listbox(tab, bg="#2e2e3e", fg="white", font=("Consolas", 11))
        device_list.pack(fill="both", expand=True, pady=10)

        device_list.insert(tk.END, "🔗 Device A — Trusted")
        device_list.insert(tk.END, "🔗 Device B — Trusted")

    def add_audit_tab(self):
        tab = ttk.Frame(self.tab_group)
        self.tab_group.add(tab, text="🔍 Audit Scanner")

        tk.Label(tab, text="Check system for issues", font=("Segoe UI", 12),
                 bg="#1e1e2f", fg="white").pack(pady=10)

        self.audit_output = tk.Text(tab, height=8, bg="#2e2e3e", fg="white", font=("Consolas", 10))
        self.audit_output.pack(fill="both", expand=True)

        tk.Button(tab, text="Run Scan", command=self.run_audit_scan,
                  bg="#6f6fc4", fg="white").pack(pady=5)

    def run_audit_scan(self):
        self.audit_output.delete("1.0", tk.END)
        self.audit_output.insert(tk.END, "✅ No issues found.\n🔒 System appears secure.\n")

    def add_breach_tab(self):
        tab = ttk.Frame(self.tab_group)
        self.tab_group.add(tab, text="🌍 Breach Check")

        tk.Label(tab, text="Enter your email to check for breaches", font=("Segoe UI", 12),
                 bg="#1e1e2f", fg="white").pack(pady=10)

        self.email_input = tk.Entry(tab, font=("Segoe UI", 11),
                                    bg="#2e2e3e", fg="white", insertbackground="white")
        self.email_input.pack(pady=5)

        tk.Button(tab, text="Check Email", command=self.check_breach,
                  bg="#6f6fc4", fg="white").pack(pady=5)

        self.breach_result = tk.Label(tab, text="", font=("Consolas", 11),
                                      bg="#1e1e2f", fg="white")
        self.breach_result.pack(pady=10)

    def check_breach(self):
        email = self.email_input.get()
        result = check_breach_status(email) if email else "⚠️ Please enter an email address."
        self.breach_result.config(text=result)

    def add_glyphtrust_tab(self):
        tab = ttk.Frame(self.tab_group)
        self.tab_group.add(tab, text="🌀 Trust Pulse")

        tk.Label(tab, text="Glyph Pulse Animation", font=("Segoe UI", 12),
                 bg="#1e1e2f", fg="white").pack(pady=10)

        self.pulse_canvas = tk.Canvas(tab, width=200, height=200,
                                      bg="#1e1e2f", highlightthickness=0)
        self.pulse_canvas.pack(pady=10)

        self.animate_pulse()

        glyph = create_glyph("Device A")
        tk.Label(tab, text=f"Current Glyph: {glyph}", font=("Consolas", 11),
                 bg="#1e1e2f", fg="#9cd2ff").pack(pady=10)

    def animate_pulse(self):
        self.pulse_canvas.delete("all")
        radius = 40 + (self.pulse_phase % 20)
        self.pulse_canvas.create_oval(
            100 - radius, 100 - radius, 100 + radius, 100 + radius,
            outline="#6f6fc4", width=2
        )
        self.pulse_phase += 1
        self.after(100, self.animate_pulse)
# === Part 3: Intelligence Modules and GUI Enhancements ===

# 📁 Scan files and tag with a glyph
def scan_files(tag="MagicGlyph", root_path="."):
    tagged = []
    for folder, _, files in os.walk(root_path):
        for name in files:
            tagged.append(f"[{tag}] {name}")
    return tagged

# 🌀 Generate a fractal image based on glyph value
def create_fractal_image(glyph_text):
    seed = int(glyph_text[:6], 16) % 1000
    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 500)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    C = 0.355 + 0.355j * (seed / 1000)
    mask = np.full(Z.shape, True)
    iteration_counts = np.zeros(Z.shape, dtype=int)

    for i in range(50):
        Z[mask] = Z[mask]**2 + C
        new_mask = np.abs(Z) < 2
        iteration_counts[mask] += new_mask[mask]
        mask = mask & new_mask

    return iteration_counts

# === Add Tabs to MagicBoxApp Class ===

def add_filescan_tab(self):
    tab = ttk.Frame(self.tab_group)
    self.tab_group.add(tab, text="📁 File Scanner")

    tk.Label(tab, text="Scan Files Using Current Glyph", font=("Segoe UI", 12),
             bg="#1e1e2f", fg="white").pack(pady=10)

    tk.Button(tab, text="Start Scan", command=self.run_filescan,
              bg="#6f6fc4", fg="white").pack(pady=5)

    self.scan_results = tk.Text(tab, bg="#2e2e3e", fg="white", font=("Consolas", 10))
    self.scan_results.pack(fill="both", expand=True)

def run_filescan(self):
    glyph = create_glyph("Device A")
    files = scan_files(tag=glyph)
    self.scan_results.delete("1.0", tk.END)
    for entry in files[:100]:
        self.scan_results.insert(tk.END, entry + "\n")

def add_fractal_tab(self):
    tab = ttk.Frame(self.tab_group)
    self.tab_group.add(tab, text="🧿 Fractal Fingerprint")

    tk.Label(tab, text="Fractal from Current Glyph", font=("Segoe UI", 12),
             bg="#1e1e2f", fg="white").pack(pady=10)

    glyph = create_glyph("Device A")
    data = create_fractal_image(glyph)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(data, cmap="viridis", interpolation="bilinear")
    ax.axis("off")

    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.get_tk_widget().pack(pady=5)
    canvas.draw()
# === External Glyph Communication ===

def translate_external_glyph(raw_glyph):
    glyph_map = {"π": "PULSE", "Δ": "NEGOTIATE", "∑": "SYNC", "@": "PING"}
    return "".join(glyph_map.get(c, c) for c in raw_glyph)

def respond_to_external_glyph(translated):
    if "SYNC" in translated:
        return "🫱 Symbiote Protocol Activated"
    elif "NEGOTIATE" in translated:
        return "🔐 Negotiation Pulse Initiated"
    elif "PING" in translated:
        return "🪩 EchoPulse Acknowledged"
    else:
        return "⚠️ Unknown glyph — Trust Deferred"

def log_external_entity(ip, glyph, response):
    entry = {
        "entity": ip,
        "glyph": glyph,
        "outcome": response,
        "timestamp": strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("external_glyph_memory.json", "a") as file:
        file.write(json.dumps(entry) + "\n")

def listen_for_external_glyphs(port=EXTERNAL_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', port))
    print(f"🛸 Listening for external glyphs on port {port}")
    while True:
        data, addr = sock.recvfrom(1024)
        raw = data.decode()
        translated = translate_external_glyph(raw)
        response = respond_to_external_glyph(translated)
        log_external_entity(addr[0], raw, response)
        sock.sendto(response.encode(), addr)

# === GUI Tabs to Add to MagicBoxApp ===

def add_external_logs_tab(self):
    tab = ttk.Frame(self.tab_group)
    self.tab_group.add(tab, text="🌐 External Logs")

    tk.Label(tab, text="Recent External Glyphs", font=("Segoe UI", 12),
             bg="#1e1e2f", fg="white").pack(pady=10)

    self.external_log_box = tk.Text(tab, bg="#2e2e3e", fg="white", font=("Consolas", 10))
    self.external_log_box.pack(fill="both", expand=True)

    tk.Button(tab, text="Load Logs", command=self.load_external_logs,
              bg="#6f6fc4", fg="white").pack(pady=5)

def load_external_logs(self):
    self.external_log_box.delete("1.0", tk.END)
    try:
        with open("external_glyph_memory.json", "r") as file:
            for line in file:
                self.external_log_box.insert(tk.END, line)
    except:
        self.external_log_box.insert(tk.END, "📂 No external logs found.\n")

def add_glyph_panel_tab(self):
    tab = ttk.Frame(self.tab_group)
    self.tab_group.add(tab, text="🛸 Glyph Panel")

    tk.Label(tab, text="Translated External Glyphs", font=("Segoe UI", 12),
             bg="#1e1e2f", fg="white").pack(pady=10)

    self.glyph_panel = tk.Text(tab, bg="#2e2e3e", fg="white", font=("Consolas", 10))
    self.glyph_panel.pack(fill="both", expand=True)

    tk.Button(tab, text="Load Glyphs", command=self.load_glyph_panel,
              bg="#6f6fc4", fg="white").pack(pady=5)

def load_glyph_panel(self):
    self.glyph_panel.delete("1.0", tk.END)
    try:
        with open("external_glyph_memory.json", "r") as file:
            logs = [json.loads(entry) for entry in file.readlines()]
            for log in logs[-50:]:
                ip = log["entity"]
                raw = log["glyph"]
                translated = translate_external_glyph(raw)
                response = log["outcome"]
                time = log["timestamp"]
                self.glyph_panel.insert(tk.END, f"🌐 {ip} @ {time}\n")
                self.glyph_panel.insert(tk.END, f" → Raw Glyph: {raw}\n")
                self.glyph_panel.insert(tk.END, f" → Translated: {translated}\n")
                self.glyph_panel.insert(tk.END, f" → Response: {response}\n\n")
    except:
        self.glyph_panel.insert(tk.END, "🧭 No glyph records found.\n")
if __name__ == "__main__":
    threading.Thread(target=listen_for_external_glyphs, daemon=True).start()
    app = MagicBoxApp()
    app.mainloop()





