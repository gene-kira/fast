import tkinter as tk
from tkinter import ttk, messagebox
import platform, json

# === Autoload WMI ===
try:
    import wmi
except ImportError:
    messagebox.showerror("Missing Dependency", "WMI module not found.\nRun: pip install wmi in your command prompt.")
    raise SystemExit

# === RAM Profiler ===
def get_memory_slots():
    c = wmi.WMI()
    memory_devices = c.Win32_PhysicalMemory()
    slot_data = []

    for i, device in enumerate(memory_devices):
        slot_data.append({
            "slot": f"DIMM {i+1}",
            "channel": "A" if i % 2 == 0 else "B",
            "rank": "Dual" if int(device.Speed) > 3000 else "Single",
            "speed": str(device.Speed),
            "latency": f"CL{device.Speed // 200}",
            "voltage": str(device.ConfiguredVoltage),
            "manufacturer": device.Manufacturer.strip()
        })

    while len(slot_data) < 4:
        slot_data.append({
            "slot": f"DIMM {len(slot_data)+1}",
            "channel": "-",
            "rank": "-",
            "speed": "-",
            "latency": "-",
            "voltage": "-",
            "manufacturer": "-"
        })

    return slot_data

# === Bandwidth Health Scoring ===
def calculate_bandwidth_score(slots):
    total_speed = 0
    active_slots = 0
    ranks = {"Single": 0, "Dual": 0}
    channels = {"A": 0, "B": 0}

    for s in slots:
        if s["speed"].isdigit():
            total_speed += int(s["speed"])
            active_slots += 1
            ranks[s["rank"]] += 1
            channels[s["channel"]] += 1

    avg_speed = total_speed / active_slots if active_slots else 0
    balance_bonus = 0
    if channels["A"] == channels["B"]:
        balance_bonus += 10
    if ranks["Dual"] >= 2:
        balance_bonus += 10

    score = int((avg_speed / 3200) * 60 + balance_bonus)
    return min(score, 100)

# === GUI ===
class RAMViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MagicBox RAM Suite")
        self.geometry("720x450")
        self.configure(bg="#2d2d30")

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#2d2d30")
        self.style.configure("Header.TLabel", background="#2d2d30", foreground="#89bdff", font=("Segoe UI", 16, "bold"))
        self.style.configure("Slot.TLabelframe", background="#444", foreground="white", font=("Segoe UI", 10, "bold"))
        self.style.configure("Slot.TLabel", background="#444", foreground="#eee", font=("Segoe UI", 9))

        ttk.Label(self, text="MagicBox Memory Diagnostics", style="Header.TLabel").pack(pady=10)

        self.perf_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Performance Mode", variable=self.perf_var).pack()

        self.score_label = ttk.Label(self, text="", style="Header.TLabel")
        self.score_label.pack(pady=5)

        self.slot_frame = ttk.Frame(self)
        self.slot_frame.pack()

        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(pady=10)
        ttk.Button(self.button_frame, text="Refresh", command=self.refresh_data).grid(row=0, column=0, padx=10)
        ttk.Button(self.button_frame, text="Export to JSON", command=self.export_json).grid(row=0, column=1, padx=10)

        self.slot_data = []
        self.refresh_data()

    def refresh_data(self):
        self.slot_data = get_memory_slots()
        score = calculate_bandwidth_score(self.slot_data)
        snark = self.get_snark(score)
        self.score_label.config(text=f"Estimated Bandwidth Health: {score}/100 — {snark}")
        self.render_slots()

    def render_slots(self):
        for widget in self.slot_frame.winfo_children():
            widget.destroy()

        for i, slot in enumerate(self.slot_data):
            frame = ttk.Labelframe(self.slot_frame, text=slot["slot"], style="Slot.TLabelframe")
            frame.grid(row=0, column=i, padx=10, pady=5, ipadx=10, ipady=10)

            ttk.Label(frame, text=f"Channel: {slot['channel']}", style="Slot.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(frame, text=f"Rank: {slot['rank']}", style="Slot.TLabel").grid(row=1, column=0, sticky="w")
            ttk.Label(frame, text=f"Speed: {slot['speed']} MHz", style="Slot.TLabel").grid(row=2, column=0, sticky="w")
            ttk.Label(frame, text=f"Latency: {slot['latency']}", style="Slot.TLabel").grid(row=3, column=0, sticky="w")

            tip = f"Voltage: {slot['voltage']}V\nManufacturer: {slot['manufacturer']}"
            frame.bind("<Enter>", lambda e, msg=tip: self.show_tooltip(msg))
            frame.bind("<Leave>", lambda e: self.hide_tooltip())

    def get_snark(self, score):
        if not self.perf_var.get():
            return ""
        if score >= 80:
            return "Your layout is pristine. The Butler tips his hat."
        elif score >= 50:
            return "Acceptable... but the Butler glances at your BIOS."
        else:
            return "The Butler frowns. Your setup offends proper etiquette."

    def show_tooltip(self, msg):
        self.tooltip = tk.Toplevel(self)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{self.winfo_pointerx()+10}+{self.winfo_pointery()+10}")
        tk.Label(self.tooltip, text=msg, background="#333", foreground="white",
                 relief="solid", borderwidth=1, font=("Segoe UI", 9)).pack()

    def hide_tooltip(self):
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()

    def export_json(self):
        try:
            with open("ram_diagnostics.json", "w") as f:
                json.dump(self.slot_data, f, indent=4)
            messagebox.showinfo("Export Complete", "Diagnostics saved to ram_diagnostics.json")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

# === Launch ===
if __name__ == "__main__":
    if platform.system() != "Windows":
        messagebox.showwarning("Unsupported OS", "This suite is made for Windows.")
        raise SystemExit
    RAMViewer().mainloop()

