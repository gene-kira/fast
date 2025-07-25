import os, json
from datetime import datetime
import tkinter as tk
from tkinter import ttk

try:
    from system_info import get_full_profile
except ImportError:
    def get_full_profile():
        return {
            "system": {"cpu_model": "Intel i9", "temperature": 55},
            "memory": {"usage_percent": 68, "used_gb": 10, "total_gb": 16},
            "ml": {"inference": "Normal", "confidence": 0.87}
        }

class DashboardController:
    def __init__(self, refresh_interval=10000):
        self.refresh_interval = refresh_interval
        self.profile_cache = {}
        self.callbacks = []
        self.snark_enabled = False
        self.accessibility_mode = False
        self.theme_data = {}
        self.load_gui_theme()
        self.update_profile()

    def register_panel(self, callback):
        self.callbacks.append(callback)

    def update_profile(self):
        profile = get_full_profile()
        self.profile_cache = profile
        self.write_log(profile)
        for cb in self.callbacks:
            cb(profile)
        root.after(self.refresh_interval, self.update_profile)

    def write_log(self, profile):
        try:
            os.makedirs("logs", exist_ok=True)
            path = os.path.join("logs", "performance_history.json")
            entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "profile": profile}
            log = []
            if os.path.exists(path):
                with open(path, "r") as f:
                    log = json.load(f)
            log.append(entry)
            with open(path, "w") as f:
                json.dump(log, f, indent=4)
        except Exception as e:
            print("Log write error:", e)

    def get_previous_profile(self):
        path = "logs/performance_history.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                log = json.load(f)
                return log[-2]["profile"] if len(log) >= 2 else {}
        return {}

    def compare_profiles(self, current, previous):
        return [
            (section, key, previous.get(section, {}).get(key), current[section][key])
            for section in current
            for key in current[section]
            if current[section][key] != previous.get(section, {}).get(key)
        ]

    def toggle_snark(self):
        self.snark_enabled = not self.snark_enabled

    def toggle_accessibility(self):
        self.accessibility_mode = not self.accessibility_mode
        self.theme_data["label_font"] = ("Segoe UI", 14) if self.accessibility_mode else ("Segoe UI", 10)
        self.theme_data["title_font"] = ("Segoe UI", 18, "bold") if self.accessibility_mode else ("Segoe UI", 14, "bold")

    def export_latest_profile(self):
        try:
            with open("latest_export.csv", "w") as f:
                f.write("Section,Key,Value\n")
                for section, data in self.profile_cache.items():
                    for k, v in data.items():
                        f.write(f"{section},{k},{v}\n")
        except Exception as e:
            print("Export error:", e)

    def load_gui_theme(self):
        self.theme_data = {
            "bg": "#1e1e2e",
            "fg": "#f5e0dc",
            "highlight": "#f38ba8",
            "label_font": ("Segoe UI", 10),
            "title_font": ("Segoe UI", 14, "bold")
        }

# 🏁 GUI Bootstrap
root = tk.Tk()
root.title("Butler Diagnostics: MagicBox Edition")
controller = DashboardController()

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", foreground=controller.theme_data["fg"], background=controller.theme_data["bg"],
                font=controller.theme_data["label_font"])
style.configure("TLabelframe", background=controller.theme_data["bg"], foreground=controller.theme_data["highlight"],
                font=controller.theme_data["title_font"])
style.configure("TFrame", background=controller.theme_data["bg"])

def refresh_styles():
    font = controller.theme_data["label_font"]
    style.configure("TLabel", font=font)
    style.configure("TLabelframe", font=controller.theme_data["title_font"])

main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill="both", expand=True)

def toggle_snark():
    controller.toggle_snark()
    snark_button.config(text=f"Snark Mode: {'enabled' if controller.snark_enabled else 'disabled'}")

def toggle_accessibility():
    controller.toggle_accessibility()
    refresh_styles()
    accessibility_button.config(text=f"Accessibility: {'ON' if controller.accessibility_mode else 'OFF'}")

def export_profile():
    controller.export_latest_profile()
    export_button.config(text="Exported!")

# 🎛 Control Buttons
button_frame = ttk.Frame(main_frame)
button_frame.pack(fill="x", pady=5)

snark_button = ttk.Button(button_frame, text="Snark Mode: disabled", command=toggle_snark)
accessibility_button = ttk.Button(button_frame, text="Accessibility: OFF", command=toggle_accessibility)
export_button = ttk.Button(button_frame, text="Export to CSV", command=export_profile)

snark_button.pack(side="left", padx=5)
accessibility_button.pack(side="left", padx=5)
export_button.pack(side="right", padx=5)

# 📦 Panels
class SystemPanel(ttk.LabelFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, text="System Info")
        controller.register_panel(self.update_with_profile)
        self.cpu_label = ttk.Label(self)
        self.temp_label = ttk.Label(self)
        self.cpu_label.pack(anchor="w", padx=5, pady=2)
        self.temp_label.pack(anchor="w", padx=5, pady=2)

    def update_with_profile(self, profile):
        sys = profile.get("system", {})
        self.cpu_label.config(text=f"CPU: {sys.get('cpu_model', 'N/A')}")
        self.temp_label.config(text=f"Temp: {sys.get('temperature', 'N/A')} °C")

class RAMViewer(ttk.LabelFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, text="Memory")
        controller.register_panel(self.update_from_profile)
        self.ram_label = ttk.Label(self)
        self.ram_bar = ttk.Progressbar(self, maximum=100)
        self.ram_label.pack(anchor="w", padx=5, pady=2)
        self.ram_bar.pack(fill="x", padx=5, pady=2)

    def update_from_profile(self, profile):
        mem = profile.get("memory", {})
        used, total = mem.get("used_gb", 0), mem.get("total_gb", 0)
        percent = mem.get("usage_percent", 0)
        self.ram_label.config(text=f"{used} / {total} GB used")
        self.ram_bar.config(value=percent)

class MLPredictorPanel(ttk.LabelFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, text="ML Predictor")
        controller.register_panel(self.update_with_profile)
        self.pred_label = ttk.Label(self)
        self.conf_bar = ttk.Progressbar(self, maximum=100)
        self.pred_label.pack(anchor="w", padx=5, pady=2)
        self.conf_bar.pack(fill="x", padx=5, pady=2)

    def update_with_profile(self, profile):
        ml = profile.get("ml", {})
        line = f"Prediction: {ml.get('inference', 'N/A')}"
        if controller.snark_enabled:
            line += f" 🙄 ({int(ml.get('confidence', 0)*100)}%)"
        self.pred_label.config(text=line)
        self.conf_bar.config(value=int(ml.get("confidence", 0) * 100))

class CommentaryPanel(ttk.LabelFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, text="Butler Commentary")
        controller.register_panel(self.update_with_profile)
        self.textbox = tk.Text(self, wrap="word", height=6, bg="#2e2e3e", fg="#f5e0dc",
                               font=("Segoe UI", 10), borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, command=self.textbox.yview)
        self.textbox.config(yscrollcommand=self.scrollbar.set)
        self.textbox.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def update_with_profile(self, current):
        prev = controller.get_previous_profile()
        diffs = controller.compare_profiles(current, prev)
        self.textbox.delete("1.0", tk.END)
        if not diffs:
            self.textbox.insert(tk.END, "🧘 Butler: System steady. No mood swings today.")
        else:
            for section, key, old, new in diffs:
                line = f"{section} » {key}: {old} → {new}\n"
                if controller.snark_enabled:
                    line = f"🤔 {section.title()} » '{key}' changed from {old} to {new}. Riveting.\n"
                self.textbox.insert(tk.END, line)

# 🧩 Load panels
SystemPanel(main_frame, controller).pack(fill="x", pady=5)
RAMViewer(main_frame, controller).pack(fill="x", pady=5)
MLPredictorPanel(main_frame, controller).pack(fill="x", pady=5)
CommentaryPanel(main_frame, controller).pack(fill="both", pady=5, expand=True)

root.configure(bg=controller.theme_data["bg"])
root.geometry("420x520")
root.mainloop()

