import os, json, sys
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
        self.theme = "magicbox"
        self.load_gui_theme()
        self.update_profile()

    def register_panel(self, update_callback):
        self.callbacks.append(update_callback)

    def update_profile(self):
        profile = get_full_profile()
        self.profile_cache = profile
        self.write_log(profile)
        for callback in self.callbacks:
            callback(profile)

        # Schedule next update
        root.after(self.refresh_interval, self.update_profile)

    def write_log(self, profile):
        try:
            os.makedirs("logs", exist_ok=True)
            log_path = os.path.join("logs", "performance_history.json")
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "profile": profile
            }
            log = []
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    log = json.load(f)
            log.append(entry)
            with open(log_path, "w") as f:
                json.dump(log, f, indent=4)
        except Exception as e:
            print("Log write error:", e)

    def load_gui_theme(self):
        self.theme_data = {
            "bg": "#1e1e2e",
            "fg": "#f5e0dc",
            "highlight": "#f38ba8",
            "label_font": ("Segoe UI", 11),
            "title_font": ("Segoe UI", 14, "bold")
        }

class SystemPanel(ttk.LabelFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, text="System Info")
        self.controller = controller
        controller.register_panel(self.update_with_profile)

        self.cpu_label = ttk.Label(self)
        self.temp_label = ttk.Label(self)
        self.cpu_label.pack(anchor="w", padx=5, pady=2)
        self.temp_label.pack(anchor="w", padx=5, pady=2)

    def update_with_profile(self, profile_data):
        sysinfo = profile_data.get("system", {})
        self.cpu_label.config(text=f"CPU: {sysinfo.get('cpu_model', 'N/A')}")
        self.temp_label.config(text=f"Temp: {sysinfo.get('temperature', 'N/A')} °C")

class RAMViewer(ttk.LabelFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, text="Memory")
        self.controller = controller
        controller.register_panel(self.update_from_profile)

        self.ram_label = ttk.Label(self)
        self.ram_bar = ttk.Progressbar(self, maximum=100)
        self.ram_label.pack(anchor="w", padx=5, pady=2)
        self.ram_bar.pack(fill="x", padx=5, pady=2)

    def update_from_profile(self, profile_data):
        mem = profile_data.get("memory", {})
        used = mem.get("used_gb", 0)
        total = mem.get("total_gb", 0)
        percent = mem.get("usage_percent", 0)
        self.ram_label.config(text=f"{used} / {total} GB used")
        self.ram_bar.config(value=percent)

class MLPredictorPanel(ttk.LabelFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, text="ML Predictor")
        self.controller = controller
        controller.register_panel(self.update_with_profile)

        self.pred_label = ttk.Label(self)
        self.conf_bar = ttk.Progressbar(self, maximum=100)
        self.pred_label.pack(anchor="w", padx=5, pady=2)
        self.conf_bar.pack(fill="x", padx=5, pady=2)

    def update_with_profile(self, profile_data):
        ml = profile_data.get("ml", {})
        self.pred_label.config(text=f"Prediction: {ml.get('inference', 'N/A')}")
        self.conf_bar.config(value=int(ml.get("confidence", 0) * 100))

# 🌟 Launch MagicBox GUI
root = tk.Tk()
root.title("Butler Diagnostics: MagicBox Edition")
controller = DashboardController()

# Apply theme globally
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", foreground=controller.theme_data["fg"], background=controller.theme_data["bg"], font=controller.theme_data["label_font"])
style.configure("TLabelframe", background=controller.theme_data["bg"], foreground=controller.theme_data["highlight"], font=controller.theme_data["title_font"])
style.configure("TFrame", background=controller.theme_data["bg"])

main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill="both", expand=True)

panels = [
    SystemPanel(main_frame, controller),
    RAMViewer(main_frame, controller),
    MLPredictorPanel(main_frame, controller)
]

for panel in panels:
    panel.pack(fill="x", pady=5)

root.configure(bg=controller.theme_data["bg"])
root.geometry("400x300")
root.mainloop()

