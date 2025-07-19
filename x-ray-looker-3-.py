# === 🧙 AutoLoader Start ===
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required libraries
required = ["Pillow", "numpy"]

for pkg in required:
    try:
        __import__(pkg.lower() if pkg != "Pillow" else "PIL")
    except ImportError:
        install(pkg)
# === 🧙 AutoLoader End ===

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np

# === Dummy Anomaly Detection ===
def detect_anomalies(image, sensitivity):
    np.random.seed(42)
    boxes = []
    for _ in range(int(sensitivity / 25)):
        x, y = np.random.randint(50, 700), np.random.randint(50, 400)
        w, h = np.random.randint(30, 80), np.random.randint(30, 80)
        score = np.random.uniform(0.4, 0.95)
        boxes.append(((x, y, x + w, y + h), score))
    return boxes

def get_color(score):
    if score > 0.8:
        return "red"
    elif score > 0.6:
        return "orange"
    else:
        return "yellow"

class MagicBoxGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧙 MagicBox Anomaly Sentinel")
        self.geometry("1200x700")
        self.configure(bg="#1b1b2f")

        self.loaded_image = None
        self.tk_img = None

        self._build_header()
        self._build_canvas()
        self._build_controls()
        self._build_footer()

    def _build_header(self):
        header = tk.Label(self, text="MagicBox Vision Portal", font=("Orbitron", 20),
                          fg="#f0f0f0", bg="#1b1b2f")
        header.pack(pady=10)

    def _build_canvas(self):
        self.canvas_frame = tk.Frame(self, bg="#1b1b2f")
        self.canvas_frame.pack(pady=10, side=tk.LEFT)

        self.canvas = tk.Canvas(self.canvas_frame, width=800, height=500,
                                bg="#2e2e38", highlightthickness=2,
                                highlightbackground="#5f27cd")
        self.canvas.pack()

    def _build_controls(self):
        control_frame = tk.Frame(self, bg="#1b1b2f")
        control_frame.pack(pady=10, side=tk.LEFT, fill=tk.Y)

        load_btn = ttk.Button(control_frame, text="📁 Load Image", command=self._load_image)
        load_btn.pack(pady=5)

        self.scan_var = tk.IntVar()
        scan_toggle = ttk.Checkbutton(control_frame, text="🔍 Scan Mode", variable=self.scan_var)
        scan_toggle.pack(pady=5)

        self.sensitivity = ttk.Scale(control_frame, from_=0, to=100, orient="horizontal")
        self.sensitivity.set(50)
        self.sensitivity.pack(pady=5)

        sens_label = ttk.Label(control_frame, text="Sensitivity")
        sens_label.pack()

        scan_btn = ttk.Button(control_frame, text="⚡ Run Scan", command=self._scan_image)
        scan_btn.pack(pady=5)

        export_btn = ttk.Button(control_frame, text="📤 Export Log", command=self._export_log)
        export_btn.pack(pady=5)

        self.log_box = tk.Text(control_frame, width=40, height=20, bg="#141421", fg="#f0f0f0")
        self.log_box.pack(pady=10)

    def _build_footer(self):
        self.status = tk.Label(self, text="Awaiting anomaly...", font=("Consolas", 12),
                               fg="#f0f0f0", bg="#1b1b2f")
        self.status.pack(pady=10, side=tk.BOTTOM)

    def _load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            img = Image.open(file_path).resize((800, 500))
            self.loaded_image = img.copy()
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.status.config(text=f"Loaded: {file_path.split('/')[-1]}")
            self.log_box.delete("1.0", tk.END)

    def _scan_image(self):
        if self.scan_var.get() and self.loaded_image:
            sensitivity_val = self.sensitivity.get()
            results = detect_anomalies(self.loaded_image, sensitivity_val)

            img_with_overlay = self.loaded_image.copy()
            draw = ImageDraw.Draw(img_with_overlay)

            self.log_box.delete("1.0", tk.END)
            for box, score in results:
                color = get_color(score)
                draw.rectangle(box, outline=color, width=2)
                self.log_box.insert(tk.END, f"🔍 Anomaly @ {box}, Severity: {score:.2f}\n")

            self.tk_img = ImageTk.PhotoImage(img_with_overlay)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.status.config(text=f"{len(results)} anomalies detected")

        elif not self.loaded_image:
            self.status.config(text="⚠️ Load an image first.")
        else:
            self.status.config(text="Scan Mode is off.")

    def _export_log(self):
        with open("magicbox_log.txt", "w") as f:
            f.write(self.log_box.get("1.0", tk.END))
        self.status.config(text="🔖 Log saved as magicbox_log.txt")

if __name__ == "__main__":
    app = MagicBoxGUI()
    app.mainloop()

