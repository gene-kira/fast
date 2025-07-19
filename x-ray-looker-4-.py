# === Autoloader ===
import subprocess, sys
def install(package): subprocess.check_call([sys.executable, "-m", "pip", "install", package])
for pkg in ["torch", "torchvision", "Pillow", "numpy"]: 
    try: __import__(pkg.lower() if pkg != "Pillow" else "PIL")
    except ImportError: install(pkg)

# === Imports ===
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import torch
from torchvision import transforms

# === Model Loader Stub ===
def load_model():
    try:
        model = torch.load("resnet_model.pth", map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:
        print("⚠️ Error loading model:", e)
        return None

# === Inference Stub ===
def run_inference(pil_img, model, sensitivity):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Match your training config
    ])
    img_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)  # 🧠 Replace this with your actual output format
        # === Dummy Post-Processing ===
        # Stub assumes model returns [N x 5] boxes + scores
        anomalies = []
        if hasattr(outputs, "boxes"):
            for box, score in zip(outputs.boxes, outputs.scores):
                if score.item() > sensitivity / 100:
                    coords = [int(c * 800) for c in box.cpu().numpy()]  # Scale to canvas
                    anomalies.append((tuple(coords), score.item()))
    return anomalies

# === Severity Coloring ===
def get_color(score):
    return "red" if score > 0.8 else "orange" if score > 0.6 else "yellow"

# === GUI ===
class MagicBoxGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧠 MagicBox ResNet Sentinel")
        self.geometry("1200x700")
        self.configure(bg="#1b1b2f")
        self.model = load_model()
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

        ttk.Button(control_frame, text="📁 Load Image", command=self._load_image).pack(pady=5)
        self.scan_var = tk.IntVar()
        ttk.Checkbutton(control_frame, text="🔍 Scan Mode", variable=self.scan_var).pack(pady=5)

        self.sensitivity = ttk.Scale(control_frame, from_=0, to=100, orient="horizontal")
        self.sensitivity.set(50)
        self.sensitivity.pack(pady=5)
        ttk.Label(control_frame, text="Sensitivity").pack()

        ttk.Button(control_frame, text="⚡ Run Scan", command=self._scan_image).pack(pady=5)
        ttk.Button(control_frame, text="📤 Export Log", command=self._export_log).pack(pady=5)
        self.log_box = tk.Text(control_frame, width=40, height=20, bg="#141421", fg="#f0f0f0")
        self.log_box.pack(pady=10)

    def _build_footer(self):
        self.status = tk.Label(self, text="Awaiting anomaly...", font=("Consolas", 12),
                               fg="#f0f0f0", bg="#1b1b2f")
        self.status.pack(pady=10, side=tk.BOTTOM)

    def _load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if path:
            img = Image.open(path).resize((800, 500))
            self.loaded_image = img.copy()
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.status.config(text=f"Loaded: {path.split('/')[-1]}")
            self.log_box.delete("1.0", tk.END)

    def _scan_image(self):
        if self.scan_var.get() and self.loaded_image and self.model:
            sensitivity_val = self.sensitivity.get()
            results = run_inference(self.loaded_image, self.model, sensitivity_val)

            img_overlay = self.loaded_image.copy()
            draw = ImageDraw.Draw(img_overlay)
            self.log_box.delete("1.0", tk.END)

            for box, score in results:
                draw.rectangle(box, outline=get_color(score), width=2)
                self.log_box.insert(tk.END, f"🔎 Anomaly @ {box}, Severity: {score:.2f}\n")

            self.tk_img = ImageTk.PhotoImage(img_overlay)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.status.config(text=f"{len(results)} anomalies detected")
        elif not self.loaded_image:
            self.status.config(text="⚠️ Load an image first.")
        elif not self.model:
            self.status.config(text="⚠️ Model not loaded.")

    def _export_log(self):
        with open("magicbox_resnet_log.txt", "w") as f:
            f.write(self.log_box.get("1.0", tk.END))
        self.status.config(text="🔖 Log saved as magicbox_resnet_log.txt")

if __name__ == "__main__":
    app = MagicBoxGUI()
    app.mainloop()

