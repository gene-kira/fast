import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np

# === Dummy anomaly detection ===
def detect_anomalies(image, sensitivity):
    np.random.seed(42)
    boxes = []
    for _ in range(int(sensitivity / 25)):
        x, y = np.random.randint(50, 700), np.random.randint(50, 400)
        w, h = np.random.randint(30, 80), np.random.randint(30, 80)
        boxes.append((x, y, x + w, y + h))
    return boxes

class MagicBoxGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧙 MagicBox Anomaly Sentinel")
        self.geometry("1000x700")
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
        self.canvas_frame.pack(pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, width=800, height=500,
                                bg="#2e2e38", highlightthickness=2,
                                highlightbackground="#5f27cd")
        self.canvas.pack()

    def _build_controls(self):
        control_frame = tk.Frame(self, bg="#1b1b2f")
        control_frame.pack(pady=10)

        load_btn = ttk.Button(control_frame, text="📁 Load Image", command=self._load_image)
        load_btn.grid(row=0, column=0, padx=5)

        self.scan_var = tk.IntVar()
        scan_toggle = ttk.Checkbutton(control_frame, text="🔍 Scan Mode", variable=self.scan_var)
        scan_toggle.grid(row=0, column=1, padx=5)

        self.sensitivity = ttk.Scale(control_frame, from_=0, to=100, orient="horizontal")
        self.sensitivity.set(50)
        self.sensitivity.grid(row=0, column=2, padx=5)

        sens_label = ttk.Label(control_frame, text="Sensitivity")
        sens_label.grid(row=0, column=3, padx=5)

        scan_btn = ttk.Button(control_frame, text="⚡ Run Scan", command=self._scan_image)
        scan_btn.grid(row=0, column=4, padx=5)

    def _build_footer(self):
        self.status = tk.Label(self, text="Awaiting anomaly...", font=("Consolas", 12),
                               fg="#f0f0f0", bg="#1b1b2f")
        self.status.pack(pady=10)

    def _load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            img = Image.open(file_path).resize((800, 500))
            self.loaded_image = img.copy()
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.status.config(text=f"Loaded: {file_path.split('/')[-1]}")

    def _scan_image(self):
        if self.scan_var.get() and self.loaded_image:
            sensitivity_val = self.sensitivity.get()
            anomalies = detect_anomalies(self.loaded_image, sensitivity_val)

            img_with_boxes = self.loaded_image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            for box in anomalies:
                draw.rectangle(box, outline="red", width=2)

            self.tk_img = ImageTk.PhotoImage(img_with_boxes)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.status.config(text=f"{len(anomalies)} anomaly{'ies' if len(anomalies) != 1 else ''} detected")
        elif not self.loaded_image:
            self.status.config(text="⚠️ Load an image before scanning.")
        else:
            self.status.config(text="Scan Mode is off.")

if __name__ == "__main__":
    app = MagicBoxGUI()
    app.mainloop()

