import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

class MagicBoxGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧙 MagicBox Anomaly Sentinel")
        self.geometry("1000x700")
        self.configure(bg="#1b1b2f")

        self._build_header()
        self._build_canvas()
        self._build_controls()
        self._build_footer()

    def _build_header(self):
        header = tk.Label(self, text="MagicBox Vision Portal", font=("Orbitron", 20), fg="#f0f0f0", bg="#1b1b2f")
        header.pack(pady=10)

    def _build_canvas(self):
        self.canvas_frame = tk.Frame(self, bg="#1b1b2f")
        self.canvas_frame.pack(pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, width=800, height=500, bg="#2e2e38", highlightthickness=2, highlightbackground="#5f27cd")
        self.canvas.pack()

    def _build_controls(self):
        control_frame = tk.Frame(self, bg="#1b1b2f")
        control_frame.pack(pady=10)

        # Load Image Button
        load_btn = ttk.Button(control_frame, text="📁 Load Image", command=self._load_image)
        load_btn.grid(row=0, column=0, padx=5)

        # Scan Toggle
        self.scan_var = tk.IntVar()
        scan_toggle = ttk.Checkbutton(control_frame, text="🔍 Scan Mode", variable=self.scan_var)
        scan_toggle.grid(row=0, column=1, padx=5)

        # Sensitivity Slider
        sensitivity = ttk.Scale(control_frame, from_=0, to=100, orient="horizontal")
        sensitivity.set(50)
        sensitivity.grid(row=0, column=2, padx=5)
        sens_label = ttk.Label(control_frame, text="Sensitivity")
        sens_label.grid(row=0, column=3, padx=5)

    def _build_footer(self):
        self.status = tk.Label(self, text="Awaiting anomaly...", font=("Consolas", 12), fg="#f0f0f0", bg="#1b1b2f")
        self.status.pack(pady=10)

    def _load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            img = Image.open(file_path)
            img = img.resize((800, 500))
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.status.config(text=f"Loaded: {file_path.split('/')[-1]}")

if __name__ == "__main__":
    app = MagicBoxGUI()
    app.mainloop()

