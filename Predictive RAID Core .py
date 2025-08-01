# ========== 📦 Autoload & Setup ==========
try:
    import tkinter as tk
    from tkinter import messagebox
    import random
    from collections import deque, defaultdict
except ImportError as e:
    print("Missing library:", e)
    raise SystemExit("Please ensure all libraries are installed.")

# ========== 🔁 Predictive RAID Core ==========
class PredictiveRaid:
    def __init__(self, disks=4, cache_size=10):
        self.disks = [defaultdict(str) for _ in range(disks)]
        self.cache = deque(maxlen=cache_size)
        self.read_history = deque(maxlen=50)
        self.predictions = defaultdict(int)

    def write(self, block_id, data):
        disk_id = block_id % len(self.disks)
        parity = self._calculate_parity(data)
        self.disks[disk_id][block_id] = data
        self.cache.append((block_id, data))
        return f"Write ✔️ Block {block_id} → Disk {disk_id}, Parity: {parity}"

    def read(self, block_id):
        self.read_history.append(block_id)
        prediction = self._predict_next_block()
        disk_id = block_id % len(self.disks)
        data = self.disks[disk_id].get(block_id, "<missing>")
        return f"Read 🧠 Block {block_id} ← Disk {disk_id}\nPredicted Next Block: {prediction}\nData: {data}"

    def _calculate_parity(self, data):
        return ''.join(chr(ord(c)^1) for c in data)  # Dummy parity

    def _predict_next_block(self):
        if len(self.read_history) < 2:
            return random.randint(0, 100)
        recent = list(self.read_history)[-3:]
        delta = recent[-1] - recent[-2]
        return recent[-1] + delta

# ========== 🎨 GUI – MagicBox ==========
class MagicBoxApp:
    def __init__(self, master):
        self.master = master
        self.master.title("📦 MagicBox RAID – Predictive Edition")
        self.master.configure(bg="#2a2a2a")
        self.raid = PredictiveRaid()

        # Font settings
        self.font_style = ("Helvetica", 14)
        self.title_font = ("Helvetica", 18, "bold")

        # Layout
        tk.Label(master, text="EchoStream MagicBox", font=self.title_font, fg="cyan", bg="#2a2a2a").pack(pady=10)

        self.block_entry = tk.Entry(master, font=self.font_style, width=10)
        self.block_entry.pack()
        self.block_entry.insert(0, "1")

        self.data_entry = tk.Entry(master, font=self.font_style, width=20)
        self.data_entry.pack()
        self.data_entry.insert(0, "alpha")

        tk.Button(master, text="📝 Write Block", font=self.font_style, bg="#444", fg="white",
                  command=self.write_block).pack(pady=5)

        tk.Button(master, text="🔎 Read Block", font=self.font_style, bg="#444", fg="white",
                  command=self.read_block).pack(pady=5)

        self.output_label = tk.Label(master, text="", font=self.font_style, fg="white", bg="#2a2a2a", justify=tk.LEFT)
        self.output_label.pack(pady=10)

    def write_block(self):
        try:
            block_id = int(self.block_entry.get())
            data = self.data_entry.get()
            result = self.raid.write(block_id, data)
            self.output_label.config(text=result)
        except ValueError:
            messagebox.showerror("Input Error", "Block ID must be an integer.")

    def read_block(self):
        try:
            block_id = int(self.block_entry.get())
            result = self.raid.read(block_id)
            self.output_label.config(text=result)
        except ValueError:
            messagebox.showerror("Input Error", "Block ID must be an integer.")

# ========== 🚀 Start MagicBox ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxApp(root)
    root.mainloop()

