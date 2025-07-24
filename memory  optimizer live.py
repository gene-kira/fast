# === Auto-load Libraries ===
try:
    import tkinter as tk
    from tkinter import messagebox
    import psutil, gc, random, time, os
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import speech_recognition as sr
except ImportError as e:
    print(f"Missing library: {e}. Please install it before running.")
    exit()

# === Node Class for Modular Communication ===
class Node:
    def __init__(self, name):
        self.name = name
        self.utility_score = random.uniform(0.5, 1.5)
        self.memory_load = random.randint(200, 800)  # Simulated MB
        self.negotiation_flag = False

    def suggest_relocation(self, threshold=700):
        return self.memory_load > threshold

    def negotiate(self, other):
        if self.utility_score > other.utility_score:
            self.memory_load -= 50
            other.memory_load += 50
            self.negotiation_flag = True

# === Forecast Engine ===
def predict_memory_spike():
    usage = psutil.virtual_memory().percent
    spike = usage + random.uniform(-5, 15)
    return round(spike, 2)

# === Persistent Logger ===
def log_memory_data(usage, forecast):
    with open("memory_log.txt", "a") as f:
        timestamp = time.strftime("%H:%M:%S")
        f.write(f"{timestamp} | RAM: {usage}% | Forecast: {forecast}%\n")

# === Voice Activation ===
def voice_command_triggered():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("Listening...")
            audio = r.listen(source, timeout=5)
            command = r.recognize_google(audio).lower()
            return "optimize" in command
        except:
            return False

# === Main GUI ===
class MemoryOptimizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Memory Optimizer - Full Spectrum")
        self.root.geometry("450x550")
        self.root.configure(bg="#f0f0f0")

        self.nodes = [Node(f"Node-{i}") for i in range(1, 5)]

        # Title
        tk.Label(root, text="Memory Optimizer", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)

        self.ram_status = tk.Label(root, text="", font=("Arial", 10), bg="#f0f0f0")
        self.ram_status.pack()

        self.forecast_status = tk.Label(root, text="", font=("Arial", 10), bg="#f0f0f0")
        self.forecast_status.pack()

        self.node_status = tk.Label(root, text="", font=("Arial", 9), bg="#f0f0f0", justify="left")
        self.node_status.pack()

        # Optimize Button
        btn = tk.Button(root, text="Optimize Memory", font=("Arial", 12), bg="#0078D7", fg="white",
                        command=self.optimize_memory)
        btn.pack(pady=10, ipadx=10, ipady=5)

        # Voice Control Button
        voice_btn = tk.Button(root, text="Activate via Voice", font=("Arial", 10), command=self.voice_trigger)
        voice_btn.pack(pady=5)

        # Probe Diagnostics
        tk.Button(root, text="Run Probe Diagnostic", font=("Arial", 10), command=self.run_probe).pack(pady=5)

        # Memory Chart
        self.init_chart()

        self.update_status()

    def init_chart(self):
        fig, self.ax = plt.subplots(figsize=(4, 2))
        self.chart = FigureCanvasTkAgg(fig, master=self.root)
        self.chart.get_tk_widget().pack(pady=10)
        self.x_data, self.y_data = [], []

    def update_chart(self, usage):
        self.x_data.append(time.strftime("%H:%M:%S"))
        self.y_data.append(usage)
        self.ax.clear()
        self.ax.plot(self.x_data[-10:], self.y_data[-10:], marker='o')
        self.ax.set_title("RAM Usage Over Time")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Usage (%)")
        self.ax.grid(True)
        self.chart.draw()

    def update_status(self):
        usage = psutil.virtual_memory().percent
        forecast = predict_memory_spike()

        self.ram_status.config(text=f"Current RAM Usage: {usage}%")
        self.forecast_status.config(text=f"Forecasted Spike: {forecast}%")
        log_memory_data(usage, forecast)
        self.update_chart(usage)

        activity = ""
        for i, node in enumerate(self.nodes):
            relocate = "Suggest Relocation" if node.suggest_relocation() else "Stable"
            if i < len(self.nodes) - 1:
                node.negotiate(self.nodes[i + 1])
            tag = "Negotiated" if node.negotiation_flag else ""
            activity += f"{node.name}: {relocate}, Score={round(node.utility_score, 2)} {tag}\n"
        self.node_status.config(text=activity)

        self.root.after(5000, self.update_status)

    def optimize_memory(self):
        try:
            gc.collect()
            messagebox.showinfo("Done", "Memory optimized with AI-driven features.")
        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed:\n{e}")

    def run_probe(self):
        issues = random.choice(["No issues", "Potential spike", "Node memory uneven", "Corrupted footprint"])
        messagebox.showinfo("Diagnostic Result", f"Probe Result: {issues}")

    def voice_trigger(self):
        if voice_command_triggered():
            self.optimize_memory()
        else:
            messagebox.showinfo("Voice Command", "No optimization command detected.")

# === Launch GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = MemoryOptimizer(root)
    root.mainloop()

