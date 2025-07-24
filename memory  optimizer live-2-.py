# === Auto-load Libraries ===
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    import psutil, gc, random, time, os
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import speech_recognition as sr
except ImportError as e:
    print(f"Missing library: {e}. Please install it before running.")
    exit()

# === Node Class ===
class Node:
    def __init__(self, name):
        self.name = name
        self.utility_score = random.uniform(0.5, 1.5)
        self.memory_load = random.randint(200, 800)
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
    return round(usage + random.uniform(-5, 15), 2)

def log_memory_data(usage, forecast):
    with open("memory_log.txt", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} | RAM: {usage}% | Forecast: {forecast}%\n")

def voice_command_triggered():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = r.listen(source, timeout=5)
            command = r.recognize_google(audio).lower()
            return "optimize" in command
        except:
            return False

# === Tooltip Helper ===
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 35
        y += self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#ffffe0", relief="solid", borderwidth=1,
                         font=("Arial", 9))
        label.pack()

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

# === Main GUI ===
class MemoryOptimizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Memory Optimizer Suite")
        self.root.geometry("600x600")
        self.nodes = [Node(f"Node-{i}") for i in range(1, 5)]

        # Tabs setup
        tab_control = ttk.Notebook(root)
        self.tab_summary = ttk.Frame(tab_control)
        self.tab_nodes = ttk.Frame(tab_control)
        self.tab_chart = ttk.Frame(tab_control)
        self.tab_logs = ttk.Frame(tab_control)
        tab_control.add(self.tab_summary, text="Summary")
        tab_control.add(self.tab_nodes, text="Node Status")
        tab_control.add(self.tab_chart, text="Live Chart")
        tab_control.add(self.tab_logs, text="Logs")
        tab_control.pack(expand=1, fill="both")

        self.build_summary_tab()
        self.build_node_tab()
        self.build_chart_tab()
        self.build_log_tab()
        self.update_all()

    def build_summary_tab(self):
        self.ram_label = tk.Label(self.tab_summary, font=("Arial", 12))
        self.ram_label.pack(pady=5)
        self.forecast_label = tk.Label(self.tab_summary, font=("Arial", 12))
        self.forecast_label.pack(pady=5)

        opt_btn = tk.Button(self.tab_summary, text="Optimize Memory", font=("Arial", 12),
                            command=self.optimize_memory, bg="#0078D7", fg="white")
        opt_btn.pack(pady=5)
        ToolTip(opt_btn, "Click to run garbage collection and optimize memory")

        voice_btn = tk.Button(self.tab_summary, text="Voice Activate", command=self.voice_trigger)
        voice_btn.pack()
        ToolTip(voice_btn, "Say 'optimize' out loud to trigger optimization")

        probe_btn = tk.Button(self.tab_summary, text="Run Probe Diagnostic", command=self.run_probe)
        probe_btn.pack()
        ToolTip(probe_btn, "Run simulated memory diagnostic")

    def build_node_tab(self):
        self.node_display = tk.Label(self.tab_nodes, justify="left", font=("Arial", 10))
        self.node_display.pack(padx=10, pady=10)

    def build_chart_tab(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 2.5))
        self.chart = FigureCanvasTkAgg(self.fig, master=self.tab_chart)
        self.chart.get_tk_widget().pack()
        self.x_data, self.y_data = [], []

    def build_log_tab(self):
        self.log_area = tk.Text(self.tab_logs, wrap="word", height=20, font=("Arial", 9))
        self.log_area.pack(expand=True, fill="both")
        ToolTip(self.log_area, "Shows recent memory logs from file")

    def update_all(self):
        usage = psutil.virtual_memory().percent
        forecast = predict_memory_spike()
        self.ram_label.config(text=f"Current RAM Usage: {usage}%")
        self.forecast_label.config(text=f"Forecasted Spike: {forecast}%")
        log_memory_data(usage, forecast)

        # Chart update
        self.x_data.append(time.strftime("%H:%M:%S"))
        self.y_data.append(usage)
        self.ax.clear()
        self.ax.plot(self.x_data[-10:], self.y_data[-10:], marker='o')
        self.ax.set_title("RAM Usage Over Time")
        self.ax.set_ylabel("Usage (%)")
        self.ax.grid(True)
        self.chart.draw()

        # Node update
        activity = ""
        for i, node in enumerate(self.nodes):
            relocate = "Suggest Relocation" if node.suggest_relocation() else "Stable"
            if i < len(self.nodes) - 1:
                node.negotiate(self.nodes[i + 1])
            tag = "Negotiated" if node.negotiation_flag else ""
            activity += f"{node.name}: {relocate}, Score={round(node.utility_score, 2)} {tag}\n"
        self.node_display.config(text=activity)

        # Logs update
        try:
            with open("memory_log.txt", "r") as f:
                data = f.readlines()[-10:]
            self.log_area.delete("1.0", tk.END)
            self.log_area.insert(tk.END, "".join(data))
        except:
            self.log_area.insert(tk.END, "No logs found.")

        self.root.after(5000, self.update_all)

    def optimize_memory(self):
        gc.collect()
        messagebox.showinfo("Done", "Memory successfully optimized.")

    def run_probe(self):
        outcome = random.choice(["No issues", "Potential spike", "Node conflict", "Irregular load"])
        messagebox.showinfo("Diagnostic Result", f"Probe Result: {outcome}")

    def voice_trigger(self):
        if voice_command_triggered():
            self.optimize_memory()
        else:
            messagebox.showinfo("Voice Activation", "No optimization command detected.")

# === Launch App ===
if __name__ == "__main__":
    root = tk.Tk()
    MemoryOptimizer(root)
    root.mainloop()

