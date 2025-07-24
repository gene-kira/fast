# magicbox_app.py

import tkinter as tk
from tkinter import ttk
from probe_types import PulseProbe, DeepProbe, ReflectiveProbe
from neural_graph_expanded import NeuralGraph
from chart_visuals import GlyphChartFrame
from node_mesh import MeshRadarFrame

def run_app():
    app = tk.Tk()
    app.title("MagicBox GlyphGuard — Quantum Edition")
    app.geometry("1200x800")
    app.configure(bg="#2c2c2c")
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background="#2c2c2c")
    style.configure("TLabel", foreground="#f0f0f0", background="#2c2c2c", font=("Arial", 12))
    style.configure("TButton", foreground="#ffffff", background="#3a3f5c", font=("Arial", 11))

    notebook = ttk.Notebook(app)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    # 🟢 Control Tab
    control_tab = ttk.Frame(notebook)
    notebook.add(control_tab, text="Probe Control")

    probe_log = tk.Text(control_tab, height=15, bg="#111", fg="#9ae6b4", font=("Consolas", 10))
    probe_log.pack(fill="both", padx=10, pady=10)

    chart_tab = ttk.Frame(notebook)
    notebook.add(chart_tab, text="Glyph Confidence Chart")

    chart_panel = GlyphChartFrame(chart_tab)
    chart_panel.pack(fill="both", expand=True)

    mesh_tab = ttk.Frame(notebook)
    notebook.add(mesh_tab, text="Distributed Mesh Radar")

    radar = MeshRadarFrame(mesh_tab)
    radar.pack(fill="both", expand=True)

    graph = NeuralGraph()

    # 🧪 One-Click Probe Launchers
    def launch_probe(probe_class):
        probe = probe_class(f"GP-{probe_class.__name__}")
        probe.retry()
        probe.decay_signal()
        feedback = graph.link_probe(probe)
        probe_log.insert(tk.END, f"\n→ {probe_class.__name__} launched")
        probe_log.insert(tk.END, f"\nSignal: {probe.signal:.2f} {probe.display_battery()}")
        probe_log.insert(tk.END, f"\nFeedback: {feedback}\n")
        chart_panel.signal_data.append(feedback["confidence"])
        chart_panel.setup_chart()
        radar.pulse_nodes()

    ttk.Button(control_tab, text="🔴 Launch PulseProbe", command=lambda: launch_probe(PulseProbe)).pack(pady=5)
    ttk.Button(control_tab, text="🔵 Launch DeepProbe", command=lambda: launch_probe(DeepProbe)).pack(pady=5)
    ttk.Button(control_tab, text="🟣 Launch ReflectiveProbe", command=lambda: launch_probe(ReflectiveProbe)).pack(pady=5)

    app.mainloop()

