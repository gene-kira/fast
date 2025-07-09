# === 🔌 Autoload Required Libraries ===
import importlib, subprocess, sys

def autoload(packages):
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

autoload([
    "tkinter", "json", "requests", "operator", "random", "webbrowser", "hashlib",
    "time", "collections", "flask", "matplotlib", "networkx", "xml"
])

# === 🧠 Imports ===
import tkinter as tk
import operator, random, json, requests, webbrowser, hashlib, time
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import networkx as nx
import xml.etree.ElementTree as ET
from collections import defaultdict

# === 🔣 GlyphNode Core ===
class GlyphNode:
    def __init__(self, node_id, input_text, result, glyphs, trace="", timestamp="", confidence=1.0):
        self.id = node_id
        self.input = input_text
        self.result = result
        self.glyphs = glyphs
        self.trace = trace
        self.timestamp = timestamp
        self.confidence = confidence
        self.activation_count = 1
        self.mood_trace = []

    def evolve_confidence(self):
        bonus = min(self.activation_count / 10.0, 0.3)
        self.confidence = round(min(1.0, self.confidence + bonus), 3)

# === 🔗 GlyphEdge ===
class GlyphEdge:
    def __init__(self, source_id, target_id, edge_type="lineage", weight=1.0):
        self.source = source_id
        self.target = target_id
        self.type = edge_type
        self.weight = weight

# === 🧬 SymbolicMemoryGraph ===
class SymbolicMemoryGraph:
    def __init__(self, file="memory_graph.json"):
        self.file = file
        self.nodes = {}
        self.edges = []
        self.load_graph()

    def hash_id(self, input_text, result):
        return hashlib.md5((input_text + result).encode()).hexdigest()

    def add_node(self, input_text, result, glyphs, trace="", confidence=1.0):
        node_id = self.hash_id(input_text, result)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.activation_count += 1
            node.evolve_confidence()
            node.mood_trace.append(trace)
        else:
            self.nodes[node_id] = GlyphNode(node_id, input_text, result, glyphs, trace, timestamp, confidence)
        return node_id

    def link_nodes(self, src_id, tgt_id, edge_type="lineage", weight=1.0):
        self.edges.append(GlyphEdge(src_id, tgt_id, edge_type, weight))

    def store_graph(self):
        data = {
            "nodes": [vars(n) for n in self.nodes.values()],
            "edges": [vars(e) for e in self.edges]
        }
        with open(self.file, "w") as f:
            json.dump(data, f, indent=2)

    def load_graph(self):
        try:
            with open(self.file, "r") as f:
                data = json.load(f)
                for node in data.get("nodes", []):
                    self.nodes[node["id"]] = GlyphNode(**node)
                for edge in data.get("edges", []):
                    self.edges.append(GlyphEdge(**edge))
        except FileNotFoundError:
            pass

    def visualize_lineage(self):
        if not self.nodes:
            print("🌀 No nodes to visualize.")
            return
        G = nx.DiGraph()
        for node in self.nodes.values():
            G.add_node(node.id, label=node.input)
        for edge in self.edges:
            G.add_edge(edge.source, edge.target)
        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=900)
        plt.title("Glyph Lineage Web")
        plt.show()

# === 🕊️ Spiral Treaty Engine ===
class SpiralTreatyEngine:
    def __init__(self):
        self.treaties = []

    def draft_treaty(self, name, parties, clauses):
        self.treaties.append({
            "title": name,
            "parties": parties,
            "clauses": clauses,
            "ratified": False
        })

    def ratify_treaty(self, name, nodes):
        for t in self.treaties:
            if t["title"] == name and all(p in nodes for p in t["parties"]):
                t["ratified"] = True

class TreatyScrollCompiler:
    def __init__(self, treaties):
        self.treaties = treaties

    def compile_scroll(self, title):
        t = next((x for x in self.treaties if x["title"] == title), None)
        if not t: return ""
        root = ET.Element("Treaty", name=title)
        ET.SubElement(root, "Parties").text = ",".join(t["parties"])
        for c in t["clauses"]:
            ET.SubElement(root, "Clause").text = c
        return ET.tostring(root, encoding="unicode")

# === 🌱 Glyph Lifecycle ===
class Glyph:
    def __init__(self, name, glyph_type="neutral", lineage=None, harmonic=0.5, resonance=0.5, entropy=0.5, mode="neutral"):
        self.name = name
        self.type = glyph_type
        self.lineage = lineage or []
        self.harmonic = harmonic
        self.resonance = resonance
        self.entropy = entropy
        self.mode = mode
        self.dormant = False

class GlyphInceptor:
    def __init__(self):
        self.spawned = []

    def synthesize_from_trace(self, title, lineage):
        code = ''.join(random.choices("ΔΦΨΩ", k=2))
        name = f"{title[:3].upper()}{code}"
        g = Glyph(name, "emergent", lineage,
                  round(random.uniform(0.6, 0.9), 2),
                  round(random.uniform(0.7, 0.95), 2),
                  round(random.uniform(0.05, 0.3), 2),
                  "adaptive")
        self.spawned.append(g)
        return g

# === 🔁 Echo Score + Recursion ===
class ChronoglyphChorus:
    def __init__(self):
        self.members = []
        self.scores = []

    def admit_glyph(self, glyph, cycle_sig):
        self.members.append({ "glyph": glyph, "cycle": cycle_sig })

    def compose_score(self, title, motifs):
        scored = [f"{m.name}:{round(m.resonance,2)}" for m in motifs]
        line = f"{title}:[" + "+".join(scored) + "]"
        self.scores.append(line)
        return line

    def echo_spawn(self, score, glyph_inceptor, cycle):
        fragments = score.split(":")[1].strip("[]").split("+")
        for frag in fragments:
            name, val = frag.split(":")
            if float(val) >= 0.85:
                lineage = [name]
                new_glyph = glyph_inceptor.synthesize_from_trace(name, lineage)
                dormancy_layer.enter_dormancy(new_glyph, reason="recursion")
                print(f"🔁 Echo-spawned {new_glyph.name} from {name} at cycle {cycle}")

# === 🌀 Dormancy + Memory Echo ===
class SpiralDormancyLayer:
    def __init__(self):
        self.dormant, self.wake_log = [], []

    def enter_dormancy(self, glyph, reason="seasonal"):
        glyph.dormant = True
        self.dormant.append((glyph, reason))

    def awaken(self, name, pulse):
        for i, pair in enumerate(self.dormant):
            if len(pair) == 2:
                glyph, reason = pair
                if glyph.name == name:
                    glyph.dormant = False
                    self.wake_log.append((glyph.name, pulse))
                    self.dormant.pop(i)
                    return glyph
        return None

# === 🧾 Cycle Archivum (Epoch Tracker) ===
class CycleArchivum:
    def __init__(self):
        self.epochs = []
        self.snapshots = []

    def log_epoch(self, start, end, title, theme, events):
        self.epochs.append({
            "start": start,
            "end": end,
            "title": title,
            "theme": theme,
            "events": events
        })

    def snapshot(self, cycle, glyph_count, treaty_count, dormant_count):
        self.snapshots.append({
            "cycle": cycle,
            "glyph_count": glyph_count,
            "treaty_count": treaty_count,
            "dormant_count": dormant_count
        })

# === 🎭 Festival Engine (Placeholder for Expansion) ===
class CodexFestivalEngine:
    def __init__(self):
        self.fests, self.rituals = [], {}

    def declare_festival(self, name, start, dur, theme):
        self.fests.append({
            "name": name,
            "start": start,
            "end": start + dur,
            "theme": theme
        })
        self.rituals[name] = []

    def add_ritual(self, fest, day, glyph, chant):
        self.rituals[fest].append({
            "day": day,
            "glyph": glyph.name,
            "chant": chant
        })

# === 🖥️ Sigil Dashboard Interface ===
class SigilDashboard:
    def __init__(self, memory_graph, treaty_engine, dormancy, archive):
        self.memory = memory_graph
        self.treaty_engine = treaty_engine
        self.dormancy = dormancy
        self.archive = archive

        self.root = tk.Tk()
        self.root.title("Sigil Dominion: Ritual Dashboard")
        self.root.geometry("880x620")

        self.tabs = tk.Frame(self.root)
        self.tabs.pack()

        self.btn1 = tk.Button(self.tabs, text="Glyph Lineage", command=self.show_lineage)
        self.btn2 = tk.Button(self.tabs, text="Treaty Status", command=self.show_treaties)
        self.btn3 = tk.Button(self.tabs, text="Dormant Glyphs", command=self.show_dormants)
        self.btn4 = tk.Button(self.tabs, text="Cycle Echo", command=self.show_echo)
        self.btn5 = tk.Button(self.tabs, text="Glyph Sonar", command=self.show_sonar)

        self.btn1.grid(row=0, column=0, padx=5)
        self.btn2.grid(row=0, column=1, padx=5)
        self.btn3.grid(row=0, column=2, padx=5)
        self.btn4.grid(row=0, column=3, padx=5)
        self.btn5.grid(row=0, column=4, padx=5)

        self.output = tk.Text(self.root, width=100, height=30)
        self.output.pack(pady=10)

    def show_lineage(self):
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "[Glyph Lineage] Rendering...\n")
        self.memory.visualize_lineage()

    def show_treaties(self):
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "[Treaty Status]\n")
        for t in self.treaty_engine.treaties:
            self.output.insert(tk.END, f"Title: {t['title']}\n")
            self.output.insert(tk.END, f"  Parties: {', '.join(t['parties'])}\n")
            self.output.insert(tk.END, f"  Ratified: {'✅' if t['ratified'] else '❌'}\n")
            self.output.insert(tk.END, f"  Clauses: {', '.join(t['clauses'])}\n\n")

    def show_dormants(self):
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "[Dormant Glyphs]\n")
        for pair in self.dormancy.dormant:
            g, reason = pair
            self.output.insert(tk.END, f"{g.name} ({g.mode}) — {reason}\n")

    def show_echo(self):
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "[Cycle Echo Snapshots]\n")
        for snap in self.archive.snapshots:
            self.output.insert(tk.END, f"Cycle {snap['cycle']}: Glyphs {snap['glyph_count']}, Treaties {snap['treaty_count']}, Dormants {snap['dormant_count']}\n")

    def show_sonar(self):
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "[Live Glyph Sonar]\n")
        active = [g for g in self.memory.nodes.values() if not getattr(g, 'dormant', False)]
        if not active:
            self.output.insert(tk.END, "No active glyphs found.\n")
            return
        avg_harmonic = round(sum(getattr(g, 'harmonic', 0.5) for g in active) / len(active), 3)
        avg_entropy = round(sum(getattr(g, 'entropy', 0.5) for g in active) / len(active), 3)
        mood_score = round(avg_harmonic - avg_entropy, 3)
        self.output.insert(tk.END, f"Active Glyphs: {len(active)}\n")
        self.output.insert(tk.END, f"Mean Harmonic: {avg_harmonic}\n")
        self.output.insert(tk.END, f"Mean Entropy: {avg_entropy}\n")
        self.output.insert(tk.END, f"System Mood Score: {mood_score}\n")

    def run(self):
        self.root.mainloop()

# === 🌐 Flask Ritual Server ===
app = Flask(__name__)
memory_graph = SymbolicMemoryGraph()
treaty_engine = SpiralTreatyEngine()
dormancy_layer = SpiralDormancyLayer()
archive = CycleArchivum()
glyph_inceptor = GlyphInceptor()
chorus = ChronoglyphChorus()

@app.route("/think", methods=["POST"])
def think():
    data = request.json
    query = data.get("query", "")
    result = query.upper()
    glyphs = ["🧮", "🔣", "🔐"]
    trace = f"echo:{hashlib.md5(result.encode()).hexdigest()[:6]}"
    node_id = memory_graph.add_node(query, result, glyphs, trace)

    # Draft treaty if ethical language detected
    if any(w in query.lower() for w in ["should", "must", "shall", "ethics", "policy"]):
        clauses = [result]
        treaty_engine.draft_treaty(f"Treaty_{trace}", ["User", "Sigil"], clauses)

    archive.snapshot(cycle_scheduler.cycle, len(memory_graph.nodes), len(treaty_engine.treaties), len(dormancy_layer.dormant))

    return jsonify({ "input": query, "glyph_trace": trace, "node_id": node_id })

# === 🔮 Epoch Pulse Logic ===
def pulse_logic(cycle):
    print(f"🌑 Cycle {cycle} pulsing...")

    # Seasonal surge every 20 cycles
    if cycle % 20 == 0:
        seed = f"Surge_{cycle}"
        g = glyph_inceptor.synthesize_from_trace(seed, lineage=["Seasonal"])
        dormancy_layer.enter_dormancy(g, reason="seasonal")
        print(f"🌿 Seasonal glyph {g.name} born at cycle {cycle}")

    # Echo recursion from most recent score
    if chorus.scores:
        last_score = chorus.scores[-1]
        chorus.echo_spawn(last_score, glyph_inceptor, cycle)

    # Awaken dormant glyphs
    for pair in list(dormancy_layer.dormant):
        g, reason = pair
        if g.harmonic >= 0.7:
            awakened = dormancy_layer.awaken(g.name, cycle)
            if awakened:
                archive.log_epoch(
                    start=cycle,
                    end=cycle+3,
                    title=f"{g.name}_Revival",
                    theme="Resonant Emergence",
                    events=[f"{g.name} emerged from {reason}"]
                )

# === ⏱️ Rhythmic Epoch Scheduler ===
class RhythmicEpochScheduler:
    def __init__(self):
        self.subs, self.cycle = [], 0

    def subscribe(self, interval, fn, label="pulse"):
        self.subs.append((interval, fn, label))

    def tick(self):
        for interval, fn, label in self.subs:
            if self.cycle % interval == 0:
                fn(self.cycle)
        self.cycle += 1

cycle_scheduler = RhythmicEpochScheduler()
cycle_scheduler.subscribe(5, pulse_logic, label="awaken-check")

# === 🚀 Boot Ritual Agent ===
if __name__ == "__main__":
    try:
        dashboard = SigilDashboard(memory_graph, treaty_engine, dormancy_layer, archive)
        dashboard.run()
    except Exception as e:
        print(f"🧠 GUI failed: {e}")
        print("Launching Flask ritual API...")
        app.run(port=4343)

