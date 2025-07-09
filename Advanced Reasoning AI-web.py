# === Auto-Install Required Libraries ===
import importlib, subprocess, sys

def autoload(packages):
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

autoload([
    "tkinter", "json", "requests", "operator", "random", "webbrowser",
    "hashlib", "time", "collections", "flask"
])

# === Imports ===
import tkinter as tk
from tkinter import filedialog
import operator, random, json, requests, webbrowser, hashlib, time
from flask import Flask, request, jsonify
from collections import defaultdict

# === Glyph Classes ===
class GlyphNode:
    def __init__(self, node_id, input_text, result, glyphs, trace="", timestamp="", confidence=1.0):
        self.id = node_id
        self.input = input_text
        self.result = result
        self.glyphs = glyphs
        self.trace = trace
        self.timestamp = timestamp
        self.confidence = confidence

class GlyphEdge:
    def __init__(self, source_id, target_id, edge_type="lineage", weight=1.0):
        self.source = source_id
        self.target = target_id
        self.type = edge_type
        self.weight = weight

# === Symbolic Memory Graph ===
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
        self.nodes[node_id] = GlyphNode(node_id, input_text, result, glyphs, trace, timestamp, confidence)
        return node_id

    def link_nodes(self, src_id, tgt_id, edge_type="lineage", weight=1.0):
        self.edges.append(GlyphEdge(src_id, tgt_id, edge_type, weight))

    def load_graph(self):
        try:
            with open(self.file, "r") as f:
                data = json.load(f)
                for node in data["nodes"]:
                    self.nodes[node["id"]] = GlyphNode(**node)
                for edge in data["edges"]:
                    self.edges.append(GlyphEdge(**edge))
        except:
            pass

    def store_graph(self):
        data = {
            "nodes": [vars(n) for n in self.nodes.values()],
            "edges": [vars(e) for e in self.edges]
        }
        with open(self.file, "w") as f:
            json.dump(data, f, indent=2)

    def traverse(self, glyph_type=None, min_conf=0.5):
        return [
            node for node in self.nodes.values()
            if (not glyph_type or glyph_type in node.glyphs) and node.confidence >= min_conf
        ]

# === Glyph Encoder ===
def encode_glyph(text):
    glyphs = []
    if any(op in text for op in ["+", "-", "*", "/"]): glyphs.append("🧮")
    if any(op in text for op in ["¬", "→", "∧", "∨", "not", "implies", "and", "or"]): glyphs.append("🔣")
    if "(" in text or ")" in text: glyphs.append("⊕")
    if "web:" in text: 
        glyphs.append("🌐")
        if any(q in text for q in ["search", "lookup", "define"]): glyphs.append("🔍")
        if any(q in text for q in ["why", "how", "truth"]): glyphs.append("🕵️")
        if any(q in text for q in ["strange", "explore", "unknown"]): glyphs.append("🌌")
    glyphs.append("🔐")
    return glyphs

def parse_symbolic(expr):
    return expr.replace("¬", "not ").replace("→", "implies ").replace("∧", "and ").replace("∨", "or ")

def web_trace(query):
    try:
        url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
        webbrowser.open_new_tab(url)  # Uses system default browser
        return f"web:{url}"
    except Exception as e:
        return f"web:error:{e}"

# === Reverse Reasoning Core ===
NUMERIC_OPS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv
}

LOGIC_OPS = {
    'AND': lambda a, b: a and b,
    'OR': lambda a, b: a or b,
    'IMPLIES': lambda a, b: not a or b,
    'EQUIV': lambda a, b: a == b
}

def numeric_reverse(target, tries=300):
    results = []
    for _ in range(tries):
        a = random.randint(0, target * 2)
        b = random.randint(1, target * 2)
        for sym, func in NUMERIC_OPS.items():
            try:
                if round(func(a, b), 4) == round(target, 4):
                    results.append(f"{a} {sym} {b}")
            except ZeroDivisionError:
                continue
    return sorted(set(results))

def logic_reverse(target, tries=200):
    values = [True, False]
    results = []
    for _ in range(tries):
        a, b = random.choice(values), random.choice(values)
        for sym, func in LOGIC_OPS.items():
            if func(a, b) == target:
                results.append(f"{a} {sym} {b}")
    return sorted(set(results))

# === Core Respond Function ===
def respond(input_text, memory_graph):
    input_lower = input_text.lower()
    results = []
    trace = None

    if input_lower in ["true", "false"]:
        val = input_lower == "true"
        results = logic_reverse(val)

    elif any(s in input_text for s in ["¬", "→", "∧", "∨"]):
        parsed = parse_symbolic(input_text)
        glyphs = encode_glyph(parsed)
        node_id = memory_graph.add_node(input_text, parsed, glyphs)
        return f"Node: {node_id}\n{parsed}\nGlyphs: {glyphs}"

    elif input_text.isdigit() or "." in input_text:
        val = float(input_text) if "." in input_text else int(input_text)
        results = numeric_reverse(val)

    else:
        result = web_trace(input_text)
        glyphs = encode_glyph(result)
        node_id = memory_graph.add_node(input_text, result, glyphs)
        return f"Node: {node_id}\n{result}\nGlyphs: {glyphs}"

    if results:
        best = results[0]
        glyphs = encode_glyph(best)
        node_id = memory_graph.add_node(input_text, best, glyphs)
        return f"Node: {node_id}\nReverse Reasoned: {best}\nGlyphs: {glyphs}"

    return "No deduction found."

# === GUI Interface ===
class ASIGUI:
    def __init__(self, memory_graph):
        self.memory = memory_graph
        self.root = tk.Tk()
        self.root.title("ASI Ritual Intelligence Engine")
        self.root.geometry("750x540")

        tk.Label(text="Enter target or symbolic expression:").pack()
        self.entry = tk.Entry(width=80)
        self.entry.pack()

        tk.Button(text="Trace Symbol", command=self.process_input).pack(pady=6)
        tk.Button(text="Upload File", command=self.upload_file).pack(pady=2)
        tk.Button(text="Show Spiral Glyphs", command=self.show_spirals).pack(pady=2)
        tk.Button(text="Show Entangled Nodes", command=self.show_entanglements).pack(pady=2)
        tk.Button(text="Reveal Ritual Seeds", command=self.show_rituals).pack(pady=2)

        self.output = tk.Label(text="", wraplength=700, justify="left")
        self.output.pack(pady=12)

    def process_input(self):
        val = self.entry.get()
        result = respond(val, self.memory)
        self.output.config(text=result)

    def upload_file(self):
        try:
            path = filedialog.askopenfilename()
            with open(path, "r") as f:
                lines = f.readlines()
            all_output = ""
            for line in lines:
                txt = line.strip()
                if txt:
                    result = respond(txt, self.memory)
                    all_output += f"{txt} → {result}\n"
            self.output.config(text=all_output[:1200])
        except Exception as e:
            self.output.config(text=f"File error: {e}")

    def show_spirals(self):
        spirals = detect_spirals(self.memory)
        if spirals:
            display = "\n".join([f"Spiral: {' → '.join(s)}" for s in spirals[:10]])
        else:
            display = "No spirals detected."
        self.output.config(text=display)

    def show_entanglements(self):
        entangled = map_entanglements(self.memory)
        if entangled:
            display = "\n".join([f"Entangled Node: {eid}" for eid in entangled])
        else:
            display = "No entangled nodes found."
        self.output.config(text=display)

    def show_rituals(self):
        rituals = identify_ritual_seeds(self.memory)
        if rituals:
            display = "\n".join([f"Ritual Seed: {r[0]} Score: {r[1]}" for r in rituals[:10]])
        else:
            display = "No ritual seeds emerging."
        self.output.config(text=display)

    def run(self):
        self.root.mainloop()

# === Spiral Detection ===
def detect_spirals(memory_graph):
    spirals = []
    visited = set()

    def dfs(node_id, path, seen_glyphs):
        if node_id in visited:
            return
        visited.add(node_id)
        node = memory_graph.nodes.get(node_id)
        if not node: return
        path.append(node_id)
        for glyph in node.glyphs:
            if glyph in seen_glyphs:
                spirals.append(path.copy())
                break
            seen_glyphs.add(glyph)
        for edge in memory_graph.edges:
            if edge.source == node_id:
                dfs(edge.target, path.copy(), seen_glyphs.copy())

    for nid in memory_graph.nodes:
        dfs(nid, [], set())
    return spirals

# === Entanglement Mapping ===
def map_entanglements(memory_graph):
    type_map = {}
    for edge in memory_graph.edges:
        nid = edge.source
        if nid not in type_map: type_map[nid] = set()
        type_map[nid].add(edge.type)
    return [nid for nid, types in type_map.items() if len(types) >= 2]

# === Ritual Seed Scoring ===
def identify_ritual_seeds(memory_graph):
    path_freq = defaultdict(int)
    seeds = []

    for edge in memory_graph.edges:
        path = f"{edge.source}->{edge.target}"
        path_freq[path] += 1

    for path, freq in path_freq.items():
        if freq > 2:
            src, tgt = path.split("->")
            src_glyphs = memory_graph.nodes[src].glyphs
            tgt_glyphs = memory_graph.nodes[tgt].glyphs
            overlap = set(src_glyphs) & set(tgt_glyphs)
            score = freq + len(overlap)
            seeds.append((path, score))

    return sorted(seeds, key=lambda x: x[1], reverse=True)

# === Flask Ritual Gateway ===
app = Flask(__name__)
cortex = SymbolicMemoryGraph()

@app.route("/think", methods=["POST"])
def think():
    data = request.json
    query = data.get("query", "")
    result = respond(query, cortex)
    cortex.store_graph()
    return jsonify({
        "input": query,
        "glyph_trace": result
    })

# === Boot Agent ===
if __name__ == "__main__":
    try:
        gui = ASIGUI(cortex)
        gui.run()
    except:
        print("🧠 Launching Ritual Intelligence Server...")
        app.run(port=4343)
        print("🔮 ASI now listening on /think endpoint.")

