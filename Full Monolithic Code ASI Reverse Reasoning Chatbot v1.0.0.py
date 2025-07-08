# === Auto-Install Required Libraries ===
import importlib, subprocess, sys

def autoload(packages):
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

autoload(["tkinter", "json", "requests", "operator", "random"])

# === Imports ===
import tkinter as tk
from tkinter import filedialog
import operator, random, json, requests, webbrowser

# === Thought Trace Memory Kernel ===
class MemoryKernel:
    def __init__(self, file="glyph_memory.json"):
        self.file = file
        try:
            with open(file, "r") as f:
                self.data = json.load(f)
        except:
            self.data = []

    def store(self, input_text, result, glyphs, trace=None):
        entry = {
            "input": input_text,
            "result": result,
            "glyphs": glyphs,
            "trace": trace or "Initial deduction"
        }
        self.data.append(entry)
        with open(self.file, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_lineage(self, current_input):
        for entry in reversed(self.data):
            if entry["input"] != current_input and entry["glyphs"]:
                return f"Based on earlier deduction '{entry['input']} → {entry['result']}'"
        return None

    def dump(self):
        return "\n".join([f"{e['input']} → {e['result']} {e['glyphs']}, Trace: {e['trace']}" for e in self.data])

# === Glyph Encoder ===
def encode_glyph(text):
    glyphs = []
    if any(op in text for op in ["+", "-", "*", "/"]):
        glyphs.append("🧮")
    if any(op in text for op in ["¬", "→", "∧", "∨", "not", "implies", "and", "or"]):
        glyphs.append("🔣")
    if "(" in text or ")" in text:
        glyphs.append("⊕")
    if "web:" in text:
        glyphs.append("🌐")
    glyphs.append("🔐")
    return glyphs

# === Symbolic Parser ===
def parse_symbolic(expr):
    return expr.replace("¬", "not ").replace("→", "implies ").replace("∧", "and ").replace("∨", "or ")

# === Bing Web Search Integration ===
def web_trace(query):
    try:
        url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)
        return f"web:{url}"
    except:
        return "web:error"

# === Reasoning Core ===
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

# === Reasoning Engine ===
def respond(input_text, memory):
    input_lower = input_text.lower()

    # Boolean
    if input_lower in ["true", "false"]:
        val = input_lower == "true"
        results = logic_reverse(val)

    # Symbolic
    elif any(s in input_text for s in ["¬", "→", "∧", "∨"]):
        parsed = parse_symbolic(input_text)
        glyphs = encode_glyph(parsed)
        trace = memory.get_lineage(input_text)
        memory.store(input_text, parsed, glyphs, trace)
        return f"{trace or ''}\n{parsed}\nGlyphs: {glyphs}"

    # Numeric
    elif input_text.isdigit() or "." in input_text:
        val = float(input_text) if "." in input_text else int(input_text)
        results = numeric_reverse(val)

    # Web or unknown
    else:
        result = web_trace(input_text)
        glyphs = encode_glyph(result)
        trace = memory.get_lineage(input_text)
        memory.store(input_text, result, glyphs, trace)
        return f"{trace or ''}\n{result}\nGlyphs: {glyphs}"

    if results:
        best = results[0]
        glyphs = encode_glyph(best)
        trace = memory.get_lineage(input_text)
        memory.store(input_text, best, glyphs, trace)
        return f"{trace or ''}\nReverse Reasoned: {best}\nGlyphs: {glyphs}"

    return "No deduction found."

# === GUI Interface ===
class ASIGUI:
    def __init__(self, memory):
        self.memory = memory
        self.root = tk.Tk()
        self.root.title("ASI Quantum Reasoning Engine")
        self.root.geometry("600x400")

        tk.Label(text="Enter symbol, logic, or numeric target:").pack()
        self.entry = tk.Entry(width=70)
        self.entry.pack()

        tk.Button(text="Think & Trace", command=self.process_input).pack(pady=6)
        tk.Button(text="Upload File", command=self.upload_file).pack()

        self.output = tk.Label(text="", wraplength=550, justify="left")
        self.output.pack()

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
            self.output.config(text=all_output[:1000])
        except Exception as e:
            self.output.config(text=f"File error: {e}")

    def run(self):
        self.root.mainloop()

# === Boot Agent ===
if __name__ == "__main__":
    mem = MemoryKernel()
    gui = ASIGUI(mem)
    gui.run()
    print("\n🔐 Final Thought Trace Dump:")

