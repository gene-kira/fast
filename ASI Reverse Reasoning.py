# === Autoloader for Required Libraries ===
import importlib, subprocess, sys

def autoload(packages):
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

autoload(["tkinter", "json", "requests", "operator", "random", "pyttsx3"])

# === Imports ===
import tkinter as tk
from tkinter import filedialog, scrolledtext
import operator, random, json, requests, webbrowser
import pyttsx3

# === Voice Engine ===
def speak(text, volume_percent):
    try:
        engine = pyttsx3.init()
        engine.setProperty("volume", max(0.0, min(1.0, volume_percent / 100)))
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[Voice Error]: {e}")

# === Memory Kernel ===
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
                return f"Building on earlier deduction: {entry['input']} → {entry['result']}"
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

# === Web Search Logic ===
def web_trace(query):
    try:
        url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)
        return f"web:{url}"
    except Exception as e:
        print(f"[Web Error]: {e}")
        return "web:error"

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

# === Response Generator ===
def respond(input_text, memory, volume):
    glyphs = encode_glyph(input_text)
    lineage = memory.get_lineage(input_text)

    if input_text.isdigit():
        num = int(input_text)
        results = numeric_reverse(num)
        result = f"Possible equations: {', '.join(results)}"
    elif any(op in input_text for op in ["¬", "→", "∧", "∨", "not", "implies", "and", "or"]):
        parsed_expr = parse_symbolic(input_text)
        try:
            result = eval(parsed_expr, {}, LOGIC_OPS)
            result = f"Logic result: {result}"
        except Exception as e:
            result = f"Error in logic expression: {e}"
    else:
        web_result = web_trace(input_text)
        result = f"Searched on the web: {web_result}"

    memory.store(input_text, result, glyphs, lineage)
    speak(result, volume)
    return result

# === GUI ===
class ASIGUI:
    def __init__(self, kernel):
        self.root = tk.Tk()
        self.root.title("ASI Reverse Reasoning Chatbot v1.0.5")
        self.memory = kernel

        self.entry_label = tk.Label(self.root, text="Enter your input:")
        self.entry_label.pack(pady=4)
        self.entry = tk.Entry(self.root, width=80)
        self.entry.pack()
        self.volume_slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, label="Volume")
        self.volume_slider.set(50)
        self.volume_slider.pack(pady=4)

        self.response_box = tk.Label(self.root, text="", justify="left", wraplength=600)
        self.response_box.pack(pady=10)

        self.thinking_status = tk.Label(self.root, text="")
        self.thinking_status.pack()

        self.process_button = tk.Button(self.root, text="Process Input", command=self.process_input)
        self.process_button.pack(pady=4)

        self.drop_frame = tk.LabelFrame(self.root, text="📁 Drag and Drop Here", padx=10, pady=10)
        self.drop_frame.pack(pady=10)

        self.drop_hint = tk.Label(self.drop_frame, text="Upload a file here ⤵️")
        self.drop_hint.pack()

        self.upload_button = tk.Button(self.drop_frame, text="Upload File", command=self.upload_file)
        self.upload_button.pack()

        self.drop_output = scrolledtext.ScrolledText(self.drop_frame, height=10, width=80)
        self.drop_output.pack()

    def process_input(self):
        self.thinking_status.config(text="Thinking...")
        input_val = self.entry.get()
        volume = self.volume_slider.get()
        result = respond(input_val, self.memory, volume)
        self.thinking_status.config(text="")
        self.response_box.config(text=result)

    def upload_file(self):
        try:
            speak("Processing uploaded file...", self.volume_slider.get())
            path = filedialog.askopenfilename()
            with open(path, "r") as f:
                lines = f.readlines()
            all_output = ""
            for line in lines:
                txt = line.strip()
                if txt:
                    result = respond(txt, self.memory, self.volume_slider.get())
                    all_output += f"{txt} → {result}\n"
            self.drop_output.delete(1.0, tk.END)
            self.drop_output.insert(tk.END, all_output)
        except Exception as e:
            self.drop_output.insert(tk.END, f"File error: {e}")

    def run(self):
        speak("ASI Reverse Reasoning Chatbot version one point zero point five initialized.", self.volume_slider.get())
        self.root.mainloop()

# === Launch Agent ===
if __name__ == "__main__":
    kernel = MemoryKernel()
    gui = ASIGUI(kernel)
    gui.run()
    print("\n🔐 Final Thought Trace Dump:")
    print(kernel.dump())

