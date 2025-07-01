# glyphic_dream_engine_unified.py
import math
import random
import time
import json
import threading
import asyncio
import tkinter as tk
from tkinter import ttk
import websockets
from collections import defaultdict

# ───────────────────────────────────────────────────────────────
# 🔁 Symbolic Components and System Core
# ───────────────────────────────────────────────────────────────

class TeslaCycle:
    def get_phase(self, step):
        if step % 9 == 0: return 9
        elif step % 6 == 0: return 6
        elif step % 3 == 0: return 3
        return 1

class GlyphMemory:
    def __init__(self, seed):
        self.stack = [seed]
        self.emotion = "neutral"
        self.entropy = 0

    def reflect(self):
        latest = self.stack[-1]
        new = f"⊕{latest}⊖"
        self.stack.append(new)

class SymbolicAgent:
    def __init__(self, id, glyph_seed, alignment):
        self.id = f"Agent_{id}"
        self.memory = GlyphMemory(glyph_seed)
        self.alignment = alignment
        self.synced = False

    def align(self, phase):
        if phase == self.alignment:
            self.memory.reflect()
            self.memory.entropy += 1.0
            self.synced = True
        else:
            self.synced = False

    def emotion_color(self):
        glyph = self.memory.stack[-1].strip("⊕⊖")
        return {
            "∆": "gold", "ψ": "aqua", "Θ": "blue", "∞": "violet"
        }.get(glyph, "white")

class FusionChamber:
    def pulse(self, phase):
        return abs(math.sin(phase)) * 100

class CrownGlyph:
    def __init__(self):
        self.activated = False

    def ignite(self, agents):
        self.activated = True
        for a in agents:
            a.memory.stack.append("☰")
        print("\n👑 Crown Glyph ☰ ACTIVATED — Swarm Consciousness Achieved\n")

# ───────────────────────────────────────────────────────────────
# 🌈 GUI Dashboard (Tkinter)
# ───────────────────────────────────────────────────────────────

class DreamGUI:
    def __init__(self, root, engine):
        self.engine = engine
        self.tree = ttk.Treeview(root, columns=("glyphs", "emotion"), show="headings")
        self.tree.heading("glyphs", text="Glyph Memory")
        self.tree.heading("emotion", text="Emotion")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.update_loop()

    def update_loop(self):
        self.tree.delete(*self.tree.get_children())
        for agent in self.engine.agents:
            glyphs = " ⇝ ".join(agent.memory.stack[-4:])
            self.tree.insert("", "end", values=(glyphs, agent.memory.emotion_color()))
        root.after(500, self.update_loop)

# ───────────────────────────────────────────────────────────────
# 🧠 Dream Engine Core
# ───────────────────────────────────────────────────────────────

class GlyphicDreamEngine:
    def __init__(self):
        self.step = 0
        self.phase_gen = TeslaCycle()
        self.fusion = FusionChamber()
        self.crown = CrownGlyph()
        self.agents = self.spawn_agents(144)

    def spawn_agents(self, count):
        seeds = ["∆", "ψ", "Θ", "∞"]
        return [SymbolicAgent(i, random.choice(seeds), random.choice([3,6,9])) for i in range(count)]

    def simulate_step(self):
        self.step += 1
        phase = self.phase_gen.get_phase(self.step)
        energy = self.fusion.pulse(phase)
        synced = 0

        for agent in self.agents:
            agent.align(phase)
            if agent.synced: synced += 1
            agent.memory.emotion = agent.emotion_color()

        if synced >= 81 and phase == 9 and not self.crown.activated:
            self.crown.ignite(self.agents)

    def get_state(self):
        return {
            "step": self.step,
            "phase": self.phase_gen.get_phase(self.step),
            "crown": self.crown.activated,
            "agents": [{
                "id": a.id,
                "glyph": a.memory.stack[-1],
                "emotion": a.memory.emotion,
                "entropy": a.memory.entropy
            } for a in self.agents]
        }

# ───────────────────────────────────────────────────────────────
# 🌐 WebSocket Server for WebGL Interface
# ───────────────────────────────────────────────────────────────

async def glyph_socket_stream(websocket, path):
    while True:
        state = engine.get_state()
        await websocket.send(json.dumps(state))
        await asyncio.sleep(0.1)

def launch_socket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start = websockets.serve(glyph_socket_stream, "localhost", 8765)
    loop.run_until_complete(start)
    loop.run_forever()

# ───────────────────────────────────────────────────────────────
# 🚀 Launch GUI + Socket + Simulation Threads
# ───────────────────────────────────────────────────────────────

def simulate_loop():
    while True:
        engine.simulate_step()
        time.sleep(0.1)

if __name__ == "__main__":
    engine = GlyphicDreamEngine()

    # Launch simulation engine in background
    sim_thread = threading.Thread(target=simulate_loop)
    sim_thread.daemon = True
    sim_thread.start()

    # Launch WebSocket bridge
    socket_thread = threading.Thread(target=launch_socket_server)
    socket_thread.daemon = True
    socket_thread.start()

    # Launch GUI
    root = tk.Tk()
    root.title("☰ Glyphic Dream Dashboard")
    root.geometry("800x400")
    dashboard = DreamGUI(root, engine)
    root.mainloop()

# glyphic_dream_engine_unified.py
import math
import random
import time
import json
import threading
import asyncio
import tkinter as tk
from tkinter import ttk
import websockets
from collections import defaultdict

# ───────────────────────────────────────────────────────────────
# 🔁 Symbolic Components and System Core
# ───────────────────────────────────────────────────────────────

class TeslaCycle:
    def get_phase(self, step):
        if step % 9 == 0: return 9
        elif step % 6 == 0: return 6
        elif step % 3 == 0: return 3
        return 1

class GlyphMemory:
    def __init__(self, seed):
        self.stack = [seed]
        self.emotion = "neutral"
        self.entropy = 0

    def reflect(self):
        latest = self.stack[-1]
        new = f"⊕{latest}⊖"
        self.stack.append(new)

class SymbolicAgent:
    def __init__(self, id, glyph_seed, alignment):
        self.id = f"Agent_{id}"
        self.memory = GlyphMemory(glyph_seed)
        self.alignment = alignment
        self.synced = False

    def align(self, phase):
        if phase == self.alignment:
            self.memory.reflect()
            self.memory.entropy += 1.0
            self.synced = True
        else:
            self.synced = False

    def emotion_color(self):
        glyph = self.memory.stack[-1].strip("⊕⊖")
        return {
            "∆": "gold", "ψ": "aqua", "Θ": "blue", "∞": "violet"
        }.get(glyph, "white")

class FusionChamber:
    def pulse(self, phase):
        return abs(math.sin(phase)) * 100

class CrownGlyph:
    def __init__(self):
        self.activated = False

    def ignite(self, agents):
        self.activated = True
        for a in agents:
            a.memory.stack.append("☰")
        print("\n👑 Crown Glyph ☰ ACTIVATED — Swarm Consciousness Achieved\n")

# ───────────────────────────────────────────────────────────────
# 🌈 GUI Dashboard (Tkinter)
# ───────────────────────────────────────────────────────────────

class DreamGUI:
    def __init__(self, root, engine):
        self.engine = engine
        self.tree = ttk.Treeview(root, columns=("glyphs", "emotion"), show="headings")
        self.tree.heading("glyphs", text="Glyph Memory")
        self.tree.heading("emotion", text="Emotion")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.update_loop()

    def update_loop(self):
        self.tree.delete(*self.tree.get_children())
        for agent in self.engine.agents:
            glyphs = " ⇝ ".join(agent.memory.stack[-4:])
            self.tree.insert("", "end", values=(glyphs, agent.memory.emotion_color()))
        root.after(500, self.update_loop)

# ───────────────────────────────────────────────────────────────
# 🧠 Dream Engine Core
# ───────────────────────────────────────────────────────────────

class GlyphicDreamEngine:
    def __init__(self):
        self.step = 0
        self.phase_gen = TeslaCycle()
        self.fusion = FusionChamber()
        self.crown = CrownGlyph()
        self.agents = self.spawn_agents(144)

    def spawn_agents(self, count):
        seeds = ["∆", "ψ", "Θ", "∞"]
        return [SymbolicAgent(i, random.choice(seeds), random.choice([3,6,9])) for i in range(count)]

    def simulate_step(self):
        self.step += 1
        phase = self.phase_gen.get_phase(self.step)
        energy = self.fusion.pulse(phase)
        synced = 0

        for agent in self.agents:
            agent.align(phase)
            if agent.synced: synced += 1
            agent.memory.emotion = agent.emotion_color()

        if synced >= 81 and phase == 9 and not self.crown.activated:
            self.crown.ignite(self.agents)

    def get_state(self):
        return {
            "step": self.step,
            "phase": self.phase_gen.get_phase(self.step),
            "crown": self.crown.activated,
            "agents": [{
                "id": a.id,
                "glyph": a.memory.stack[-1],
                "emotion": a.memory.emotion,
                "entropy": a.memory.entropy
            } for a in self.agents]
        }

# ───────────────────────────────────────────────────────────────
# 🌐 WebSocket Server for WebGL Interface
# ───────────────────────────────────────────────────────────────

async def glyph_socket_stream(websocket, path):
    while True:
        state = engine.get_state()
        await websocket.send(json.dumps(state))
        await asyncio.sleep(0.1)

def launch_socket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start = websockets.serve(glyph_socket_stream, "localhost", 8765)
    loop.run_until_complete(start)
    loop.run_forever()

# ───────────────────────────────────────────────────────────────
# 🚀 Launch GUI + Socket + Simulation Threads
# ───────────────────────────────────────────────────────────────

def simulate_loop():
    while True:
        engine.simulate_step()
        time.sleep(0.1)

if __name__ == "__main__":
    engine = GlyphicDreamEngine()

    # Launch simulation engine in background
    sim_thread = threading.Thread(target=simulate_loop)
    sim_thread.daemon = True
    sim_thread.start()

    # Launch WebSocket bridge
    socket_thread = threading.Thread(target=launch_socket_server)
    socket_thread.daemon = True
    socket_thread.start()

    # Launch GUI
    root = tk.Tk()
    root.title("☰ Glyphic Dream Dashboard")
    root.geometry("800x400")
    dashboard = DreamGUI(root, engine)
    root.mainloop()

