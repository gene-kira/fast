# 🔧 Autoloader
import subprocess, sys
def autoload(package):
    try:
        globals()[package] = __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        globals()[package] = __import__(package)

# 📦 Load required libraries
for pkg in ["pygame", "pyttsx3"]:
    autoload(pkg)

import tkinter as tk
import threading
import time
import math
import json

pygame.mixer.init()
engine = pyttsx3.init()
voice_alert_active = True

# 🧬 Agent Registry
agent_registry = {
    "Sentinel_007": {"role": "Sentinel", "trust": 0.91},
    "Archivist_314": {"role": "Archivist", "trust": 0.88},
    "Seeker_453": {"role": "Seeker", "trust": 0.67}
}

# 🧿 Verdict Palette (Renamed)
VERDICTS = {
    "Home": {"emoji": "🏅", "color": "#00FFAA", "trait": "Resilient"},
    "Tracking": {"emoji": "🔁", "color": "#8888FF", "trait": "Mutable"},
    "Self-Destruction": {"emoji": "❌", "color": "#FF4444", "trait": "Anomaly"}
}

# 📜 Glyph Lineage Memory
glyph_lineage = {
    "🧿_Glyph_001": {"ancestry": ["🌀_Initiation"], "verdicts": ["Home"], "traits": ["Resilient"]},
    "🧿_Glyph_002": {"ancestry": ["🧿_Glyph_001"], "verdicts": ["Tracking"], "traits": ["Mutable"]}
}

# 🗃️ Memory Logger
def log_event(event_type, data):
    entry = {
        "event": event_type,
        "details": data,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("memory_graph.json", "a") as f:
        f.write(json.dumps(entry) + "\n")

# 🧠 Verdict Sealer with Trait Inheritance
class PluralitySeal:
    def __init__(self, glyph_id, quorum_size=3, trust_threshold=0.7, timeout=30):
        self.glyph_id = glyph_id
        self.quorum_size = quorum_size
        self.trust_threshold = trust_threshold
        self.timeout = timeout
        self.votes = {}  # agent → {verdict, comment}
        self.start_time = time.time()
        self.sealed = False

    def cast_vote(self, agent_id, verdict, comment=""):
        if self.sealed or verdict not in VERDICTS:
            return False
        agent = agent_registry.get(agent_id)
        if not agent or agent["trust"] < self.trust_threshold:
            return False
        if agent_id in self.votes:
            return False
        self.votes[agent_id] = {
            "verdict": verdict,
            "comment": comment,
            "trust": agent["trust"],
            "role": agent["role"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return True

    def resolve_consensus(self):
        if len(self.votes) < self.quorum_size:
            return None
        verdict_count = {}
        for vote in self.votes.values():
            v = vote["verdict"]
            verdict_count[v] = verdict_count.get(v, 0) + 1
        majority = max(verdict_count, key=verdict_count.get)
        minority = [v["verdict"] for v in self.votes.values() if v["verdict"] != majority]

        glyph = glyph_lineage.get(self.glyph_id, {"ancestry": [], "verdicts": [], "traits": []})
        glyph["verdicts"].append(majority)
        glyph["traits"].append(VERDICTS[majority]["trait"])
        glyph["traits"] += [f"EchoOf:{v}" for v in set(minority)]
        glyph_lineage[self.glyph_id] = glyph

        log_event("PluralitySealConsensus", {
            "glyph": self.glyph_id,
            "majority": majority,
            "votes": self.votes,
            "residue": list(set(minority)),
            "traits_added": glyph["traits"]
        })
        self.sealed = True
        return majority

# 🖥️ Ritual UI Launcher
def launch_seal_ui(seal: PluralitySeal):
    root = tk.Tk()
    root.title(f"Seal Ritual — {seal.glyph_id}")
    canvas = tk.Canvas(root, width=400, height=400, bg="black")
    canvas.pack()

    # 🔇 ACK Voice Alert Button
    def ack_alert():
        for agent_id, props in agent_registry.items():
            if props["role"] in ["Sentinel", "Archivist"] and props["trust"] > 0.7:
                global voice_alert_active
                voice_alert_active = False
                log_event("VoiceAlertAck", {"by": agent_id})
                print(f"🔇 Voice alert silenced by {agent_id}")
                break

    ack_btn = tk.Button(root, text="ACK Alert 🔇", command=ack_alert, bg="gray", fg="white", width=20)
    ack_btn.pack(pady=5)

    # 🗣️ Voice Alert on Init
    if voice_alert_active:
        engine.say(f"Seal ritual initialized for {seal.glyph_id}")
        engine.runAndWait()

    # 🧿 Central Glyph Node
    glyph_core = canvas.create_oval(170, 170, 230, 230, fill="#2222FF", outline="white", width=3)
    canvas.create_text(200, 160, text=seal.glyph_id, fill="white", font=("Helvetica", 10))

    # ⏳ Countdown Arc
    def update_countdown():
        while not seal.sealed:
            elapsed = time.time() - seal.start_time
            remaining = max(0, seal.timeout - elapsed)
            angle = (remaining / seal.timeout) * 360
            canvas.delete("shimmer")
            canvas.create_arc(160, 160, 240, 240, start=90, extent=-angle,
                              outline="#8888FF", style="arc", width=2, tags="shimmer")
            time.sleep(0.1)

    threading.Thread(target=update_countdown, daemon=True).start()

    # 💬 Vote Handler with Commentary
    def cast(agent_id, verdict, comment):
        if seal.cast_vote(agent_id, verdict, comment):
            idx = len(seal.votes) - 1
            angle = idx * 120
            x = 200 + 90 * math.cos(math.radians(angle))
            y = 200 + 90 * math.sin(math.radians(angle))
            emoji = VERDICTS[verdict]["emoji"]
            orb = canvas.create_text(x, y, text=emoji, fill="white", font=("Helvetica", 20))
            trail = canvas.create_line(x, y, 200, 200, fill=VERDICTS[verdict]["color"], width=2)
            rationale = canvas.create_text(x, y + 30, text=f"“{comment}”", fill="#CCCCCC", font=("Helvetica", 8))

    # 🎛️ Agent Panels
    for agent_id in agent_registry.keys():
        frame = tk.Frame(root, bg="black")
        frame.pack(pady=1)
        tk.Label(frame, text=f"{agent_id}:", fg="white", bg="black").pack(side="left", padx=5)
        entry = tk.Entry(frame, width=25)
        entry.pack(side="left")

        for verdict in VERDICTS.keys():
            tk.Button(
                frame,
                text=verdict,
                command=lambda a=agent_id, v=verdict, e=entry: cast(a, v, e.get()),
                width=14, bg="darkblue", fg="white"
            ).pack(side="left", padx=2)

    # 🎭 Verdict Morph Animation
    def morph_glyph(verdict):
        color = VERDICTS[verdict]["color"]
        emoji = VERDICTS[verdict]["emoji"]

        # Animate morph
        for i in range(10):
            r = 30 + i * 2
            canvas.coords(glyph_core, 200 - r, 200 - r, 200 + r, 200 + r)
            canvas.itemconfig(glyph_core, fill=color)
            canvas.update()
            time.sleep(0.05)

        canvas.itemconfig(glyph_core, fill=color)
        canvas.create_text(200, 200, text=emoji, fill="white", font=("Helvetica", 26))
        canvas.create_text(200, 225, text=f"Sealed: {verdict}", fill="#DDDDDD", font=("Helvetica", 10))

    # 🔄 Monitor Consensus
    def monitor_seal():
        while not seal.sealed:
            if len(seal.votes) >= seal.quorum_size:
                majority = seal.resolve_consensus()
                if voice_alert_active:
                    engine.say(f"Verdict sealed: {majority}")
                    engine.runAndWait()
                morph_glyph(majority)
                break
            time.sleep(0.5)

    threading.Thread(target=monitor_seal, daemon=True).start()
    root.mainloop()

# 🚀 Ritual Launch Example
if __name__ == "__main__":
    glyph_id = "🧿_Glyph_003"
    seal = PluralitySeal(glyph_id=glyph_id, quorum_size=3)
    launch_seal_ui(seal)

