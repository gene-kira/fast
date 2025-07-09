# 🔧 Autoloader
import subprocess, sys
def autoload(pkg):
    try:
        globals()[pkg] = __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        globals()[pkg] = __import__(pkg)

# 📦 Required packages
for package in ["pygame", "pyttsx3"]:
    autoload(package)

# 🔑 Core modules
import tkinter as tk
import threading
import time
import math
import random
import json

# 🔊 Voice engine
pygame.mixer.init()
engine = pyttsx3.init()
voice_alert_active = True

# 🧬 Agent Registry
agent_registry = {
    "Sentinel_007": {"role": "Sentinel", "trust": 0.91},
    "Archivist_314": {"role": "Archivist", "trust": 0.88},
    "Seeker_453": {"role": "Seeker", "trust": 0.67}
}

# 🎭 Verdict Palette
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
def log_event(event_type, payload):
    entry = {
        "event": event_type,
        "details": payload,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("memory_graph.json", "a") as f:
        f.write(json.dumps(entry) + "\n")

# 🧠 Verdict Engine with Trait Inheritance
class PluralitySeal:
    def __init__(self, glyph_id, quorum_size=3, trust_threshold=0.7, timeout=30):
        self.glyph_id = glyph_id
        self.quorum_size = quorum_size
        self.trust_threshold = trust_threshold
        self.timeout = timeout
        self.votes = {}
        self.start_time = time.time()
        self.sealed = False

    def cast_vote(self, agent_id, verdict, comment=""):
        if self.sealed or verdict not in VERDICTS:
            return False
        agent = agent_registry.get(agent_id)
        if not agent or agent["trust"] < self.trust_threshold or agent_id in self.votes:
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
        counts = {}
        for vote in self.votes.values():
            v = vote["verdict"]
            counts[v] = counts.get(v, 0) + 1
        majority = max(counts, key=counts.get)
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

# 🖥️ Launch Ritual UI
def launch_seal_ui(seal: PluralitySeal):
    root = tk.Tk()
    root.title(f"Seal Ritual — {seal.glyph_id}")
    canvas = tk.Canvas(root, width=400, height=400, bg="black")
    canvas.pack()

    # 🔇 ACK Alert Button (top-level)
    def ack_alert():
        for agent_id, props in agent_registry.items():
            if props["role"] in ["Sentinel", "Archivist"] and props["trust"] > 0.7:
                global voice_alert_active
                voice_alert_active = False
                log_event("VoiceAlertAck", {"by": agent_id})
                print(f"🔇 Voice alert silenced by {agent_id}")
                break

    tk.Button(root, text="ACK Alert 🔇", command=ack_alert,
              bg="gray", fg="white", width=20).pack(pady=5)

    # 🗣️ Ritual startup voice
    if voice_alert_active:
        engine.say(f"Seal ritual initialized for {seal.glyph_id}")
        engine.runAndWait()

    # 🧿 Central Glyph
    glyph_core = canvas.create_oval(170, 170, 230, 230, fill="#2222FF", outline="white", width=3)
    canvas.create_text(200, 160, text=seal.glyph_id, fill="white", font=("Helvetica", 10))

    # ⏳ Countdown shimmer
    def update_countdown():
        while not seal.sealed:
            angle = ((seal.timeout - (time.time() - seal.start_time)) / seal.timeout) * 360
            canvas.delete("shimmer")
            canvas.create_arc(160, 160, 240, 240, start=90, extent=-angle,
                              outline="#8888FF", style="arc", width=2, tags="shimmer")
            time.sleep(0.1)
    threading.Thread(target=update_countdown, daemon=True).start()

    # 🔁 Radar animation for Tracking
    def animate_tracking_radar(cx, cy):
        def pulse(radius):
            ring = canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                      outline="#00AAFF", width=2)
            for _ in range(10):
                canvas.itemconfig(ring, width=random.randint(1, 3))
                canvas.update()
                time.sleep(0.04)
            canvas.delete(ring)
        for r in range(40, 140, 25):
            canvas.after(r * 2, lambda rad=r: pulse(rad))

    # ❌ Fragment implosion for Self-Destruction
    def animate_implosion_fragments(cx, cy):
        frags = []
        for _ in range(12):
            dx, dy = random.randint(-30, 30), random.randint(-30, 30)
            frag = canvas.create_oval(cx, cy, cx + 6, cy + 6, fill="#FF4444", outline="")
            frags.append((frag, dx, dy))
        for _ in range(15):
            for f, dx, dy in frags:
                canvas.move(f, dx / 15, dy / 15)
            canvas.update()
            time.sleep(0.05)
        for f, _, _ in frags:
            canvas.delete(f)

    # 🎭 Glyph morph choreography
    def morph_glyph(verdict):
        color = VERDICTS[verdict]["color"]
        emoji = VERDICTS[verdict]["emoji"]
        trait = VERDICTS[verdict]["trait"]

        canvas.itemconfig(glyph_core, fill=color)
        canvas.update()
        time.sleep(0.2)

        if verdict == "Home":
            for i in range(5):
                r = 30 + i * 4
                canvas.coords(glyph_core, 200 - r, 200 - r, 200 + r, 200 + r)
                canvas.update()
                time.sleep(0.1)
            canvas.create_text(200, 240, text="Sanctuary Established", fill="white", font=("Helvetica", 10))

        elif verdict == "Tracking":
            animate_tracking_radar(200, 200)
            canvas.create_text(200, 240, text="Beacon Deployed", fill="#00AAFF", font=("Helvetica", 10))

        elif verdict == "Self-Destruction":
            animate_implosion_fragments(200, 200)
            canvas.create_text(200, 240, text="Echo Erased", fill="#FF4444", font=("Helvetica", 10))

        canvas.create_text(200, 200, text=emoji, fill="white", font=("Helvetica", 26))
        canvas.create_text(200, 225, text=f"Sealed: {verdict}", fill="#DDDDDD", font=("Helvetica", 10))
        canvas.create_text(200, 255, text=f"Trait: {trait}", fill=color, font=("Helvetica", 10))

    # 💬 Cast vote handler
    def cast(agent_id, verdict, comment):
        if seal.cast_vote(agent_id, verdict, comment):
            idx = len(seal.votes) - 1
            angle = idx * 120
            x = 200 + 90 * math.cos(math.radians(angle))
            y = 200 + 90 * math.sin(math.radians(angle))
            emoji = VERDICTS[verdict]["emoji"]
            canvas.create_text(x, y, text=emoji, fill="white", font=("Helvetica", 20))
            canvas.create_line(x, y, 200, 200, fill=VERDICTS[verdict]["color"], width=2)
            canvas.create_text(x, y + 30, text=f"“{comment}”", fill="#CCCCCC", font=("Helvetica", 8))

    # 🎛️ Agent vote panels
    for aid in agent_registry:
        frame = tk.Frame(root, bg="black")
        frame.pack(pady=1)
        tk.Label(frame, text=f"{aid}:", fg="white", bg="black").pack(side="left", padx=5)
        entry = tk.Entry(frame, width=25)
        entry.pack(side="left")
        for verdict in VERDICTS:
            tk.Button(frame, text=verdict,
                      command=lambda a=aid, v=verdict, e=entry: cast(a, v, e.get()),
                      width=14, bg="darkblue", fg="white").pack(side="left", padx=2)

    # 🧬 Monitor and morph on seal
    def monitor_seal():
        while not seal.sealed:
            if len(seal.votes) >= seal.quorum_size:
                v = seal.resolve_consensus()
                if voice_alert_active:
                    engine.say(f"Verdict sealed: {v}")
                    engine.runAndWait()
                morph_glyph(v)
                break
            time.sleep(0.5)

    threading.Thread(target=monitor_seal, daemon=True).start()
    root.mainloop()

# 🚀 Launch Ritual
if __name__ == "__main__":
    glyph_id = "🧿_Glyph_003"
    seal = PluralitySeal(glyph_id=glyph_id, quorum_size=3)
    launch_seal_ui(seal)

