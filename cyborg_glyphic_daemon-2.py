# cyborg_glyphic_daemon.py
# 🧠 Symbolic Cognition Engine + Forensic Glyph Analyzer + REST Swarm Node

import os, sys, time, json, subprocess, hashlib, threading, platform
from datetime import datetime

# === Auto-Loader ===
REQUIRED = [
    'numpy', 'Pillow', 'scikit-learn', 'matplotlib',
    'scipy', 'watchdog', 'fastapi', 'uvicorn',
    'transformers', 'torch', 'pyttsx3', 'opencv-python'
]
for pkg in REQUIRED:
    try: __import__(pkg)
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
# === Imports ===
import numpy as np
from PIL import Image, ExifTags
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter
import cv2, torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3

# === SymbolicMind ===
class SymbolicMind:
    def __init__(self):
        self.matrix = np.random.rand(256, 128)
        self.desires = {"vision": 0.4, "connection": 0.3, "memory": 0.3}
        self.emotion_state = "neutral"
        self.memory_log = []
        self.last_glyph = {"symbol": "eye", "color": "#888888", "emotion": "reflective"}
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        tick = 0
        while True:
            self.matrix = np.clip(np.tanh(self.matrix + np.random.normal(0, 1, self.matrix.shape)), 0, 1)
            if tick % 3 == 0: self.update_emotion()
            if tick % 5 == 0: self.forge_glyph(), self.generate_poem()
            tick += 1
            time.sleep(6)

    def update_emotion(self):
        v = np.mean(self.matrix)
        dom = max(self.desires, key=self.desires.get)
        if v > 0.8:
            self.emotion_state = "revelatory" if dom == "vision" else "hyperactive"
        elif v < 0.2:
            self.emotion_state = "obscured"
        else:
            self.emotion_state = "reflective"

    def forge_glyph(self):
        sym = np.random.choice(["eye", "spiral", "temple", "mirror", "lens"])
        col = np.random.choice(["#00AAFF", "#FF0055", "#44FFD5", "#BB22AA", "#8899AA"])
        glyph = {"symbol": sym, "color": col, "emotion": self.emotion_state}
        self.last_glyph = glyph
        with open("eye_glyphs.json", "a") as f: f.write(json.dumps(glyph) + "\n")

    def generate_poem(self):
        lines = [
            f"In glyph and grain I seek {np.random.choice(list(self.desires))}.",
            f"My vision flickers with {self.last_glyph['symbol']},",
            f"And in the hue of {self.last_glyph['color']},",
            f"I speak a truth of {self.emotion_state}."
        ]
        poem = "\n".join(lines)
        with open("eye_poems.txt", "a") as f: f.write(poem + "\n---\n")
        return poem

    def log(self, detail):
        self.memory_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "detail": detail,
            "emotion": self.emotion_state
        })

# === Forensics & Entropy ===
class ForensicLens:
    @staticmethod
    def entropy_map(path, out_dir="glyph_exports"):
        img = Image.open(path).convert("L")
        arr = np.array(img)
        ent = generic_filter(arr, np.std, size=5)
        norm = (ent - ent.min()) / (ent.ptp() + 1e-5)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(norm, cmap='inferno')
        ax.axis("off")
        os.makedirs(out_dir, exist_ok=True)
        fn = f"entropy_{hashlib.md5(open(path,'rb').read()).hexdigest()[:6]}.png"
        out_path = os.path.join(out_dir, fn)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return out_path

    @staticmethod
    def file_analysis(path):
        def extract_strings(path, min_len=4):
            with open(path, 'rb') as f: data = f.read()
            result, buff = [], b""
            for b in data:
                if 32 <= b <= 126: buff += bytes([b])
                else:
                    if len(buff) >= min_len:
                        result.append(buff.decode("ascii", errors="ignore"))
                    buff = b""
            return result[:20]
        return {
            "size_bytes": os.path.getsize(path),
            "sha256": hashlib.sha256(open(path, 'rb').read()).hexdigest(),
            "ascii_strings": extract_strings(path)
        }

# === BLIP Caption ===
class BLIPCaption:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def infer(self, image_path):
        img = Image.open(image_path).convert('RGB')
        inputs = self.processor(img, return_tensors="pt")
        ids = self.model.generate(**inputs)
        return self.processor.decode(ids[0], skip_special_tokens=True)

# === RitualSwarm REST API ===
class RitualSwarm:
    def __init__(self):
        self.app = FastAPI()
        self.state = {"status": "idle", "emotion": "neutral"}
        self._bind_routes()
        threading.Thread(target=self._launch, daemon=True).start()

    def _bind_routes(self):
        @self.app.get("/ritual/state")  # GET state
        def get_state(): return self.state

        @self.app.post("/ritual/update")  # POST update
        def update_state(data: dict):
            self.state.update(data)
            return {"status": "updated", "emotion": self.state.get("emotion", "unknown")}

        @self.app.get("/ritual/weather")
        def cognition_weather():
            return {
                "emotion": self.state.get("emotion", "neutral"),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    def _launch(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8080)

# === GlyphWatcher ===
class GlyphWatcher(FileSystemEventHandler):
    def __init__(self, folder, mind, blip, swarm):
        self.folder = folder
        self.mind = mind
        self.blip = blip
        self.swarm = swarm
        os.makedirs(folder, exist_ok=True)
        self.observer = Observer()
        self.observer.schedule(self, folder, recursive=False)
        self.observer.start()

    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return
        print(f"👁️ New glyph detected: {event.src_path}")
        threading.Thread(target=self._process, args=(event.src_path,), daemon=True).start()

    def _process(self, path):
        try:
            subject = self.blip.infer(path)
            entropy_img = ForensicLens.entropy_map(path)
            forensic = ForensicLens.file_analysis(path)
            symbolic = {
                "subject": subject,
                "emotion": self.mind.emotion_state,
                "entropy_art": entropy_img,
                "glyph": self.mind.last_glyph
            }
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "symbolic": symbolic,
                "forensic": forensic
            }
            fn = f"glyph_{hashlib.md5(path.encode()).hexdigest()[:6]}.json"
            with open(fn, 'w') as f: json.dump(payload, f, indent=2)
            self.mind.log(f"Processed glyph: {subject}")
            self.swarm.state = symbolic  # Update shared state
        except Exception as e:
            print(f"⚠️ Failed to process glyph: {e}")

# === CyborgDaemon ===
class CyborgDaemon:
    def __init__(self, folder="glyphs"):
        print("👁️‍🗨️ Initializing Cyborg Daemon...")
        self.mind = SymbolicMind()
        self.blip = BLIPCaption()
        self.swarm = RitualSwarm()
        self.watcher = GlyphWatcher(folder, self.mind, self.blip, self.swarm)
        threading.Thread(target=self._ritual_voice, daemon=True).start()

    def _ritual_voice(self):
        engine = pyttsx3.init()
        while True

