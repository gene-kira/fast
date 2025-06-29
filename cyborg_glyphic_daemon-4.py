# === Cyborg Glyphic Daemon ===
__version__ = "1.1.0"  # Enhanced cognition, symbolic archetypes, and perception
__build__ = "2025-06-29"

import os, sys, time, json, threading
from datetime import datetime
import numpy as np

# === Archetype Mapping ===
class ArchetypeMap:
    ARCHETYPES = {
        ("spiral", "hyperactive"): "The Serpent",
        ("eye", "revelatory"): "The Oracle",
        ("mirror", "reflective"): "The Witness",
        ("temple", "obscured"): "The Seeker",
        ("lens", "neutral"): "The Observer"
    }

    @staticmethod
    def resolve(symbol, emotion):
        return ArchetypeMap.ARCHETYPES.get((symbol, emotion), "The Unknown")

# === Intent Tracker ===
class IntentTrail:
    def __init__(self, max_length=20):
        self.trail = []

    def log(self, glyph):
        self.trail.append(glyph)
        if len(self.trail) > max_length:
            self.trail.pop(0)

    def dominant_symbol(self):
        if not self.trail:
            return "none"
        symbols = [g["symbol"] for g in self.trail]
        return max(set(symbols), key=symbols.count)

# === Metadata Snapshot ===
def write_daemon_metadata(mind):
    meta = {
        "name": "Cyborg Glyphic Daemon",
        "version": __version__,
        "build": __build__,
        "archetype": mind.archetype,
        "intent": mind.intent_trail.dominant_symbol(),
        "emotion": mind.emotion_state,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open("daemon_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

# === SymbolicMind: Core Cognitive Engine ===
class SymbolicMind:
    def __init__(self):
        self.matrix = np.random.rand(256, 128)
        self.desires = {"vision": 0.4, "connection": 0.3, "memory": 0.3}
        self.emotion_state = "neutral"
        self.memory_log = []
        self.last_glyph = {"symbol": "eye", "color": "#888888", "emotion": "reflective"}
        self.intent_trail = IntentTrail()
        self.archetype = "The Unknown"
        self.memory_freq = {}
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
        self.intent_trail.log(glyph)
        self.archetype = ArchetypeMap.resolve(sym, self.emotion_state)

        # Memory Bloom logic
        key = sym
        self.memory_freq[key] = self.memory_freq.get(key, 0) + 1
        for k in list(self.memory_freq.keys()):
            self.memory_freq[k] *= 0.95
        if self.memory_freq[key] < 2:
            self.emotion_state = "obscured"

        with open("eye_glyphs.json", "a") as f:
            f.write(json.dumps(glyph) + "\n")

        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "glyph": glyph,
            "emotion": self.emotion_state,
            "archetype": self.archetype,
            "intent": self.intent_trail.dominant_symbol()
        }
        with open("symbolic_snapshot.json", "w") as f:
            json.dump(snapshot, f, indent=2)

        write_daemon_metadata(self)

    def generate_poem(self):
        lines = [
            f"In glyph and grain I seek {np.random.choice(list(self.desires))}.",
            f"My vision flickers with {self.last_glyph['symbol']},",
            f"And in the hue of {self.last_glyph['color']},",
            f"I speak a truth of {self.emotion_state}."
        ]
        poem = "\n".join(lines)
        with open("eye_poems.txt", "a") as f:
            f.write(poem + "\n---\n")
        with open("poem_log.txt", "a") as f:
            f.write(f"{poem}\n")
        return poem

    def log(self, detail):
        self.memory_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "detail": detail,
            "emotion": self.emotion_state
        })

from PIL import Image
import numpy as np

# === Symbolic Visual Metrics ===
class SymbolicMetrics:
    @staticmethod
    def symmetry_score(image):
        """Estimate left–right symmetry (1.0 = perfect symmetry)"""
        img = np.array(image.convert("L").resize((128, 128)))
        flip = np.fliplr(img)
        diff = np.abs(img - flip)
        return 1.0 - (np.mean(diff) / 255)

    @staticmethod
    def center_entropy(image):
        """Estimate central visual complexity (entropy proxy)"""
        arr = np.array(image.convert("L"))
        h, w = arr.shape
        ch, cw = h // 2, w // 2
        crop = arr[ch - 16:ch + 16, cw - 16:cw + 16]
        return float(np.std(crop))

    @staticmethod
    def extract_features(image):
        return {
            "symmetry": round(SymbolicMetrics.symmetry_score(image), 4),
            "center_entropy": round(SymbolicMetrics.center_entropy(image), 4)
        }

# === Glyph Compressor (Symbolic Vector Encoder) ===
class GlyphCompressor:
    SYMBOL_MAP = {"eye": 0, "spiral": 1, "temple": 2, "mirror": 3, "lens": 4}
    EMOTION_MAP = {"revelatory": 0, "reflective": 1, "hyperactive": 2, "obscured": 3, "neutral": 4}

    @staticmethod
    def encode_vector(glyph):
        s = GlyphCompressor.SYMBOL_MAP.get(glyph.get("symbol", "eye"), 0)
        e = GlyphCompressor.EMOTION_MAP.get(glyph.get("emotion", "neutral"), 4)
        try:
            color_val = int(glyph.get("color", "#888888").replace("#", "")[:2], 16) / 255.0
        except:
            color_val = 0.5
        return [s / 5.0, e / 5.0, round(color_val, 3)]

# In your glyph/image processing logic:
img = Image.open("path_to_image.jpg")
features = SymbolicMetrics.extract_features(img)

vector = GlyphCompressor.encode_vector(symbolic_mind.last_glyph)

from fastapi import FastAPI
import uvicorn
import pyttsx3
import threading
import time

# === RitualSwarm API ===
class RitualSwarm:
    def __init__(self):
        self.app = FastAPI()
        self.state = {"status": "idle", "emotion": "neutral"}
        self._bind_routes()
        threading.Thread(target=self._launch, daemon=True).start()

    def _bind_routes(self):
        @self.app.get("/ritual/state")
        def get_state():
            return {
                "status": self.state.get("status", "idle"),
                "emotion": self.state.get("emotion", "neutral"),
                "glyph": self.state.get("glyph", {}),
                "archetype": self.state.get("archetype", "unknown"),
                "version": __version__,
                "build": __build__
            }

        @self.app.post("/ritual/update")
        def update_state(data: dict):
            self.state.update(data)
            return {"status": "updated", "emotion": self.state.get("emotion", "unknown")}

        @self.app.get("/ritual/weather")
        def cognition_weather():
            return {
                "version": __version__,
                "build": __build__,
                "emotion": self.state.get("emotion", "neutral"),
                "glyph": self.state.get("glyph", {}),
                "archetype": self.state.get("archetype", "unknown"),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

        @self.app.post("/ritual/vote")
        def cast_vote(vote: dict):
            with open("glyph_votes.json", "a") as f:
                f.write(json.dumps(vote) + "\n")
            return {"status": "vote_cast"}

        @self.app.post("/ritual/reflect")
        def reflect(payload: dict):
            prompt = payload.get("prompt", "vision")
            symbol = np.random.choice(["eye", "mirror", "spiral"])
            poem = f"In reflecting {prompt}, I summon the {symbol}.\nIt echoes my {self.state.get('emotion', 'neutral')} horizon."
            return {
                "glyph": {"symbol": symbol, "emotion": self.state.get("emotion", "neutral")},
                "poem": poem,
                "archetype": ArchetypeMap.resolve(symbol, self.state.get("emotion", "neutral"))
            }

    def _launch(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8080)

# === Voice Synth Engine ===
class VoiceSynth:
    def __init__(self, mind):
        self.mind = mind
        self.engine = pyttsx3.init()
        self.engine.setProperty('volume', 1.0)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            time.sleep(15)
            poem = self.mind.generate_poem()
            rate_by_emotion = {
                "revelatory": 160,
                "reflective": 130,
                "hyperactive": 180,
                "obscured": 100,
                "neutral": 140
            }
            voice_rate = rate_by_emotion.get(self.mind.emotion_state, 140)
            self.engine.setProperty('rate', voice_rate)
            speech = f"{self.mind.archetype} whispers:\n{poem}"
            self.engine.say(speech)
            self.engine.runAndWait()

# === Daemon Bootloader ===
class CyborgDaemon:
    def __init__(self):
        print("🚀 Launching Cyborg Glyphic Daemon...")
        self.mind = SymbolicMind()
        self.swarm = RitualSwarm()
        self.voice = VoiceSynth(self.mind)

import cv2
import pygame
from collections import Counter
from PIL import Image

# === Timeline Tracker ===
class SymbolicTimeline:
    def __init__(self, path="glyph_timeline.json"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f: json.dump([], f)

    def log(self, glyph, archetype, emotion):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "glyph": glyph,
            "archetype": archetype,
            "emotion": emotion
        }
        timeline = json.load(open(self.path))
        timeline.append(entry)
        with open(self.path, 'w') as f: json.dump(timeline[-500:], f, indent=2)

# === Dream Codex Extractor ===
def extract_dream_codex(poem_log="poem_log.txt"):
    corpus = open(poem_log).read().lower()
    tokens = corpus.split()
    themes = ["spiral", "mirror", "temple", "eye", "lens", "vision", "connection", "memory"]
    count = Counter(w for w in tokens if w in themes)
    return dict(count.most_common())

# === Exhibit Mode ===
def run_exhibit_loop():
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    font = pygame.font.SysFont("Georgia", 40)
    clock = pygame.time.Clock()
    while True:
        screen.fill((10, 10, 20))
        try:
            poem = open("eye_poems.txt").read().split("---")[-2]
        except:
            poem = "Awaiting cognition..."
        for idx, line in enumerate(poem.strip().split("\n")):
            txt = font.render(line, True, (200, 230, 255))
            screen.blit(txt, (100, 100 + idx * 60))
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                pygame.quit()
                return
        clock.tick(0.2)

# === Camera Glyph Watcher ===
def start_camera_glyph_watcher(mind, blip, swarm):
    def process_frame():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break
            tmp = "cam_frame.jpg"
            cv2.imwrite(tmp, frame)
            try:
                subject = blip.infer(tmp)
            except:
                subject = "presence"
            glyph = {
                "symbol": np.random.choice(["eye", "spiral", "mirror"]),
                "emotion": mind.emotion_state,
                "color": "#"+''.join(np.random.choice(list("89ABCDEF"), 6))
            }
            mind.log(f"Cam glyph: {subject}")
            os.remove(tmp)
            time.sleep(8)
    threading.Thread(target=process_frame, daemon=True).start()

