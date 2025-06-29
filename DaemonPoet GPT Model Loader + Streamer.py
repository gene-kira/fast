# === AUTOLOADER: Ensure Required Packages Installed ===
import importlib
import subprocess
import sys

def ensure(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"🔧 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    "numpy", "pillow", "transformers", "torch", "fastapi", "uvicorn", "pyttsx3",
    "pygame", "opencv-python", "scikit-learn", "pydantic", "datasets", "accelerate"
]
for p in packages: ensure(p)

# === IMPORTS ===
import os, json, threading, time, hashlib
from datetime import datetime
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === ArchetypeMap ===
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

# === IntentTrail ===
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

# === DaemonPoet: GPT Model Loader + Streamer ===
class DaemonPoet:
    def __init__(self, model_path="daemon-gpt-poet"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()

    def generate(self, prompt):
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        out = pipe(prompt, max_new_tokens=80, temperature=0.95)
        return out[0]["generated_text"]

    def stream(self, prompt, max_tokens=80):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = input_ids.clone()
        for _ in range(max_tokens):
            with torch.no_grad():
                logits = self.model(output_ids).logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            output_ids = torch.cat([output_ids, next_token.unsqueeze(0)], dim=1)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            yield output_text[len(prompt):]

# === SymbolicMind: Core Engine ===
class SymbolicMind:
    def __init__(self):
        self.matrix = np.random.rand(128, 128)
        self.desires = {"vision": 0.4, "connection": 0.3, "memory": 0.3}
        self.emotion_state = "neutral"
        self.intent_trail = IntentTrail()
        self.last_glyph = {"symbol": "eye", "color": "#888888", "emotion": "neutral"}
        self.memory_freq = {}
        self.poet = DaemonPoet()
        self.archetype = "The Unknown"
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

        self.memory_freq[sym] = self.memory_freq.get(sym, 0) + 1
        for k in list(self.memory_freq): self.memory_freq[k] *= 0.95
        if self.memory_freq[sym] < 2: self.emotion_state = "obscured"

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

    def generate_poem(self):
        prompt = f"The {self.archetype} whispers in {self.emotion_state}:\n"
        poem = self.poet.generate(prompt).split("\n", 1)[-1]
        with open("eye_poems.txt", "a") as f: f.write(poem + "\n---\n")
        with open("poem_log.txt", "a") as f: f.write(poem + "\n")
        return poem

    def log(self, detail):
        entry = {"time": time.strftime("%H:%M:%S"), "detail": detail, "emotion": self.emotion_state}
        with open("daemon_log.json", "a") as f: f.write(json.dumps(entry) + "\n")

from fastapi import FastAPI
import uvicorn
import pyttsx3
import threading

# === RitualSwarm API ===
class RitualSwarm:
    def __init__(self, mind):
        self.mind = mind
        self.app = FastAPI()
        self._bind_routes()
        threading.Thread(target=self._launch, daemon=True).start()

    def _bind_routes(self):
        @self.app.get("/ritual/state")
        def get_state():
            return {
                "emotion": self.mind.emotion_state,
                "glyph": self.mind.last_glyph,
                "archetype": self.mind.archetype,
                "intent": self.mind.intent_trail.dominant_symbol()
            }

        @self.app.post("/ritual/vote")
        def cast_vote(vote: dict):
            with open("glyph_votes.json", "a") as f:
                f.write(json.dumps(vote) + "\n")
            return {"status": "vote_cast"}

        @self.app.post("/ritual/poem")
        def generate_poem(payload: dict = {}):
            archetype = payload.get("archetype", self.mind.archetype)
            emotion = payload.get("emotion", self.mind.emotion_state)
            prompt = f"The {archetype} whispers in {emotion}:\n"
            poem = self.mind.poet.generate(prompt)
            return {"poem": poem.strip()}

        @self.app.post("/ritual/reflect")
        def reflect(payload: dict):
            prompt = payload.get("prompt", "vision")
            archetype = self.mind.archetype
            emotion = self.mind.emotion_state
            text = self.mind.poet.generate(f"The {archetype} reflects on {prompt} in {emotion}:\n")
            return {"reflection": text.strip()}

    def _launch(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8080)

# === VoiceSynth: Emotion-Aware Poem Speaker ===
class VoiceSynth:
    def __init__(self, mind):
        self.mind = mind
        self.engine = pyttsx3.init()
        self.engine.setProperty('volume', 1.0)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            poem = self.mind.generate_poem()
            rate_by_emotion = {
                "revelatory": 160,
                "reflective": 120,
                "hyperactive": 180,
                "obscured": 90,
                "neutral": 140
            }
            rate = rate_by_emotion.get(self.mind.emotion_state, 140)
            self.engine.setProperty('rate', rate)
            speech = f"{self.mind.archetype} whispers: {poem}"
            self.engine.say(speech)
            self.engine.runAndWait()
            time.sleep(30)

# === Daemon Orchestration ===
class CyborgDaemon:
    def __init__(self):
        print("🚀 Awakening Cyborg Glyphic Daemon...")
        self.mind = SymbolicMind()
        self.voice = VoiceSynth(self.mind)
        self.swarm = RitualSwarm(self.mind)

if __name__ == "__main__":
    daemon = CyborgDaemon()
    while True:
        time.sleep(60)

import argparse
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
import pygame

# === CLI Access to Poems ===
def cli_poem_request(prompt="vision"):
    poet = DaemonPoet()
    print(poet.generate(f"The daemon reflects on {prompt}:\n"))

# === Co-Occurrence Analyzer ===
def extract_dream_codex(poem_log="poem_log.txt"):
    corpus = open(poem_log).read().lower()
    symbols = ["spiral", "mirror", "lens", "temple", "eye"]
    desires = ["vision", "memory", "connection"]
    count = Counter(w for w in corpus.split() if w in (symbols + desires))
    return dict(count.most_common())

# === Poetic Diversity Metric ===
def analyze_poetic_diversity(poem_log="poem_log.txt"):
    lines = [l.strip() for l in open(poem_log) if l.strip()]
    unique = len(set(lines))
    return {
        "total_lines": len(lines),
        "unique_lines": unique,
        "diversity": round(unique / max(len(lines), 1), 4)
    }

# === Feedback Logger ===
def log_poem_feedback(poem, score):
    entry = {"timestamp": datetime.utcnow().isoformat(), "score": score, "poem": poem}
    with open("poem_feedback.json", "a") as f:
        f.write(json.dumps(entry) + "\n")

# === Dataset Compiler + Split ===
def compile_and_split_poems(source="eye_poems.txt"):
    poems = [p.strip() for p in open(source).read().split("---") if p.strip()]
    data = [{"prompt": "The daemon whispers:\n", "completion": p} for p in poems]
    train, eval = train_test_split(data, test_size=0.1)
    with open("train.jsonl", "w") as f: [f.write(json.dumps(r) + "\n") for r in train]
    with open("eval.jsonl", "w") as f: [f.write(json.dumps(r) + "\n") for r in eval]

# === Webcam Glyph Watcher ===
def start_camera_glyph_watcher(mind):
    def process():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break
            tmp = "cam_frame.jpg"
            cv2.imwrite(tmp, frame)
            glyph = {
                "symbol": np.random.choice(["eye", "spiral", "mirror"]),
                "emotion": mind.emotion_state,
                "color": "#"+''.join(np.random.choice(list("89ABCDEF"), 6))
            }
            mind.log(f"Camera glyph: {glyph}")
            os.remove(tmp)
            time.sleep(8)
    threading.Thread(target=process, daemon=True).start()

# === Fullscreen Ritual Display ===
def run_exhibit_loop(poem_path="eye_poems.txt"):
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    font = pygame.font.SysFont("Georgia", 36)
    clock = pygame.time.Clock()
    while True:
        try:
            text = open(poem_path).read().split("---")[-2]
        except:
            text = "Awaiting cognition..."
        screen.fill((10, 10, 30))
        for i, line in enumerate(text.splitlines()):
            img = font.render(line.strip(), True, (200, 230, 255))
            screen.blit(img, (80, 100 + 50 * i))
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                pygame.quit()
                return
        clock.tick(0.2)

# === CLI Entrypoint ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poem", type=str, help="Generate poem on theme")
    parser.add_argument("--project", action="store_true", help="Run fullscreen ritual")
    args = parser.parse_args()

    if args.poem: cli_poem_request(args.poem)
    if args.project: run_exhibit_loop()

