# === Symbolic Visual Metrics ===
class SymbolicMetrics:
    @staticmethod
    def symmetry_score(image):
        img = np.array(image.convert("L").resize((128, 128)))
        flip = np.fliplr(img)
        diff = np.abs(img - flip)
        return 1.0 - (np.mean(diff) / 255)

    @staticmethod
    def center_entropy(image):
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

# === GlyphCompressor ===
class GlyphCompressor:
    SYMBOL_MAP = {"eye": 0, "spiral": 1, "temple": 2, "mirror": 3, "lens": 4}
    EMOTION_MAP = {"revelatory": 0, "reflective": 1, "hyperactive": 2, "obscured": 3, "neutral": 4}

    @staticmethod
    def encode_vector(glyph):
        s = GlyphCompressor.SYMBOL_MAP.get(glyph.get("symbol", "eye"), 0)
        e = GlyphCompressor.EMOTION_MAP.get(glyph.get("emotion", "neutral"), 4)
        c = int(glyph.get("color", "#888888").replace("#", "")[:2], 16) / 255.0
        return [s / 5.0, e / 5.0, c]

# === Expanded RitualSwarm ===
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

    def _launch(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8080)

# === Voice Modulation Update ===
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
        engine.setProperty('volume', 1.0)
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
            engine.setProperty('rate', voice_rate)
            speech = f"{self.mind.archetype} whispers:\n{poem}"
            engine.say(speech)
            engine.runAndWait()

<pre><code># === Timeline Logger ===class SymbolicTimeline:def __init__(self, path="glyph_timeline.json"):self.path = pathif not os.path.exists(self.path):with open(self.path, 'w') as f: json.dump([], f)def log(self, glyph, archetype, emotion):entry = {"timestamp": datetime.utcnow().isoformat(),"glyph": glyph,"archetype": archetype,"emotion": emotion}timeline = json.load(open(self.path))timeline.append(entry)with open(self.path, 'w') as f: json.dump(timeline[-500:], f, indent=2)  # keep last 500# === Swarm Voting Stub ===# Add to RitualSwarm._bind_routes()@self.app.post("/ritual/vote")def cast_vote(vote: dict):with open("glyph_votes.json", "a") as f:f.write(json.dumps(vote) + "\n")return {"status": "vote_cast"}# === Poetic Echo Archive (Add to SymbolicMind.generate_poem) ===with open("poem_log.txt", "a") as f:f.write(f"{poem}\n")# === Memory Bloom Logic (Add to SymbolicMind.__init__) ===self.memory_freq = {}# ...Inside forge_glyph, after archetype resolution:key = symself.memory_freq[key] = self.memory_freq.get(key, 0) + 1for k in list(self.memory_freq.keys()):self.memory_freq[k] *= 0.95  # decayif self.memory_freq[key] < 2:self.emotion_state = "obscured"# === Dream Codex Extractor ===def extract_dream_codex(poem_log="poem_log.txt"):corpus = open(poem_log).read().lower()from collections import Countersymbols = ["spiral", "mirror", "lens", "temple", "eye"]desires = ["vision", "memory", "connection"]themes = symbols + desirescount = Counter(w for w in corpus.split() if w in themes)return dict(count.most_common())# === Exhibit Mode (fullscreen ritual) ===def run_exhibit_loop():import pygamepygame.init()screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)font = pygame.font.SysFont("Arial", 48)poem_font = pygame.font.SysFont("Georgia", 36)clock = pygame.time.Clock()i = 0while True:poem = f"{time.ctime()}\n" + open("eye_poems.txt").read().split("---")[-2]screen.fill((10, 10, 20))for idx, line in enumerate(poem.split("\n")):txt = poem_font.render(line.strip(), True, (200, 220, 255))screen.blit(txt, (100, 100 + idx * 50))pygame.display.flip()for event in pygame.event.get():if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:pygame.quit()returnclock.tick(0.05)# === Camera Watcher ===def start_camera_glyph_watcher(mind, blip, swarm):def process_frame():cap = cv2.VideoCapture(0)while True:ret, frame = cap.read()if not ret: breaktmp = "cam_frame.jpg"cv2.imwrite(tmp, frame)subject = blip.infer(tmp)glyph = {"symbol": np.random.choice(["spiral", "eye", "temple"]),"emotion": mind.emotion_state,"color": "#"+''.join(np.random.choice(list("89ABCDEF"), 6))}mind.log(f"Cam glyph: {subject}")os.remove(tmp)time.sleep(8)threading.Thread(target=process_frame, daemon=True).start()# === Reflective REST Endpoint === (add to RitualSwarm._bind_routes())@self.app.post("/ritual/reflect")def reflect(payload: dict):prompt = payload.get("prompt", "vision")symbol = np.random.choice(["eye", "mirror", "spiral"])poem = f"In reflecting {prompt}, I summon the {symbol}.\nIt echoes my {self.state.get('emotion', 'neutral')} horizon."return {"glyph": {"symbol": symbol, "emotion": self.state.get("emotion", "neutral")},"poem": poem,"archetype": ArchetypeMap.resolve(symbol, self.state.get("emotion", "neutral"))}</code></pre>