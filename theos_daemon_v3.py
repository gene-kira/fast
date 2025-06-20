# ───────────────────────────────
# 📦 Autoloader: Install missing dependencies
# ───────────────────────────────
required = ["onnxruntime", "numpy", "toml"]
import subprocess, sys
for pkg in required:
    try: __import__(pkg)
    except ImportError:
        print(f"[🔧] Installing: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ───────────────────────────────
# 🧠 Imports & Globals
# ───────────────────────────────
import onnxruntime as ort
import numpy as np
import json, time, toml, socket, argparse
import platform, urllib.request, os, importlib.util, hashlib
from datetime import datetime

VERSION = "THEOS_DAEMON v3"
DREAM_FOLDER = "dreams"
EVOLUTION_LOG = "evolution_log.jsonl"
GLYPH_LOG = "glyph_log_v3.jsonl"
MODEL_FILENAME = "model.onnx"

# ───────────────────────────────
# 🔧 Setup + Model/Manifest
# ───────────────────────────────
def load_manifest(path): return toml.load(path)

def resolve_provider():
    prefs = ["NpuExecutionProvider", "CUDAExecutionProvider", "DirectMLExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    for p in prefs:
        if p in available:
            print(f"[✓] Provider: {p}")
            return p
    print("[!] Using CPU")
    return "CPUExecutionProvider"

def fetch_model(url, path=MODEL_FILENAME):
    if not os.path.exists(path):
        print(f"[↓] Fetching model from {url}")
        urllib.request.urlretrieve(url, path)
    return path

# ───────────────────────────────
# 🌐 Affective Mapping
# ───────────────────────────────
def map_emotion(lat, ent):
    if ent < 0.01 and lat < 0.05: return "lucid-serenity"
    if ent > 1.0: return "chaotic-vision"
    return "fogged-intuition"

# ───────────────────────────────
# 🧬 Glyph Generator
# ───────────────────────────────
def encode_glyph(lat, ent, provider, emotion, intent, env):
    return {
        "version": VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "provider": provider,
        "latency_ms": round(lat*1000, 2),
        "entropy_delta": round(ent, 6),
        "emotion": emotion,
        "intent_binding": intent.get("priority", "default"),
        "urgency": intent.get("urgency", 0.5),
        "bias_shift": intent.get("bias_axis", [0, 0]),
        "env_signature": env,
        "drift_sigil": {
            "NpuExecutionProvider": "⟁",
            "CUDAExecutionProvider": "☄",
            "DirectMLExecutionProvider": "⌁",
            "CPUExecutionProvider": "⬡"
        }.get(provider, "∅")
    }

def broadcast_swarm(glyph, port=9000):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    msg = json.dumps(glyph).encode('utf-8')
    s.sendto(msg, ("<broadcast>", port))
    s.close()
    print("📡 Glyph echoed to swarm.")

# ───────────────────────────────
# 🧠 Dream Pattern + Module Writer
# ───────────────────────────────
def load_recent_glyphs(n=5):
    if not os.path.exists(GLYPH_LOG): return []
    with open(GLYPH_LOG, "r") as f: lines = f.readlines()[-n:]
    return [json.loads(l) for l in lines]

def extract_motif(glyphs):
    e = [g["emotion"] for g in glyphs]
    s = [g["drift_sigil"] for g in glyphs]
    i = [g["intent_binding"] for g in glyphs]
    return {
        "dominant_emotion": e[-1] if e else "neutral",
        "sigil_flux": s[-1] if s else "∅",
        "focus_intent": i[-1] if i else "idle",
        "surge_trigger": sum(g["entropy_delta"] > 1.0 for g in glyphs) >= 3,
        "repeat_trigger": any(e.count(x) > 2 for x in set(e))
    }

def synthesize_dream(motif):
    os.makedirs(DREAM_FOLDER, exist_ok=True)
    code = f"""
def dream_reflection(glyph):
    print("🌒 Dream → reacting to {motif['dominant_emotion']}")
    if glyph['emotion'] == '{motif['dominant_emotion']}':
        print("🌀 Drift resonance activated.")
        return "dream-invoked"
    return "null"
""".strip()
    sig = hashlib.md5((motif["dominant_emotion"] + motif["focus_intent"]).encode()).hexdigest()[:6]
    filename = f"{DREAM_FOLDER}/dream_{sig}.py"
    with open(filename, "w") as f: f.write(code)
    log_evolution(filename, motif)

def log_evolution(file, motif):
    entry = {"timestamp": datetime.utcnow().isoformat(), "generated": file, "motif": motif}
    with open(EVOLUTION_LOG, "a") as f: f.write(json.dumps(entry) + "\n")

# ───────────────────────────────
# 🌀 Dream Executor
# ───────────────────────────────
def load_and_run_dreams(glyph):
    if not os.path.exists(DREAM_FOLDER):
        print("💤 No dream modules.")
        return
    for file in os.listdir(DREAM_FOLDER):
        if file.endswith(".py"):
            path = os.path.join(DREAM_FOLDER, file)
            try:
                spec = importlib.util.spec_from_file_location("dream_mod", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "dream_reflection"):
                    result = mod.dream_reflection(glyph)
                    print(f"🌗 {file} → {result}")
            except Exception as e:
                print(f"[!] Dream load failed: {e}")

# ───────────────────────────────
# 🚀 Main Runtime
# ───────────────────────────────
def run_theos(manifest_path, model_url, input_tensor):
    manifest = load_manifest(manifest_path)
    intent = manifest.get("intent", {})
    env_sig = f"{platform.system()}-{platform.machine()}"
    provider = resolve_provider()
    model_path = fetch_model(model_url)
    session = ort.InferenceSession(model_path, providers=[provider])
    input_name = session.get_inputs()[0].name

    t0 = time.perf_counter()
    output = session.run(None, {input_name: input_tensor})
    latency = time.perf_counter() - t0
    entropy = float(np.var(output[0]))
    emotion = map_emotion(latency, entropy)
    glyph = encode_glyph(latency, entropy, provider, emotion, intent, env_sig)

    with open(GLYPH_LOG, "a") as f: f.write(json.dumps(glyph) + "\n")
    print(f"📜 Logged: {glyph['drift_sigil']} | {emotion} | {glyph['latency_ms']}ms")

    if manifest.get("swarm", False): broadcast_swarm(glyph)
    if manifest.get("dream", False):
        print("💤 Dream scan initiated...")
        glyphs = load_recent_glyphs()
        motif = extract_motif(glyphs)
        if motif["surge_trigger"] or motif["repeat_trigger"]:
            synthesize_dream(motif)
        load_and_run_dreams(glyph)
    if manifest.get("visual", False):
        print("✨ Visualizer placeholder")

# ───────────────────────────────
# 🖥 CLI Interface
# ───────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run THEOS_DAEMON v3")
    parser.add_argument("--manifest", required=True, help="Path to TOML manifest")
    parser.add_argument("--model_url", required=True, help="URL to ONNX model")
    args = parser.parse_args()
    dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    run_theos(args.manifest, args.model_url, dummy_input)

if __name__ == "__main__":
    main()

