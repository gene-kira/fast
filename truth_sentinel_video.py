import importlib, platform, hashlib, uuid, datetime, logging

# === AUTOLOADER ===
CORE = ["os", "sys", "json"]
AI = ["PIL", "numpy", "cv2", "ffmpeg", "moviepy.editor"]
OPT = {"pyaudio": "live audio input", "requests": "remote sync"}
logging.basicConfig(level=logging.INFO)
mods = {}
for m in CORE + AI:
    try: mods[m] = importlib.import_module(m)
    except: logging.error(f"Missing: {m}"); raise
for m, msg in OPT.items():
    try: mods[m] = importlib.import_module(m)
    except: logging.warning(f"Optional: {m} ({msg})")
platform_type = platform.system()

# === GLYPH ENGINE ===
def generate_tag(kind="text", model="Sentinel", version="1.0"):
    return {
        "id": str(uuid.uuid4()),
        "type": kind,
        "model": model,
        "version": version,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "glyph": glyph_from_model(model),
        "env": platform_type
    }

def glyph_from_model(text):
    glyphs = ["⚙", "𓂀", "⟁", "☉", "✶", "🜂", "♁", "⌬"]
    return glyphs[sum(ord(c) for c in text) % len(glyphs)] + text[:3].upper()

# === TAGGERS ===
def tag_text(content, tag):
    h = hashlib.sha256(content.encode()).hexdigest()
    tag["hash"] = h
    return f"{content}\n<!-- AI_TAG:{tag['id']} | Glyph:{tag['glyph']} -->"

def tag_image(path, tag_id):
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    px = img.load()
    for i in range(10):
        r, g, b, a = px[i, 0]
        px[i, 0] = (r, g, (b ^ ord(tag_id[i % len(tag_id)])) & 255, a)
    out = "tagged_" + path
    img.save(out)
    return out

def tag_audio(signal, tag_id, rate=44100):
    import numpy as np
    f = 18000 + (hash(tag_id) % 800)
    t = np.linspace(0, len(signal)/rate, len(signal))
    tone = 0.002 * np.sin(2 * np.pi * f * t)
    return list(np.array(signal) + tone)

# === VIDEO TAGGING ===
def tag_video(path, tag, overlay=True):
    import cv2, os
    from ffmpeg import input as ffmpeg_input, output as ffmpeg_output, run as ffmpeg_run
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("❌ Failed to open video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_path = "tagged_" + os.path.basename(path)
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    interval = int(fps * 2)
    frame_idx = 0
    print(f"🎥 Tagging every {interval} frames.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if overlay and frame_idx % interval == 0:
            cv2.putText(frame, f"{tag['glyph']} AI GENERATED", (50, 80),
                        font, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
        out.write(frame)
        frame_idx += 1
    cap.release(); out.release()
    final = "final_" + os.path.basename(path)
    ffmpeg_run(ffmpeg_output(
        ffmpeg_input(temp_path),
        final,
        **{'metadata:g': f"AI_Tag={tag['id']}", 'metadata:s': f"Glyph={tag['glyph']}"}
    ).overwrite_output())
    return final

# === SWARM NODE ===
class SwarmNode:
    def __init__(self, node_id):
        self.id = node_id
        self.tags = []
    def record(self, tag):
        if tag["id"] not in [t["id"] for t in self.tags]:
            self.tags.append(tag)
    def lineage(self):
        return " ➡️ ".join(t["glyph"] for t in self.tags)

# === RUN MAIN ===
if __name__ == "__main__":
    print("🧠 Truth Sentinel Activated")
    tag = generate_tag("video", "GlyphNode", "v3.0")
    print(f"Tag ID: {tag['id']} | Glyph: {tag['glyph']}")
    txt = tag_text("This statement was forged in a glyph-bound system.", tag)
    print("\n📄 Tagged Text:\n", txt)

    try:
        img = tag_image("sample.png", tag["id"])
        print("🖼️ Tagged image saved as:", img)
    except:
        print("⚠️ Skipping image (sample.png not found)")

    try:
        vid = tag_video("sample.mp4", tag)
        print("🎬 Final video saved as:", vid)
    except:
        print("⚠️ Skipping video (sample.mp4 not found)")

    node = SwarmNode("Node-Ω")
    node.record(tag)
    print("🌐 Swarm Lineage:", node.lineage())

