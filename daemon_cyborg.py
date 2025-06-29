# daemon_cyborg.py
import subprocess, sys, importlib
import os, threading, time, json, hashlib
from datetime import datetime

# ========== AUTOLOADER ==========
REQUIRED = [
    'numpy', 'Pillow', 'scikit-learn', 'watchdog',
    'matplotlib', 'scipy', 'transformers', 'torch',
    'fastapi', 'uvicorn'
]

def install_and_import(pkg):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"📦 Installing: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)

mods = {pkg: install_and_import(pkg.split('.')[0]) for pkg in REQUIRED}

# ======= Imports from autoloaded modules =======
np = mods['numpy']
Image = mods['PIL'].Image
KMeans = mods['sklearn'].cluster.KMeans
ExifTags = mods['PIL'].ExifTags
plt = mods['matplotlib'].pyplot
generic_filter = mods['scipy'].ndimage.generic_filter
BlipProcessor = mods['transformers'].BlipProcessor
BlipForConditionalGeneration = mods['transformers'].BlipForConditionalGeneration
torch = mods['torch']
FastAPI = mods['fastapi'].FastAPI
BaseModel = mods['fastapi'].pydantic.BaseModel
uvicorn = mods['uvicorn']
Observer = mods['watchdog'].observers.Observer
FileSystemEventHandler = mods['watchdog'].events.FileSystemEventHandler

# ========== SUBJECT INFERENCE ==========
blip_model, blip_processor = None, None

def infer_subject_name(image_path):
    global blip_model, blip_processor
    if blip_model is None:
        print("🧠 Loading BLIP model...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(image, return_tensors="pt")
    out_ids = blip_model.generate(**inputs)
    return blip_processor.decode(out_ids[0], skip_special_tokens=True)

# ========== GLYPH ENTROPY ==========
def generate_entropy_map(img_path, export_dir="glyph_exports"):
    img = Image.open(img_path).convert("L")
    arr = np.array(img)
    ent = generic_filter(arr, np.std, size=5)
    norm = (ent - ent.min()) / (ent.ptp() + 1e-5)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(norm, cmap='inferno')
    ax.axis("off")
    os.makedirs(export_dir, exist_ok=True)
    fn = f"entropy_{hashlib.md5(open(img_path,'rb').read()).hexdigest()[:6]}.png"
    out_path = os.path.join(export_dir, fn)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return out_path

# ========== CORE ANALYSIS ==========
def core_analysis(path):
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    hist = img.histogram()
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "avg_color": tuple(arr[..., :3].mean(axis=(0, 1))),
        "entropy": {
            c: round(float(np.std(arr[..., i])), 2)
            for i, c in enumerate(['R', 'G', 'B'])
        },
        "dominant_colors": [tuple(map(int, c)) for c in KMeans(n_clusters=3).fit(arr[arr[...,3]>0][:,:3]).cluster_centers_],
        "pixel_hash": hashlib.sha256(arr.tobytes()).hexdigest(),
        "histogram": {'R': hist[0:256], 'G': hist[256:512], 'B': hist[512:768]}
    }

# ========== SYMBOLIC LAYER ==========
def symbolic_analysis(subject, avg_color, img_arr):
    def score_symmetry(a): return round(100 - np.mean(np.abs(a - np.flip(a, axis=1))), 2)
    r,g,b = avg_color
    color_map = ("sky", "forest", "fire")[np.argmax([b,g,r])]
    traits = {
        "sky": ["intelligence", "intuition"],
        "forest": ["resilience", "balance"],
        "fire": ["activation", "emergence"]
    }[color_map]
    return {
        "archetype_map": {
            "archetype": subject.title(),
            "traits": traits,
            "myth": f"{subject} as emergent glyph of complexity"
        },
        "color_semantic": color_map,
        "shape_signature": {
            "symmetry": score_symmetry(img_arr),
            "aspect_ratio": round(img_arr.shape[1]/img_arr.shape[0], 2)
        }
    }

# ========== SWARM LOGIC ==========
def swarm_output(payload):
    node_id = hashlib.sha256(json.dumps(payload['image_summary'], sort_keys=True).encode()).hexdigest()
    return {
        "identity_hash": node_id,
        "swarm_descriptor": {
            "@context": "https://schema.org",
            "@type": "CreativeWork",
            "identifier": node_id,
            "keywords": payload['symbolic_interpretation']['archetype_map']['traits'],
            "dateCreated": datetime.utcnow().isoformat()
        },
        "ritual_state": {
            "phase": "ignited",
            "sigil_ready": True,
            "message": f"{node_id} seeded into cognition web"
        }
    }

# ========== BINARY DISSECTION ==========
def extract_strings(path, min_length=4):
    b = open(path, "rb").read()
    out, temp = [], b""
    for byte in b:
        if 32 <= byte <= 126: temp += bytes([byte])
        else:
            if len(temp) >= min_length:
                out.append(temp.decode("ascii", errors="ignore"))
            temp = b""
    return out[:20]

def binary_analysis(path):
    return {
        "file_size": os.path.getsize(path),
        "sha256": hashlib.sha256(open(path,'rb').read()).hexdigest(),
        "ascii_strings_found": extract_strings(path),
        "hex_dump_sample": open(path, 'rb').read(320).hex()
    }

# ========== JSON PAYLOAD ==========
def run_pipeline(path, subject_name, entropy_path):
    img = Image.open(path).convert('RGBA')
    core = core_analysis(path)
    symbolic = symbolic_analysis(subject_name, core['avg_color'], np.array(img))
    payload = {
        "image_summary": {
            "path": path,
            "subject": subject_name,
            "avg_color": core["avg_color"]
        },
        "symbolic_interpretation": symbolic
    }
    swarm = swarm_output(payload)
    full = {
        "core_analysis": core,
        "symbolic_layer": symbolic,
        "swarm_node": swarm,
        "binary_dissector": binary_analysis(path),
        "entropy_glyph": entropy_path
    }
    out_file = f"cognition_{swarm['identity_hash'][:6]}.json"
    with open(out_file, 'w') as f:
        json.dump(full, f, indent=2)
    return full, out_file

# ========== REST API SWARM BROADCAST ==========
api_app = FastAPI()
current_state = {}

@api_app.get("/ritual/state")
def get_state():
    return current_state or {"status": "idle"}

@api_app.post("/ritual/update")
def update_state(data: dict):
    global current_state
    current_state = data
    return {"status": "received"}

def start_api():
    threading.Thread(target=lambda: uvicorn.run(api_app, host="0.0.0.0", port=8080), daemon=True).start()

# ========== FOLDER WATCHER ==========
class GlyphWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"👁️ New glyph: {event.src_path}")
            threading.Thread(target=process_image, args=(event.src_path,), daemon=True).start()

def process_image(path):
    try:
        subject = infer_subject_name(path)
        glyph = generate_entropy_map(path)
        result, out_file = run_pipeline(path, subject, glyph)
        with open("daemon_log.jsonl", "a") as log:
            log.write(json.dumps({"file": path, "subject": subject, "json": out_file}) + "\n")
    except Exception as e:
        print(f"⚠️ Failed: {e}")

def start_watch(folder="glyphs"):
    os.makedirs(folder, exist_ok=True)
    obs = Observer

