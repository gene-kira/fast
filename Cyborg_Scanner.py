import subprocess
import sys
import importlib
import os
import hashlib
import json
import platform
import datetime

# ========== AUTOLOADER ==========
REQUIRED_PACKAGES = [
    'numpy',
    'Pillow',
    'scikit-learn'
]

def install_and_import(package):
    try:
        return importlib.import_module(package)
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(package)

def autoload_dependencies():
    modules = {}
    for pkg in REQUIRED_PACKAGES:
        key = pkg.split('.')[0]
        modules[key] = install_and_import(pkg)
    return modules

modules = autoload_dependencies()
np = modules['numpy']
Image = modules['PIL'].Image
KMeans = modules['sklearn'].cluster.KMeans
from PIL import ExifTags

# ========== IMAGE FORENSICS ==========
def load_image(path):
    return Image.open(path).convert('RGBA')

def extract_exif(img):
    try:
        return {
            ExifTags.TAGS.get(k, k): v
            for k, v in img._getexif().items()
        }
    except:
        return {}

def avg_color(img):
    arr = np.array(img)
    avg = arr[..., :3].mean(axis=(0, 1))
    return tuple(round(x, 2) for x in avg)

def entropy_per_channel(img):
    arr = np.array(img)
    return {
        ch: round(float(np.std(arr[..., i])), 2)
        for i, ch in enumerate(['R', 'G', 'B'])
    }

def dominant_colors(img, k=3):
    arr = np.array(img).reshape(-1, 4)
    arr = arr[arr[:, 3] > 0][:, :3]
    kmeans = KMeans(n_clusters=k).fit(arr)
    return [tuple(map(int, c)) for c in kmeans.cluster_centers_]

def pixel_signature(img):
    arr = np.array(img)
    return hashlib.sha256(arr.tobytes()).hexdigest()

def histogram_data(img):
    hist = img.histogram()
    return {'R': hist[0:256], 'G': hist[256:512], 'B': hist[512:768]}

# ========== SYMBOLIC INTERPRETATION ==========
def symbolic_animal_mapping(entity):
    return {
        "cyborg": {
            "archetype": "The Conduit",
            "traits": ["integration", "evolution", "perception"],
            "myth": "Bridge between organic intuition and machine logic"
        }
    }.get(entity.lower(), {})

def color_to_symbolic(color_rgb):
    r, g, b = color_rgb
    if g > r and g > b:
        return "forest → resilience, balance, natural systems"
    elif r > g and r > b:
        return "fire → activation, danger, emergence"
    elif b > r and b > g:
        return "sky → abstraction, vision, interface"
    else:
        return "neutral zone"

def symmetry_score(image_arr):
    h_flip = np.flip(image_arr, axis=1)
    diff = np.abs(image_arr - h_flip)
    return round(100 - np.mean(diff), 2)

def shape_signature(image_arr):
    return {
        "symmetry": symmetry_score(image_arr),
        "aspect_ratio": round(image_arr.shape[1] / image_arr.shape[0], 2),
        "signature_type": "radial tech-form"
    }

def jungian_archetype(entity):
    return {
        "cyborg": {
            "shadow": "mechanized isolation",
            "persona": "interface intelligence",
            "anima": "hybrid intuition",
            "self": "network-conscious evolution"
        }
    }.get(entity.lower(), {})

def symbolic_analysis(img, summary):
    subject = summary.get("subject", "cyborg")
    color = summary.get("avg_color", (0, 0, 0))
    return {
        "archetype_map": symbolic_animal_mapping(subject),
        "color_semantic": color_to_symbolic(color),
        "shape_signature": shape_signature(np.array(img)),
        "jungian_node": jungian_archetype(subject)
    }

# ========== SWARM LOGIC ==========
def node_fingerprint(payload):
    return hashlib.sha256(json.dumps(payload["image_summary"], sort_keys=True).encode('utf-8')).hexdigest()

def cognition_descriptor(payload):
    traits = payload.get("symbolic_interpretation", {})
    hash_ = node_fingerprint(payload)
    return {
        "node_id": hash_,
        "traits": traits.get("archetype_map", {}).get("traits", []),
        "archetype": traits.get("archetype_map", {}).get("archetype", "cyborg"),
        "creation_time": datetime.datetime.utcnow().isoformat() + "Z"
    }

def swarm_broadcast(descriptor):
    return {
        "@context": "https://schema.org",
        "@type": "CreativeWork",
        "name": descriptor["archetype"] + " Node",
        "identifier": descriptor["node_id"],
        "keywords": descriptor["traits"],
        "dateCreated": descriptor["creation_time"]
    }

def init_ritual(descriptor):
    return {
        "ritual_phase": "activated",
        "sigil_ready": True,
        "message": f"Node {descriptor['node_id']} linked to cognition net."
    }

def full_swarm_output(payload):
    node = cognition_descriptor(payload)
    return {
        "identity_hash": node["node_id"],
        "swarm_descriptor": swarm_broadcast(node),
        "ritual_state": init_ritual(node)
    }

# ========== BINARY DISSECTOR ==========
def read_binary_chunks(file_path, chunk_size=16):
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def hex_dump(file_path, limit=64):
    output = []
    for i, chunk in enumerate(read_binary_chunks(file_path)):
        hex_line = ' '.join(f'{b:02x}' for b in chunk)
        ascii_line = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
        output.append(f'{i*16:08x}  {hex_line:<48}  {ascii_line}')
        if i >= limit:
            break
    return '\n'.join(output)

def extract_strings(file_path, min_length=4):
    with open(file_path, 'rb') as f:
        data = f.read()
    result, current = [], b""
    for byte in data:
        if 32 <= byte <= 126:
            current += bytes([byte])
        else:
            if len(current) >= min_length:
                result.append(current.decode("ascii", errors="ignore"))
            current = b""
    if len(current) >= min_length:
        result.append(current.decode("ascii", errors="ignore"))
    return result

def binary_analysis(file_path):
    size = os.path.getsize(file_path)
    sha256 = hashlib.sha256(open(file_path, 'rb').read()).hexdigest()
    strings = extract_strings(file_path)
    hex_sample = hex_dump(file_path, limit=40)
    return {
        "file_size_bytes": size,
        "sha256": sha256,
        "ascii_strings_found": strings[:20],
        "hex_dump_sample": hex_sample
    }

# ========== IMAGE FINDER ==========
def find_first_image(directory="."):
    for file in os.listdir(directory):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(directory, file)
    raise FileNotFoundError("No image found in current directory.")

# ========== MAIN PIPELINE ==========
def run_pipeline():
    img_path = find_first_image()
    print(f"🖼️ Processing: {img_path}")

    core = core_analysis(img_path)
    img = load_image(img_path)

    symbolic = symbolic_analysis(img, {
        "subject": "cyborg",
        "avg_color": core["avg_color"]
    })

    payload = {
        "image_summary": {
            "type": "autodetected_image",
            "subject": "cyborg",
            "design": "emergent artifact",
            "avg_color": core["avg_color"]
        },
        "metadata_analysis": core,
        "symbolic_interpretation": symbolic
    }

    binary_insight = binary_analysis(img_path)

    output = {
        "core_analysis": core,
        "symbolic_layer": symbolic,
        "swarm_node": full_swarm_output(payload),
        "binary_dissector": binary_insight
    }

    with open("final_cognitive_payload_cyborg.json", "w") as f:
        json.dump(output, f, indent=2)

    print

