# loader.py
import subprocess
import sys
import importlib

REQUIRED_PACKAGES = [
    'numpy',
    'Pillow',
    'scikit-learn'
]

def install_and_import(package):
    try:
        return importlib.import_module(package)
    except ImportError:
        print(f"📦 Installing missing package: {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(package)

def autoload_dependencies():
    modules = {}
    for pkg in REQUIRED_PACKAGES:
        short = pkg.split('.')[0]
        modules[short] = install_and_import(pkg)
    return modules

# core.py
from loader import autoload_dependencies
modules = autoload_dependencies()

np = modules['numpy']
Image = modules['PIL'].Image
KMeans = modules['sklearn'].cluster.KMeans
from PIL import ExifTags
import hashlib, datetime

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
    arr = arr[arr[:, 3] > 0][:, :3]  # opaque only
    kmeans = KMeans(n_clusters=k).fit(arr)
    return [tuple(map(int, c)) for c in kmeans.cluster_centers_]

def pixel_signature(img):
    arr = np.array(img)
    return hashlib.sha256(arr.tobytes()).hexdigest()

def histogram_data(img):
    hist = img.histogram()
    return {'R': hist[0:256], 'G': hist[256:512], 'B': hist[512:768]}

def edge_map_placeholder(img):
    return "Sketch contour placeholder"

def lsb_check_placeholder(img):
    return False  # Extend with steg analysis as needed

def core_analysis(path):
    img = load_image(path)
    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "exif": extract_exif(img),
        "avg_color": avg_color(img),
        "entropy": entropy_per_channel(img),
        "dominant_colors": dominant_colors(img),
        "pixel_hash": pixel_signature(img),
        "histogram": histogram_data(img),
        "edges": edge_map_placeholder(img),
        "lsb_flag": lsb_check_placeholder(img)
    }

# symbolic.py
from loader import autoload_dependencies
modules = autoload_dependencies()

np = modules['numpy']

def symbolic_animal_mapping(entity):
    map = {
        "bear": {
            "archetype": "The Guardian",
            "traits": ["protection", "strength", "solitude"],
            "myth": "Shamanic and Norse traditions view the bear as a guardian at thresholds"
        }
    }
    return map.get(entity.lower(), {})

def color_to_symbolic(color_rgb):
    r, g, b = color_rgb
    if g > r and g > b:
        return "forest → nature, guardianship, balance"
    elif r > g and r > b:
        return "fire → passion, vigilance"
    elif b > r and b > g:
        return "sky → intelligence, intuition"
    return "neutral"

def symmetry_score(image_arr):
    h_flip = np.flip(image_arr, axis=1)
    diff = np.abs(image_arr - h_flip)
    return round(100 - np.mean(diff), 2)

def shape_grammar_signature(image_arr):
    return {
        "symmetry": symmetry_score(image_arr),
        "aspect_ratio": round(image_arr.shape[1] / image_arr.shape[0], 2),
        "signature_type": "radial-symmetric glyph"
    }

def jungian_archetype_link(entity):
    return {
        "bear": {
            "shadow": "anger, overprotection",
            "persona": "strength, leadership",
            "anima": "nurturing wisdom",
            "self": "threshold guardian"
        }
    }.get(entity.lower(), {})

def symbolic_analysis(img, summary):
    subject = summary.get("subject", "unknown")
    color = summary.get("avg_color", (0, 0, 0))

    return {
        "archetype_map": symbolic_animal_mapping(subject),
        "color_semantic": color_to_symbolic(color),
        "shape_signature": shape_grammar_signature(np.array(img)),
        "jungian_node": jungian_archetype_link(subject)
    }

# swarm.py
from loader import autoload_dependencies
autoload_dependencies()

import hashlib, json
from datetime import datetime

def node_identity_fingerprint(payload):
    key_data = json.dumps(payload["image_summary"], sort_keys=True)
    return hashlib.sha256(key_data.encode('utf-8')).hexdigest()

def cognition_descriptor(payload):
    traits = payload.get("symbolic_interpretation", {})
    hash_ = node_identity_fingerprint(payload)

    return {
        "node_id": hash_,
        "traits": traits.get("archetype_map", {}).get("traits", []),
        "archetype": traits.get("archetype_map", {}).get("archetype", "undefined"),
        "creation_time": datetime.utcnow().isoformat() + "Z"
    }

def swarm_broadcast_template(descriptor):
    return {
        "@context": "https://schema.org",
        "@type": "CreativeWork",
        "name": descriptor["archetype"] + " Node",
        "identifier": descriptor["node_id"],
        "keywords": descriptor["traits"],
        "dateCreated": descriptor["creation_time"]
    }

def init_ritual_sequence(node_descriptor):
    return {
        "ritual_phase": "init",
        "sigil_ready": True,
        "message": f"Node {node_descriptor['node_id']} is attuned and symbolically seeded."
    }

def full_swarm_output(payload):
    node = cognition_descriptor(payload)
    return {
        "identity_hash": node["node_id"],
        "swarm_descriptor": swarm_broadcast_template(node),
        "ritual_state": init_ritual_sequence(node)
    }

