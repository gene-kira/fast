# ╔════════════════════════════════════════════════════════════╗
# ║     EchoStream Ascension Core – Real-Time Phase I Build    ║
# ║  ▸ Full Code, Autoloader, Drive File Access, Mood Engine   ║
# ║  ▸ Killer666 Cinematic Framework, Modular & Expandable     ║
# ╚════════════════════════════════════════════════════════════╝

# 📦 AUTOLOADER
import importlib
import subprocess
import sys

def ensure_library(lib_name, pip_name=None):
    try:
        return importlib.import_module(lib_name)
    except ImportError:
        print(f"[AUTOLOAD] Installing {lib_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or lib_name])
        return importlib.import_module(lib_name)

required_libs = {
    "psutil": "psutil",
    "os": "os",
    "time": "time",
    "random": "random",
    "glob": "glob"
}

loaded_modules = {}
for lib, pip_name in required_libs.items():
    loaded_modules[lib] = ensure_library(lib, pip_name)

# 🔁 Library Handles
os = loaded_modules["os"]
time = loaded_modules["time"]
random = loaded_modules["random"]
psutil = loaded_modules["psutil"]
glob = loaded_modules["glob"]

# 🧠 Mood Engine
class MoodEngine:
    def __init__(self):
        self.states = ['Neutral', 'Alert', 'Euphoric', 'Threatened']
        self.current_state = 'Neutral'
        self.intensity = 0.5

    def update_state(self, external_factor=None):
        if external_factor == "override":
            self.current_state = 'Threatened'
            self.intensity = 1.0
        else:
            self.current_state = random.choice(self.states)
            self.intensity = round(random.uniform(0.3, 0.9), 2)
        print(f"[MOOD] ➤ {self.current_state} @ Intensity {self.intensity}")

    def get_priority_weight(self):
        return {
            'Neutral': 1.0,
            'Alert': 1.4,
            'Euphoric': 0.8,
            'Threatened': 2.0
        }.get(self.current_state, 1.0)

# 🌐 Real-Time File Drive Wrapper
class LogicalDrive:
    def __init__(self, path):
        self.path = path
        self.label = os.path.basename(path)
        self.refresh_stats()

    def refresh_stats(self):
        usage = psutil.disk_usage(self.path)
        self.capacity = usage.total
        self.free_space = usage.free
        self.last_checked = time.time()

    def get_block_path(self, block_id):
        return os.path.join(self.path, f"echostream_block_{block_id}.txt")

    def read_block(self, block_id):
        file_path = self.get_block_path(block_id)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"[READ] Block {block_id} → {file_path}")
            return content
        except FileNotFoundError:
            print(f"[MISS] Block {block_id} not found → {file_path}")
            return None

    def write_block(self, block_id, content):
        file_path = self.get_block_path(block_id)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[WRITE] Block {block_id} → {file_path}")
        except Exception as e:
            print(f"[ERROR] Could not write to {file_path}: {e}")

# 🌐 Drive Controller
class DriveNetworkController:
    def __init__(self, target_dirs=None):
        self.drives = []
        self.scan_drives(target_dirs)

    def scan_drives(self, target_dirs=None):
        partitions = psutil.disk_partitions(all=False)
        scanned = []

        if target_dirs:
            scanned = [LogicalDrive(p) for p in target_dirs if os.path.exists(p)]
        else:
            scanned = [LogicalDrive(p.mountpoint) for p in partitions if p.fstype and os.path.exists(p.mountpoint)]

        self.drives = scanned if scanned else []
        print(f"[SCAN] {len(self.drives)} drives connected.")

    def activity_loop(self, mood_engine, cycles=5, blocks_range=(0,100)):
        for cycle in range(cycles):
            print(f"\n🌌 Echo Cycle {cycle + 1}")
            mood_engine.update_state()
            for drive in self.drives:
                block_id = random.randint(*blocks_range)
                mode = "write" if random.random() < 0.5 else "read"
                drive.refresh_stats()
                if mode == "read":
                    drive.read_block(block_id)
                else:
                    content = f"{drive.label}-Data-{block_id}-Mood-{mood_engine.current_state}-{time.time()}"
                    drive.write_block(block_id, content)
            time.sleep(0.3)

# 🔁 Execution
if __name__ == "__main__":
    # 🔍 Optional: Specify exact folders to treat as logical drives
    target_paths = [
        os.path.expanduser("~/EchoDrive1"),
        os.path.expanduser("~/EchoDrive2")
    ]

    # 🔧 Create directories if missing
    for path in target_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    mood_core = MoodEngine()
    network_core = DriveNetworkController(target_dirs=target_paths)
    network_core.activity_loop(mood_core, cycles=6, blocks_range=(0,30))

