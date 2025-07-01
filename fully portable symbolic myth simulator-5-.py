# symbolic_autoloader.py – Autoloader for Symbolic Simulator

import sys
import os
import time
import random
import datetime
import traceback
from collections import defaultdict

# ─── Optional Module Imports ───
OPTIONAL_LIBRARIES = {
    "tkinter": False,
    "matplotlib.pyplot": False,
    "asyncio": False,
    "json": False,
    "numpy": False
}

def try_import(module_name, alias=None):
    try:
        globals()[alias or module_name.split('.')[-1]] = __import__(module_name)
        OPTIONAL_LIBRARIES[module_name] = True
    except ImportError:
        print(f"[AUTOLOAD] Optional module '{module_name}' not found.")

try_import("tkinter")
try_import("matplotlib.pyplot", alias="plt")
try_import("asyncio")
try_import("json")
try_import("numpy")

# ─── Feature Flags ───
FEATURES = {
    "USE_GUI": OPTIONAL_LIBRARIES.get("tkinter", False),
    "USE_PLOTS": OPTIONAL_LIBRARIES.get("matplotlib.pyplot", False),
    "USE_ASYNC": OPTIONAL_LIBRARIES.get("asyncio", False),
    "USE_NUMPY": OPTIONAL_LIBRARIES.get("numpy", False),
    "ENABLE_EXPORT_JSON": OPTIONAL_LIBRARIES.get("json", False),
}

def print_autoloader_status():
    print("\n[🔍 AUTOLOADER STATUS]")
    for lib, loaded in OPTIONAL_LIBRARIES.items():
        status = "✓ Loaded" if loaded else "⨯ Missing"
        print(f"  {lib:<20} → {status}")

def print_feature_flags():
    print("\n[🧩 FEATURE FLAGS]")
    for feat, enabled in FEATURES.items():
        mark = "✅" if enabled else "❌"
        print(f"  {mark} {feat}")

# ─── Module Registry ───
class ModuleRegistry:
    def __init__(self):
        self.modules = []

    def register(self, name, obj):
        self.modules.append((name, obj))
        print(f"[MODULE] Registered '{name}'")

    def list_all(self):
        print("\n[📚 Registered Modules]")
        for name, _ in self.modules:
            print(f"  - {name}")

# ─── Utility Functions ───
def graceful_exit(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

def warn(msg):
    print(f"[WARN] {msg}")

def read_file(path):
    if not os.path.exists(path):
        warn(f"File '{path}' does not exist.")
        return None
    with open(path, "r") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"[SAVE] Written to '{path}'")

def save_json(data, path="output.json"):
    if not FEATURES["ENABLE_EXPORT_JSON"]:
        warn("JSON export not available. Install 'json' module.")
        return
    try:
        import json
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[EXPORT] JSON saved: {path}")
    except Exception as e:
        warn(f"Could not export JSON: {e}")

def resolve_path(filename, subfolder="exports"):
    root = os.getcwd()
    folder = os.path.join(root, subfolder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(folder, filename)

# ─── Runtime Info ───
def get_session_id():
    now = datetime.datetime.now()
    return f"session-{now.strftime('%Y%m%d-%H%M%S')}"

def print_runtime_info():
    print("\n[🧭 RUNTIME INFO]")
    print(f"  Session ID: {get_session_id()}")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    print(f"  Working Dir: {os.getcwd()}")

# ─── Config Loader ───
def load_config(path="config.json"):
    if not FEATURES["ENABLE_EXPORT_JSON"]:
        warn("JSON config not supported.")
        return {}
    try:
        import json
        with open(path, "r") as f:
            config = json.load(f)
        print(f"[CONFIG] Loaded from {path}")
        return config
    except Exception as e:
        warn(f"Failed to load config: {e}")
        return {}

# ─── Hot Reload Utility ───
def reload_modules(modules):
    for m in modules:
        try:
            import importlib
            importlib.reload(m)
            print(f"[RELOAD] Reloaded: {m.__name__}")
        except Exception as e:
            warn(f"Could not reload {m}: {e}")

# ─── Autoloader – Core Essentials & Diagnostics ───
import sys
import os
import time
import random
import datetime
import traceback
from collections import defaultdict

# ─── Optional Modules ───
OPTIONAL_LIBRARIES = {
    "tkinter": False,
    "matplotlib.pyplot": False,
    "asyncio": False,
    "json": False,
    "numpy": False
}

def try_import(module_name, alias=None):
    try:
        globals()[alias or module_name.split('.')[-1]] = __import__(module_name)
        OPTIONAL_LIBRARIES[module_name] = True
    except ImportError:
        print(f"[AUTOLOAD] Optional module '{module_name}' not found.")

# Load optional utilities
try_import("tkinter")
try_import("matplotlib.pyplot", alias="plt")
try_import("asyncio")
try_import("json")
try_import("numpy")

# ─── System Boot Check ───
def print_autoloader_status():
    print("\n[🔍 AUTOLOADER STATUS]")
    for lib, loaded in OPTIONAL_LIBRARIES.items():
        status = "✓ Loaded" if loaded else "⨯ Missing"
        print(f"  {lib:<20} → {status}")

# ─── Plugin System & Dependency Flags ───

# You can toggle features by setting these flags
FEATURES = {
    "USE_GUI": OPTIONAL_LIBRARIES.get("tkinter", False),
    "USE_PLOTS": OPTIONAL_LIBRARIES.get("matplotlib.pyplot", False),
    "USE_ASYNC": OPTIONAL_LIBRARIES.get("asyncio", False),
    "USE_NUMPY": OPTIONAL_LIBRARIES.get("numpy", False),
    "ENABLE_EXPORT_JSON": OPTIONAL_LIBRARIES.get("json", False),
}

def print_feature_flags():
    print("\n[🧩 FEATURE FLAGS]")
    for feat, enabled in FEATURES.items():
        mark = "✅" if enabled else "❌"
        print(f"  {mark} {feat}")

# ─── Auto-Mount Symbolic Interfaces ───

class ModuleRegistry:
    def __init__(self):
        self.modules = []

    def register(self, name, obj):
        self.modules.append((name, obj))
        print(f"[MODULE] Registered '{name}'")

    def list_all(self):
        print("\n[📚 Registered Modules]")
        for name, _ in self.modules:
            print(f"  - {name}")

# ─── Fallback Interfaces ───

def graceful_exit(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

def warn(msg):
    print(f"[WARN] {msg}")

# ─── File Utilities ───

def read_file(path):
    if not os.path.exists(path):
        warn(f"File '{path}' does not exist.")
        return None
    with open(path, "r") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"[SAVE] Written to '{path}'")

def save_json(data, path="output.json"):
    if not FEATURES["ENABLE_EXPORT_JSON"]:
        warn("JSON export not available. Install 'json' module.")
        return
    try:
        import json
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[EXPORT] JSON saved: {path}")
    except Exception as e:
        warn(f"Could not export JSON: {e}")

# ─── Path Resolver ───

def resolve_path(filename, subfolder="exports"):
    root = os.getcwd()
    folder = os.path.join(root, subfolder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(folder, filename)

 ─── Runtime Identity ───

def get_session_id():
    now = datetime.datetime.now()
    return f"session-{now.strftime('%Y%m%d-%H%M%S')}"

def print_runtime_info():
    print("\n[🧭 RUNTIME INFO]")
    print(f"  Session ID: {get_session_id()}")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    print(f"  Working Dir: {os.getcwd()}")

# ─── Config Loader ───

def load_config(path="config.json"):
    if not FEATURES["ENABLE_EXPORT_JSON"]:
        warn("JSON config not supported.")
        return {}
    try:
        import json
        with open(path, "r") as f:
            config = json.load(f)
        print(f"[CONFIG] Loaded from {path}")
        return config
    except Exception as e:
        warn(f"Failed to load config: {e}")
        return {}

# ─── Code Reload Utility ───

def reload_modules(modules):
    for m in modules:
        try:
            import importlib
            importlib.reload(m)
            print(f"[RELOAD] Reloaded module: {m.__name__}")
        except Exception as e:
            warn(f"Could not reload {m}: {e}")

