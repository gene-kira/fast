import os, sys, shutil, ctypes, subprocess

INSTALL_DIR = os.path.join(os.environ['ProgramFiles'], 'CyborgGlyphor')
ICON_PATH = os.path.join(INSTALL_DIR, 'cyborgGlyphor.ico')
EXE_PATH = os.path.join(INSTALL_DIR, 'cyborgGlyphor_launcher.pyw')
MODELS_SRC = os.path.join(os.getcwd(), 'models')
MODELS_DST = os.path.join(INSTALL_DIR, 'models')

# === Check for admin rights ===
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

# === Copy files to INSTALL_DIR ===
def install_payload():
    print("🧠 Installing CyborgGlyphor to:", INSTALL_DIR)
    os.makedirs(INSTALL_DIR, exist_ok=True)
    shutil.copy('cyborgGlyphor_launcher.pyw', EXE_PATH)
    shutil.copy('cyborgGlyphor.ico', ICON_PATH)
    if os.path.exists(MODELS_SRC):
        shutil.copytree(MODELS_SRC, MODELS_DST, dirs_exist_ok=True)
    else:
        print("⚠️ BLIP models folder not found!")

# === Register Windows Context Menu ===
def register_context_menu():
    for ext in ['.jpg', '.jpeg', '.png']:
        cmd_key = f'HKEY_CLASSES_ROOT\\SystemFileAssociations\\{ext}\\shell\\Invoke Cyborg Ritual'
        icon_path = f'"{ICON_PATH}"'
        exe_cmd = f'"{sys.executable}" "{EXE_PATH}" "%1"'
        subprocess.call(f'reg add "{cmd_key}" /ve /d "Invoke Cyborg Ritual" /f', shell=True)
        subprocess.call(f'reg add "{cmd_key}" /v Icon /d {icon_path} /f', shell=True)
        subprocess.call(f'reg add "{cmd_key}\\command" /ve /d "{exe_cmd}" /f', shell=True)
    print("✅ Context menu rituals registered.")

# === Main ===
if __name__ == "__main__":
    if not is_admin():
        print("🛡️ Relaunching with admin privileges...")
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
        sys.exit()

    install_payload()
    register_context_menu()
    print("\n✨ Ritual installation complete. Right-click any image to begin.")

import sys
from your_glyph_module import process_image

if len(sys.argv) > 1:
    process_image(sys.argv[1])
else:
    print("⚠️ No image path received.")

