import os
import subprocess
import requests
from pathlib import Path
import zipfile
import winreg
import win32com.client

# Constants
APP_NAME = "Qwen_CoPilot"
APP_DIR = Path.home() / "AppData" / "Local" / APP_NAME
QWEN_MODEL_URL = "https://example.com/qwen_model.zip"

def ensure_directory_exists(directory):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def download_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

def copy_qwen_model():
    model_zip_path = APP_DIR / "qwen_model.zip"
    try:
        download_file(QWEN_MODEL_URL, model_zip_path)
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(APP_DIR / "model")
        os.remove(model_zip_path)
    except Exception as e:
        print(f"Error copying Qwen model: {e}")

def install_phylan():
    try:
        subprocess.run(["pip", "install", "phylan"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing Phylan: {e}")

def create_taskbar_icon():
    icon_path = APP_DIR / "model" / "qwen.exe"
    registry_key = r'Software\Classes\Directory\shell\Qwen_CoPilot'

    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, registry_key) as key:
            winreg.SetValueEx(key, '', 0, winreg.REG_SZ, 'Open with Qwen CoPilot')
            context_menu_key = f'{registry_key}\\command'
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, context_menu_key) as subkey:
                winreg.SetValueEx(subkey, '', 0, winreg.REG_SZ, f'"{icon_path}" "%1"')
    except Exception as e:
        print(f"Error creating taskbar icon: {e}")

def create_shortcut():
    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(APP_DIR / "Qwen_CoPilot.lnk"))
        shortcut.TargetPath = str(APP_DIR / "model" / "qwen.exe")
        shortcut.WorkingDirectory = str(APP_DIR)
        shortcut.Save()
    except Exception as e:
        print(f"Error creating shortcut: {e}")

def main():
    ensure_directory_exists(APP_DIR)

    # Copy Qwen Model
    copy_qwen_model()

    # Install Phylan
    install_phylan()

    # Create Taskbar Icon
    create_taskbar_icon()

    # Create Shortcut
    create_shortcut()

if __name__ == "__main__":
    main()
