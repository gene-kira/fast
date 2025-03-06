import subprocess
import sys
import importlib.util

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        sys.exit(1)

def check_and_install_packages(packages):
    for package in packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            print(f"{package} not found. Installing...")
            install_package(package)

# List of required packages
required_packages = [
    'torch',
    'torchvision',
    'opencv-python',
    'numpy',
    'pygame',
    'scipy'
]

# Check and install missing packages
check_and_install_packages(required_packages)
