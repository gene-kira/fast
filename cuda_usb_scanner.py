import usb.core
import subprocess
import os
import pynvml

def list_usb_devices():
    devices = usb.core.find(find_all=True)
    for device in devices:
        print(f"Device: {device.product} (Vendor ID: {device.idVendor}, Product ID: {device.idProduct})")

def detect_cuda_gpus():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        cuda_gpus = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            cuda_gpus.append((i, name.decode(), uuid))
        
        return cuda_gpus
    except pynvml.NVMLError as e:
        print(f"Failed to initialize NVIDIA Management Library: {e}")
        return []

def check_cuda_on_usb(cuda_gpus):
    usb_devices = list_usb_devices()
    cuda_on_usb = []
    
    for device in usb_devices:
        for gpu in cuda_gpus:
            if str(device.idVendor) in gpu[2] and str(device.idProduct) in gpu[2]:
                cuda_on_usb.append((device, gpu))
    
    return cuda_on_usb

def use_cuda_cores(cuda_on_usb):
    import torch
    device_count = torch.cuda.device_count()
    
    for i, (usb_device, gpu) in enumerate(cuda_on_usb):
        if i < device_count:
            device = f"cuda:{i}"
            print(f"Using CUDA core {gpu[0]}: {gpu[1]} on device {device}")
            
            # Example computation: Matrix multiplication
            a = torch.rand((1000, 1000), device=device)
            b = torch.rand((1000, 1000), device=device)
            c = torch.matmul(a, b)
            print(f"Matrix multiplication result on {device} (first element): {c[0][0]}")
        else:
            print(f"CUDA core {gpu[0]}: {gpu[1]} not available for computation.")

if __name__ == "__main__":
    # List all USB devices
    list_usb_devices()
    
    # Detect CUDA-capable GPUs
    cuda_gpus = detect_cuda_gpus()
    if not cuda_gpus:
        print("No CUDA-capable GPUs detected.")
    else:
        print(f"Detected {len(cuda_gpus)} CUDA-capable GPUs:")
        for gpu in cuda_gpus:
            print(f"GPU {gpu[0]}: {gpu[1]} (UUID: {gpu[2]})")
    
    # Check if any of the detected CUDA GPUs are connected via USB
    cuda_on_usb = check_cuda_on_usb(cuda_gpus)
    if not cuda_on_usb:
        print("No CUDA-capable GPUs found on USB devices.")
    else:
        print(f"Found {len(cuda_on_usb)} CUDA-capable GPUs on USB devices:")
        for usb_device, gpu in cuda_on_usb:
            print(f"Device: {usb_device.product} (Vendor ID: {usb_device.idVendor}, Product ID: {usb_device.idProduct}) - GPU {gpu[0]}: {gpu[1]}")
    
    # Use the detected CUDA cores for computation
    use_cuda_cores(cuda_on_usb)
