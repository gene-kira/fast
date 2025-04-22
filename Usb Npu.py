import usb.core
import usb.util
import subprocess
import sys

# Function to check and install necessary libraries
def install_library(library):
    try:
        __import__(library)
    except ImportError:
        print(f"Installing {library}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# Ensure pyusb is installed
install_library('pyusb')

# Function to list all USB devices and identify the NPU module
def find_npu_module():
    npu_devices = []
    devices = usb.core.find(find_all=True)
    for device in devices:
        try:
            device_info = {
                'vendor_id': device.idVendor,
                'product_id': device.idProduct,
                'manufacturer': usb.util.get_string(device, device.iManufacturer),
                'product': usb.util.get_string(device, device.iProduct)
            }
            if "NPU" in device_info['manufacturer'] or "NPU" in device_info['product']:
                npu_devices.append(device_info)
        except Exception as e:
            print(f"Error accessing device: {e}")
    return npu_devices

# Function to install any NPU-specific libraries
def install_npu_libraries(npu_devices):
    for device in npu_devices:
        vendor_id = f"{device['vendor_id']:04x}"
        product_id = f"{device['product_id']:04x}"
        library_name = f"npu_{vendor_id}_{product_id}"
        try:
            __import__(library_name)
        except ImportError:
            print(f"Installing {library_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])

# Function to initialize communication with the NPU module
def init_npu_module(npu_device):
    vendor_id = npu_device['vendor_id']
    product_id = npu_device['product_id']
    device = usb.core.find(idVendor=vendor_id, idProduct=product_id)
    if device is None:
        raise ValueError("NPU Module not found")
    
    # Claim the device
    try:
        device.detach_kernel_driver(0)
    except Exception as e:
        pass  # Already detached or no kernel driver
    
    # Set the active configuration
    device.set_configuration()
    
    # Find the correct interface and endpoint
    for config in device:
        for interface in config:
            if interface.bInterfaceClass == 255:  # Assuming NPU uses class 255, adjust as needed
                device claiming interface
                usb.util.claim_interface(device, interface.bInterfaceNumber)
                return device, interface

# Main function to set up and communicate with the NPU module
def main():
    install_library('pyusb')
    
    # Find all connected USB devices and identify NPU modules
    npu_devices = find_npu_module()
    if not npu_devices:
        print("No NPU modules found.")
        return
    
    # Install any missing NPU-specific libraries
    install_npu_libraries(npu_devices)
    
    # Initialize communication with the first detected NPU module
    npu_device = npu_devices[0]
    device, interface = init_npu_module(npu_device)
    
    print("NPU Module initialized successfully.")
    
    # Send a test command to ensure communication is working
    endpoint_in = None
    endpoint_out = None
    
    for endpoint in interface:
        if usb.util.endpoint_direction(endpoint.bEndpointAddress) == usb.util.ENDPOINT_IN:
            endpoint_in = endpoint
        elif usb.util.endpoint_direction(endpoint.bEndpointAddress) == usb.util.ENDPOINT_OUT:
            endpoint_out = endpoint
    
    if not endpoint_in or not endpoint_out:
        raise ValueError("NPU Module does not have the required endpoints")
    
    # Send a test command to the NPU module
    test_command = b'\x01'  # Example test command, adjust as needed
    device.write(endpoint_out.bEndpointAddress, test_command)
    response = device.read(endpoint_in.bEndpointAddress, len(test_command))
    
    if response == test_command:
        print("Test command sent and received successfully.")
    else:
        print("Failed to receive expected response.")

if __name__ == "__main__":
    main()
