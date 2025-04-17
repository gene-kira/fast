import subprocess
import sys
import socket
import psutil
import os
import hashlib
from ctypes import wintypes, c_ulonglong, POINTER, byref, create_string_buffer, Structure, c_ushort, c_ulong

# Define necessary structures for PEB monitoring
class STARTUPINFO(Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", c_ushort * 16),
        ("lpReserved2", POINTER(wintypes.BYTE))
    ]

class PROCESS_INFORMATION(Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD)
    ]

kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

def install_libraries():
    required_libraries = ['psutil', 'hashlib']
    for library in required_libraries:
        try:
            __import__(library)
        except ImportError:
            print(f"Installing {library}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])

def detect_reverse_shell():
    connections = psutil.net_connections(kind='tcp')
    
    for conn in connections:
        if conn.status == 'ESTABLISHED':
            local_ip, local_port = conn.laddr
            remote_ip, remote_port = conn.raddr
            
            # Check for known reverse shell ports (e.g., 443, 80, etc.)
            suspicious_ports = [443, 80]
            if remote_port in suspicious_ports:
                print(f"Suspicious connection detected: {local_ip}:{local_port} -> {remote_ip}:{remote_port}")
                
                # Terminate the process
                proc = psutil.Process(conn.pid)
                try:
                    proc.terminate()
                    proc.wait()  # Wait for the process to terminate
                    print(f"Process with PID {conn.pid} terminated.")
                except psutil.NoSuchProcess:
                    print(f"Process with PID {conn.pid} does not exist.")

def get_peb_address(pid):
    h_process = kernel32.OpenProcess(0x1F0FFF, False, pid)  # PROCESS_ALL_ACCESS
    if not h_process:
        return None
    
    module_info = create_string_buffer(sizeof(wintypes.MODULEINFO))
    if psapi.GetModuleInformation(h_process, h_process, module_info, sizeof(module_info)):
        peb_address = (c_ulonglong * 1)()
        kernel32.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, wintypes.DWORD, POINTER(c_ulonglong), wintypes.ULONG, POINTER(wintypes.ULONG)]
        status = kernel32.NtQueryInformationProcess(h_process, 0xC, peb_address, sizeof(peb_address), None)
        if status == 0:
            return peb_address[0]
    return None

def monitor_peb():
    for proc in psutil.process_iter(['pid', 'name']):
        pid = proc.info['pid']
        print(f"Monitoring PEB of process: {proc.info['name']} (PID: {pid})")
        
        peb_address = get_peb_address(pid)
        if not peb_address:
            continue
        
        # Read the PEB structure
        class _PEB(Structure):
            _fields_ = [
                ("InheritedAddressSpace", wintypes.BOOLEAN),
                ("ReadImageFileExecOptions", wintypes.BOOLEAN),
                ("BeingDebugged", wintypes.BOOLEAN),
                ("BitField", c_ushort),
                ("Mutant", c_ulonglong),
                ("ImageBaseAddress", c_ulonglong),
                ("Ldr", POINTER(wintypes.LIST_ENTRY)),
                ("ProcessParameters", POINTER(wintypes.RTL_USER_PROCESS_PARAMETERS)),
                ("SubSystemData", c_ulonglong),
                ("ProcessHeap", c_ulonglong),
                ("FastPebLock", wintypes.LONG),
                ("AtlThunkSListPtr", c_ulonglong),
                ("IFEOKey", c_ulonglong)
            ]
        
        peb = _PEB()
        if kernel32.ReadProcessMemory(h_process, peb_address, byref(peb), sizeof(_PEB), None):
            # Check for suspicious PEB fields
            if peb.BeingDebugged:
                print(f"Suspicious PEB detected in process: {proc.info['name']} (PID: {pid})")
                
                # Terminate the process
                proc = psutil.Process(pid)
                try:
                    proc.terminate()
                    proc.wait()  # Wait for the process to terminate
                    print(f"Process with PID {pid} terminated.")
                except psutil.NoSuchProcess:
                    print(f"Process with PID {pid} does not exist.")

def get_module_hash(module_path):
    with open(module_path, 'rb') as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()

def inspect_kernel_modules():
    # Known malicious module hashes
    known_malicious_hashes = {
        "0123456789abcdef0123456789abcdef": "malware_module.dll",
        # Add more known malicious hashes here
    }
    
    for proc in psutil.process_iter(['pid']):
        try:
            modules = proc.memory_maps()
            for module in modules:
                if not os.path.exists(module.path):
                    continue
                
                module_hash = get_module_hash(module.path)
                if module_hash in known_malicious_hashes:
                    print(f"Malicious kernel module detected: {module.path} (PID: {proc.info['pid']})")
                    
                    # Terminate the process
                    proc = psutil.Process(proc.info['pid'])
                    try:
                        proc.terminate()
                        proc.wait()  # Wait for the process to terminate
                        print(f"Process with PID {proc.info['pid']} terminated.")
                    except psutil.NoSuchProcess:
                        print(f"Process with PID {proc.info['pid']} does not exist.")
        except (psutil.AccessDenied, psutil.ZombieProcess):
            continue

def main():
    install_libraries()
    
    # Main functions
    detect_reverse_shell()
    monitor_peb()
    inspect_kernel_modules()

if __name__ == "__main__":
    main()
