import os
import sys
import time
import subprocess
from psutil import process_iter, wait_procs

def check_for_rogue_programs():
    for proc in process_iter(['pid', 'name']):
        if proc.info['name'] not in ['your_os_process', 'local_network_process']:
            print(f"Rogue program detected: {proc.info}")
            terminate_process(proc)

def terminate_process(process):
    try:
        process.terminate()
        process.wait(timeout=3)
    except (psutil.TimeoutExpired, psutil.AccessDenied):
        process.kill()

def monitor_disk_access():
    with open('/proc/mounts', 'r') as f:
        mounts = f.readlines()
    for mount in mounts:
        if 'rw' in mount and not mount.startswith('/dev/'):
            print(f"Unauthorized disk access detected: {mount}")
            unmount_disk(mount)

def unmount_disk(mount):
    device = mount.split()[0]
    subprocess.run(['sudo', 'umount', device])

def monitor_system_behavior():
    while True:
        check_for_rogue_programs()
        monitor_disk_access()
        time.sleep(5)  # Check every 5 seconds

if __name__ == "__main__":
    monitor_system_behavior()
