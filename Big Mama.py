import psutil
import subprocess
import time
import socket
from datetime import datetime

# Define critical files and thresholds
critical_files = [
    'C:\\Windows\\System32\\cmd.exe',
    'C:\\Windows\\System32\\python.exe',
    # Add more Windows critical files here
]

ai_specific_names = ['python', 'java']
network_threshold = 10
file_threshold = 50
memory_threshold = 100 * 1024 * 1024

# Define essential processes for system lockdown
essential_processes = ['python', 'cmd', 'explorer']

def install_libraries():
    try:
        subprocess.run(['pip', 'install', 'psutil'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install psutil: {e}")

def monitor_system():
    def detect_anomalies(clf):
        current_data = get_current_data()
        prediction = clf.predict([current_data])
        if prediction[0] == -1:
            return True  # Anomaly detected
        else:
            return False

    return detect_anomalies

def get_current_data():
    data = {}
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'connections', 'memory_info']):
        if any(file in critical_files for file in proc.cmdline):
            continue  # Skip critical system files
        if proc.info['name'].lower() in ai_specific_names:
            data['ai_process'] = True
        if sum(1 for conn in proc.get('connections', []) if conn.status == psutil.CONN_ESTABLISHED and conn.type == psutil.SOCK_STREAM) >= network_threshold:
            data['network_connections'] = True
        if len(proc.cmdline) > file_threshold:
            data['file_access'] = True
        if proc.memory_info().rss > memory_threshold:
            data['memory_usage'] = True

    return data

def isolate_and_shutdown(rogue_pid):
    def isolate():
        print(f"Isolating process with PID: {rogue_pid}")
        psutil.Process(rogue_pid).suspend()

    def shutdown():
        print(f"Shutting down process with PID: {rogue_pid}")
        psutil.Process(rogue_pid).kill()

    return isolate, shutdown

def monitor_email_and_attachments():
    import smtplib
    from email.message import EmailMessage
    import os

    def scan_emails(email_folder):
        for msg in email_folder:
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == 'application/octet-stream':
                        file_name = part.get_filename()
                        if file_name and not file_name.endswith('.txt'):
                            save_path = os.path.join('attachments', file_name)
                            with open(save_path, 'wb') as f:
                                f.write(part.get_payload(decode=True))
                            scan_and_remove_viruses(save_path)

    def check_email_attachments(email_folder):
        for msg in email_folder:
            if any(keyword in msg['subject'].lower() for keyword in ['invoice', 'payment']):
                scan_emails(msg)

def monitor_network_ports():
    def scan_open_ports(ip_address='127.0.0.1'):
        open_ports = []
        for port in range(1, 65535):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((ip_address, port))
            if result == 0:
                open_ports.append(port)
            sock.close()
        return open_ports

    def check_for_backdoors(open_ports):
        suspicious_ports = [21, 22, 80, 443, 3389]  # Common backdoor ports
        for port in open_ports:
            if port in suspicious_ports:
                print(f"Suspicious port {port} is open. Checking for backdoors.")
                process = psutil.Process()
                if any(conn.laddr.port == port for conn in process.connections()):
                    isolate, shutdown = isolate_and_shutdown(process.pid)
                    isolate()
                    shutdown()

def monitor_system_activity():
    clf = train_model()  # Ensure you have a trained model
    detect_anomalies = monitor_system(clf)

    while True:
        if detect_anomalies():
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == 'rogue_ai_process_name':  # Replace with actual name of your AI process
                    rogue_pid = proc.info['pid']
                    isolate, shutdown = isolate_and_shutdown(rogue_pid)
                    isolate()
                    shutdown()

        open_ports = scan_open_ports()
        check_for_backdoors(open_ports)

        # Monitor email and attachments
        email_folder = fetch_emails()  # Implement your method to fetch emails
        check_email_attachments(email_folder)

        # Sleep for a while before the next check to reduce CPU usage
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    install_libraries()
    monitor_system_activity()
