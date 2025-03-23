import os
import subprocess
import psutil
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import notify2
import ctypes
import tkinter as tk
from tkinter import messagebox

# Ensure necessary libraries are installed
def install_libraries():
    required_libraries = ['psutil', 'pandas', 'sklearn', 'notify2']
    for library in required_libraries:
        subprocess.run(['pip', 'install', library])

# System Metrics Collection
def collect_metrics():
    metrics = {
        'cpu': psutil.cpu_percent(interval=1),
        'memory': psutil.virtual_memory().percent,
        'disk_io': psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes,
        'network': psutil.net_io_counters().bytes_recv + psutil.net_io_counters().bytes_sent
    }
    return metrics

# CPU Management
def set_cpu_governor(governor):
    if os.name == 'posix':
        subprocess.run(['sudo', 'echo', governor, '|', 'tee', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'])
    elif os.name == 'nt':
        powercfg = ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
    elif os.name == 'mac':
        subprocess.run(['sudo', 'pmset', '-a', 'powerstate', '3'])

# Memory Management
def compress_memory():
    if os.name == 'posix':
        subprocess.run(['echo', '1', '|', 'sudo', 'tee', '/proc/sys/vm/drop_caches'])
    elif os.name == 'nt':
        subprocess.run(['ipconfig', '/flushdns'])
    elif os.name == 'mac':
        subprocess.run(['purge'])

# Disk I/O Optimization
def set_io_scheduler(scheduler):
    if os.name == 'posix':
        subprocess.run(['sudo', 'echo', scheduler, '|', 'tee', '/sys/block/sda/queue/scheduler'])

# Network Optimization
def optimize_network():
    if os.name == 'nt':
        subprocess.run(['netsh', 'int', 'ipv4', 'set', 'interface', 'name="YourInterfaceName"', 'admin=enabled'])
    elif os.name == 'posix' or os.name == 'mac':
        subprocess.run(['sudo', 'sysctl', '-w', 'net.inet.tcp.mssdflt=1448'])

# Machine Learning Model
def collect_training_data(metrics, action):
    data = metrics.copy()
    data['action'] = action
    df = pd.DataFrame([data])
    df.to_csv('training_data.csv', mode='a', header=not os.path.exists('training_data.csv'), index=False)

def train_model():
    df = pd.read_csv('training_data.csv')
    X = df.drop(columns=['action'])
    y = df['action']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    return model

def predict_and_adjust(metrics):
    with open('performance_model.pkl', 'rb') as f:
        model = pickle.load(f)

    action = model.predict([metrics])[0]
    if action == 'set_performance_mode':
        set_cpu_governor('performance')
        compress_memory()
        set_io_scheduler('deadline')
        optimize_network()
    elif action == 'set_power_saving_mode':
        set_cpu_governor('powersave')
        compress_memory()
        set_io_scheduler('cfq')

def notify_user(message):
    notify2.init("Performance Optimizer")
    n = notify2.Notification("Optimization Action", "", message)
    n.show()

# GUI for Configuration
class OptimizerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Performance Optimizer")
        self.geometry("300x200")

        self.cpu_label = tk.Label(self, text="CPU Governor:")
        self.cpu_label.pack()

        self.cpu_var = tk.StringVar(value="performance")
        self.cpu_dropdown = tk.OptionMenu(self, self.cpu_var, "performance", "powersave")
        self.cpu_dropdown.pack()

        self.optimize_button = tk.Button(self, text="Optimize Now", command=self.optimize)
        self.optimize_button.pack()

    def optimize(self):
        action = self.cpu_var.get()
        if action == 'performance':
            set_cpu_governor('performance')
            compress_memory()
            set_io_scheduler('deadline')
            optimize_network()
        elif action == 'powersave':
            set_cpu_governor('powersave')
            compress_memory()
            set_io_scheduler('cfq')

if __name__ == "__main__":
    # Ensure necessary libraries are installed
    install_libraries()

    # Collect initial metrics for training data
    initial_metrics = collect_metrics()
    action = 'set_performance_mode'
    set_cpu_governor('performance')
    compress_memory()
    set_io_scheduler('deadline')
    optimize_network()
    collect_training_data(initial_metrics, action)

    # Train the model with collected data
    model = train_model()
    with open('performance_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Create a GUI for user interaction
    app = OptimizerGUI()
    app.mainloop()

    # Continuous monitoring and adaptive tuning
    while True:
        metrics = collect_metrics()
        predict_and_adjust(metrics)
        notify_user(f"Action Taken: {action}")
        time.sleep(10)  # Adjust the interval as needed
