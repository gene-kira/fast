import os
import time
from scapy.all import sniff, IP, TCP, Raw, send
import psutil
import socket
import struct
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import pickle

# Define the process name of the game you want to monitor
GAME_PROCESS_NAME = "game.exe"  # Replace with your game's executable name

def install_libraries():
    required_libraries = [
        'scapy',
        'psutil',
        'pandas',
        'numpy',
        'sklearn'
    ]
    for lib in required_libraries:
        os.system(f'pip install {lib}')

def get_game_pid():
    game_pids = []
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == GAME_PROCESS_NAME:
            game_pids.append(proc.info['pid'])
    return game_pids

def train_game_detection_model():
    # Collect data on running processes
    process_data = []
    for proc in psutil.process_iter(['pid', 'name']):
        process_data.append({
            'pid': proc.info['pid'],
            'name': proc.info['name']
        })
    
    df_processes = pd.DataFrame(process_data)

    # Train a model to detect the game process
    X = df_processes[['pid']]
    y = (df_processes['name'] == GAME_PROCESS_NAME).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100)
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    with open('game_detection_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def ip_to_int(ip):
    return struct.unpack("!I", socket.inet_aton(ip))[0]

def collect_packets(filename="training_data.csv"):
    game_pids = get_game_pid()
    if not game_pids:
        print("Game process not found.")
        return

    def packet_filter(packet):
        if IP in packet and TCP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport

            # Check if the packet is from any of the game processes
            for pid in game_pids:
                for conn in psutil.Process(pid).connections():
                    if (conn.laddr.port == src_port and conn.raddr.port == dst_port) or \
                       (conn.laddr.port == dst_port and conn.raddr.port == src_port):
                        return True
        return False

    def packet_callback(packet):
        data = {
            'src_ip': packet[IP].src,
            'dst_ip': packet[IP].dst,
            'src_port': packet[TCP].sport,
            'dst_port': packet[TCP].dport,
            'length': len(packet)
        }
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=data.keys())
        df = df.append(data, ignore_index=True)
        df.to_csv(filename, index=False)

    sniff(filter=packet_filter, prn=packet_callback, count=10000)

def train_model():
    df = pd.read_csv("training_data.csv")

    # Feature and target variables
    X = df[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'length']]
    y = df['src_ip']  # You can choose any column as the target for training, e.g., src_ip

    # Convert IP addresses to numerical values
    X['src_ip'] = X['src_ip'].apply(ip_to_int)
    X['dst_ip'] = X['dst_ip'].apply(ip_to_int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use a pipeline to scale features and train the model
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100)
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save the trained model
    with open('packet_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def generate_fake_ip_and_port():
    fake_ip = ".".join(map(str, (random.randint(0, 255) for _ in range(4))))
    fake_port = random.randint(1024, 65535)
    return fake_ip, fake_port

def packet_callback(packet):
    if IP in packet:
        ip_layer = packet[IP]
        if TCP in packet:
            tcp_layer = packet[TCP]

            # Check if the packet is from the game process
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport

            # Convert IP addresses to numerical values
            src_ip_int = struct.unpack("!I", socket.inet_aton(src_ip))[0]
            dst_ip_int = struct.unpack("!I", socket.inet_aton(dst_ip))[0]

            # Extract features from the packet
            data = np.array([src_ip_int, dst_ip_int, src_port, dst_port, len(packet)]).reshape(1, -1)

            # Load the trained model
            with open('packet_model.pkl', 'rb') as f:
                model = pickle.load(f)

            # Predict if the packet is from the game process
            is_game_packet = model.predict(data)[0]
            if is_game_packet:
                fake_src_ip, fake_src_port = generate_fake_ip_and_port()
                fake_dst_ip, fake_dst_port = generate_fake_ip_and_port()

                # Modify the packet with fake information
                ip_layer.src = fake_src_ip
                ip_layer.dst = fake_dst_ip
                tcp_layer.sport = fake_src_port
                tcp_layer.dport = fake_dst_port

                # Rebuild the packet
                del packet[IP].chksum
                del packet[TCP].chksum
                new_packet = IP(bytes(packet))

                # Send the modified packet
                send(new_packet)

def start_interception():
    game_pids = get_game_pid()
    if not game_pids:
        print("Game process not found.")
        return

    def filter_packets(packet):
        if IP in packet and TCP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport

            # Check if the packet is from any of the game processes
            for pid in game_pids:
                for conn in psutil.Process(pid).connections():
                    if (conn.laddr.port == src_port and conn.raddr.port == dst_port) or \
                       (conn.laddr.port == dst_port and conn.raddr.port == src_port):
                        return True
        return False

    sniff(filter=filter_packets, prn=packet_callback)

if __name__ == "__main__":
    install_libraries()
    
    # Train the game detection model
    train_game_detection_model()

    # Collect training data
    collect_packets("training_data.csv")
    
    # Train the packet modification model
    train_model()
    
    # Start real-time interception and packet modification
    start_interception()
