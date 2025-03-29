# Auto-Loader for Necessary Libraries
import os
import subprocess

# Install necessary libraries
def install_libraries():
    required_libraries = [
        'scapy',
        'pandas',
        'numpy',
        'sklearn'
    ]
    
    for library in required_libraries:
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# Import necessary libraries
import os
from scapy.all import sniff, IP, TCP, UDP
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Set Up Network Monitoring
def monitor_network(interface='eth0'):
    def packet_callback(packet):
        if packet.haslayer(IP):
            ip = packet[IP]
            protocol = ip.proto
            src_ip = ip.src
            dst_ip = ip.dst
            src_port = None
            dst_port = None
            
            if packet.haslayer(TCP) or packet.haslayer(UDP):
                src_port = packet.sport
                dst_port = packet.dport
            
            # Log the packet details
            log_packet(src_ip, dst_ip, src_port, dst_port, protocol)
    
    # Start sniffing packets on the specified interface
    sniff(iface=interface, prn=packet_callback)

# Log Packet Details to a CSV for Analysis
def log_packet(src_ip, dst_ip, src_port, dst_port, protocol):
    data = {
        'src_ip': [src_ip],
        'dst_ip': [dst_ip],
        'src_port': [src_port],
        'dst_port': [dst_port],
        'protocol': [protocol]
    }
    
    df = pd.DataFrame(data)
    if not os.path.exists('network_log.csv'):
        df.to_csv('network_log.csv', index=False, mode='w')
    else:
        df.to_csv('network_log.csv', index=False, mode='a', header=False)

# Load and Preprocess the Network Log
def load_and_preprocess_data():
    # Load the network log data
    data = pd.read_csv('network_log.csv')
    
    # Convert protocol to categorical values
    data['protocol'] = data['protocol'].astype('category').cat.codes
    
    # Fill NaN values for ports
    data['src_port'] = data['src_port'].fillna(0)
    data['dst_port'] = data['dst_port'].fillna(0)
    
    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']])
    
    return scaled_data

# Train a Machine Learning Model to Identify Threats
def train_threat_classifier(scaled_data, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, labels, test_size=0.2, random_state=42)
    
    # Initialize the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    return clf

# Evaluate the Model and Block Threats
def evaluate_and_block_threats(clf, scaled_data, labels):
    # Predict threats using the trained model
    predictions = clf.predict(scaled_data)
    
    # Identify malicious IP addresses and ports
    malicious_ips = set()
    for i in range(len(predictions)):
        if predictions[i] == 1:  # Assuming 1 indicates a threat
            malicious_ips.add((labels['src_ip'][i], labels['dst_ip'][i], labels['src_port'][i], labels['dst_port'][i]))
    
    # Block the identified threats
    for src_ip, dst_ip, src_port, dst_port in malicious_ips:
        block_ip(src_ip)
        block_ip(dst_ip)
        if src_port != 0:
            block_port(src_ip, src_port)
        if dst_port != 0:
            block_port(dst_ip, dst_port)

def block_ip(ip):
    os.system(f'sudo iptables -A INPUT -s {ip} -j DROP')

def block_port(ip, port):
    os.system(f'sudo iptables -A INPUT -p tcp --dport {port} -j DROP')
    os.system(f'sudo iptables -A INPUT -p udp --dport {port} -j DROP')

# Main Function
def main(interface='eth0'):
    install_libraries()
    
    # Set up network monitoring
    monitor_network(interface)
    
    # Load and preprocess the network log
    scaled_data = load_and_preprocess_data()
    
    # Train the threat classifier
    labels = pd.read_csv('network_log.csv')  # Assuming the labels are in the same CSV file
    clf = train_threat_classifier(scaled_data, labels['label'])
    
    # Evaluate and block threats
    evaluate_and_block_threats(clf, scaled_data, labels)

if __name__ == "__main__":
    main()
