import time
from scapy.all import sniff, IP, TCP, Raw
import socket
import struct
import ctypes

# Define the process name of the game you want to monitor
* = "game.exe"  # Replace with your game's executable name

# Function to get the PID of the game process
def get_game_pid():
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == GAME_PROCESS_NAME:
            return proc.info['pid']
    return None

# Function to create a fake IP address and port
def generate_fake_ip_and_port():
    fake_ip = ".".join(map(str, (random.randint(0, 255) for _ in range(4)))
    fake_port = random.randint(1024, 65535)
    return fake_ip, fake_port

# Function to intercept and modify outgoing packets
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

            # Generate fake IP and port
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

# Start sniffing for packets from the game process
def start_interception():
    game_pid = get_game_pid()
    if not game_pid:
        print("Game process not found.")
        return

    def filter_packets(packet):
        if IP in packet and TCP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport

            # Check if the packet is from the game process
            for conn in psutil.Process(game_pid).connections():
                if (conn.laddr.port == src_port and conn.raddr.port == dst_port) or \
                   (conn.laddr.port == dst_port and conn.raddr.port == src_port):
                    return True
        return False

    sniff(filter=filter_packets, prn=packet_callback)

if __name__ == "__main__":
    start_interception()
