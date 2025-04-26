import pika
import requests
import psutil
from sklearn.ensemble import IsolationForest
from datetime import datetime
import subprocess
import time
import smtplib
from email.mime.text import MIMEText
from threading import Thread
import logging
import json
from flask import Flask, jsonify

# Configuration
RABBITMQ_HOST = 'localhost'
RABBITMQ_PORT = 5672
RABBITMQ_USER = 'guest'
RABBITMQ_PASSWORD = 'guest'
AI_BOT_QUEUE = 'ai_bot_queue'
HEALTH_CHECK_INTERVAL = 30  # seconds
ALERT_EMAIL = 'admin@example.com'

# Initialize RabbitMQ connection
def init_rabbitmq():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=credentials))
    channel = connection.channel()
    channel.queue_declare(queue=AI_BOT_QUEUE)
    return channel, connection

# Send a message to the queue
def send_message(channel, message):
    channel.basic_publish(exchange='', routing_key=AI_BOT_QUEUE, body=message)

# Receive messages from the queue
def receive_messages(channel, callback):
    def on_message(ch, method, properties, body):
        Thread(target=callback, args=(ch, method, properties, body)).start()
    channel.basic_consume(queue=AI_BOT_QUEUE, on_message_callback=on_message)
    channel.start_consuming()

# Network Port Monitoring
def monitor_ports():
    while True:
        for conn in psutil.net_connections(kind='inet'):
            print(f"Port: {conn.laddr.port}, Status: {conn.status}")
        time.sleep(1)

# USB Device Detection and Monitoring
def detect_usb_devices():
    def on_device_add(action, device):
        if action == 'add':
            info = get_device_info(device)
            send_message(channel, json.dumps({'type': 'usb', 'action': 'add', 'info': info}))
    
    with open('/var/run/udevadm-monitor.pid', 'w') as f:
        subprocess.run(['udevadm', 'monitor', '--property'], stdout=f)

# Device Profiling
def get_device_info(device):
    info = {}
    try:
        output = subprocess.check_output(['udevadm', 'info', '--query=all', f'--name={device}']).decode()
        info['manufacturer'] = extract_field(output, 'MANUFACTURER')
        info['model'] = extract_field(output, 'MODEL')
        info['serial'] = extract_field(output, 'SERIAL')
    except subprocess.CalledProcessError:
        pass
    return info

def extract_field(output, field):
    for line in output.splitlines():
        if line.startswith(f'{field}'):
            return line.split('=')[1].strip()
    return None

# Anomaly Detection
def train_anomaly_detector(data):
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(data)
    return model

def detect_anomalies(model, new_data):
    anomalies = model.predict(new_data)
    return anomalies

# Real-Time Response
def send_alert(message):
    sender = 'alert@example.com'
    receiver = ALERT_EMAIL
    msg = MIMEText(message)
    msg['Subject'] = 'AI Bot Alert'
    msg['From'] = sender
    msg['To'] = receiver
    with smtplib.SMTP('smtp.example.com') as server:
        server.login(sender, 'password')
        server.sendmail(sender, [receiver], msg.as_string())

def block_ip(ip):
    subprocess.run(['sudo', 'iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP'])
    subprocess.run(['sudo', 'iptables', '-A', 'OUTPUT', '-d', ip, '-j', 'DROP'])

# Health Monitoring and Investigation
def health_check(channel, ai_bots):
    while True:
        for bot in ai_bots:
            try:
                response = requests.get(f'http://{bot}/health')
                if response.status_code != 200:
                    send_message(channel, json.dumps({'type': 'investigate', 'bot': bot}))
            except requests.exceptions.RequestException:
                send_message(channel, json.dumps({'type': 'investigate', 'bot': bot}))
        time.sleep(HEALTH_CHECK_INTERVAL)

def investigate_bot(bot):
    try:
        response = subprocess.run(['ping', '-c', '1', bot], capture_output=True)
        if response.returncode != 0:
            send_alert(f"AI Bot at {bot} is offline and not responding to pings.")
            return
        # Check if the AI bot service is running
        response = requests.get(f'http://{bot}/status')
        if response.status_code != 200 or 'running' not in response.text:
            send_alert(f"AI Bot at {bot} is offline, service not running.")
    except Exception as e:
        send_alert(f"AI Bot at {bot} encountered an error: {str(e)}")

# Main Function
def main():
    channel, connection = init_rabbitmq()

    def on_message(ch, method, properties, body):
        message = json.loads(body)
        if message['type'] == 'usb' and message['action'] == 'add':
            info = message['info']
            send_alert(f"USB device {info['manufacturer']} {info['model']} detected.")
        elif message['type'] == 'investigate':
            bot = message['bot']
            Thread(target=investigate_bot, args=(bot,)).start()

    # Start consuming messages
    receive_messages(channel, on_message)

    # Start monitoring ports and USB devices
    port_monitor_thread = Thread(target=monitor_ports)
    port_monitor_thread.start()

    usb_monitor_thread = Thread(target=detect_usb_devices)
    usb_monitor_thread.start()

    # Load AI bots list from configuration
    with open('ai_bots.json') as f:
        ai_bots = json.load(f)

    # Start health check
    health_check_thread = Thread(target=health_check, args=(channel, ai_bots))
    health_check_thread.start()

# Flask App for Health and Status Endpoints
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return "200 OK", 200

@app.route('/status', methods=['GET'])
def status_check():
    # Check if the service is running and other necessary checks
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    main()

# Ensure Flask app runs on each AI bot
app.run(host='0.0.0.0', port=5000)
