import os
import threading
import time
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from xgboost import XGBClassifier
import sqlite3
import zipfile

# Configuration variables
known_good_ips = set()
model_path = '/path/to/your/model.pkl'
label_encoder_path = '/path/to/your/label_encoder.pkl'
admin_email = 'admin@example.com'
admin_password = 'yourpassword'
smtp_server = 'smtp.example.com'
log_file = "/var/log/kern.log"
db_path = '/path/to/your/log.db'

# Install necessary dependencies
def install_dependencies():
    os.system('sudo apt-get update')
    os.system('sudo apt-get install -y python3-pip iptables')
    os.system('pip3 install pandas scikit-learn joblib xgboost')

# Configure iptables to log network traffic
def configure_iptables():
    os.system('sudo iptables -A INPUT -j LOG --log-prefix "INBOUND: "')
    os.system('sudo iptables -A OUTPUT -j LOG --log-prefix "OUTBOUND: "')
    os.system('sudo service rsyslog restart')

# Fetch known good IPs from an external API
def fetch_known_good_ips():
    url = 'https://api.example.com/known-good-ips'
    response = requests.get(url)
    if response.status_code == 200:
        return set(response.json().get('ips', []))
    else:
        print(f"Failed to fetch known good IPs: {response.content}")
        return set()

# Function to fetch and update known good IPs every hour
def fetch_ips_periodically():
    while True:
        global known_good_ips
        known_good_ips = fetch_known_good_ips()
        time.sleep(3600)  # Fetch every hour

# Train the machine learning model using XGBoost
def train_model():
    log_file = "/var/log/kern.log"
    with open(log_file, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        if "OUTBOUND" in line or "INBOUND" in line:
            parts = line.split()
            ip_src = parts[12]
            ip_dst = parts[14]
            port = int(parts[16])
            timestamp = datetime.strptime(parts[0], '%b %d %H:%M:%S')
            data.append((ip_src, ip_dst, port, timestamp.hour))

    df = pd.DataFrame(data, columns=['ip_src', 'ip_dst', 'port', 'hour'])

    # Encode IP addresses and ports
    label_encoder = LabelEncoder()
    df['ip_src'] = label_encoder.fit_transform(df['ip_src'])
    df['ip_dst'] = label_encoder.fit_transform(df['ip_dst'])

    # Train the model
    X = df[['ip_src', 'ip_dst', 'port', 'hour']]
    y = df['ip_src'].apply(lambda x: 1 if x in known_good_ips else 0)

    model = XGBClassifier(n_estimators=100)
    model.fit(X, y)

    # Save the model and label encoder
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, label_encoder_path)

# Load the trained model and label encoder
def load_model():
    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(label_encoder_path)
        return model, label_encoder
    except FileNotFoundError as e:
        print(f"Failed to load model: {e}")
        train_model()
        return load_model()

# Initialize the SQLite database for persistent storage of logs
def initialize_database():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table
    c.execute('''
        CREATE TABLE IF NOT EXISTS network_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_src TEXT,
            ip_dst TEXT,
            port INTEGER,
            timestamp DATETIME,
            is_known_good INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# Function to send an email with a system report or emergency notification
def send_email(subject, body, attachment_path=None):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = admin_email
    msg['To'] = admin_email

    # Attach the body of the email
    msg.attach(MIMEText(body))

    if attachment_path:
        with open(attachment_path, 'rb') as attachment:
            part = MIMEApplication(
                attachment.read(),
                Name=os.path.basename(attachment_path)
            )
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
            msg.attach(part)

    try:
        server = smtplib.SMTP(smtp_server, 587)
        server.starttls()
        server.login(admin_email, admin_password)
        server.sendmail(admin_email, [admin_email], msg.as_string())
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to monitor all programs and their network communications
def monitor_network():
    model, label_encoder = load_model()

    with open(log_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if "OUTBOUND" in line or "INBOUND" in line:
            parts = line.split()
            ip_src = parts[12]
            ip_dst = parts[14]
            port = int(parts[16])
            timestamp = datetime.strptime(parts[0], '%b %d %H:%M:%S')

            transformed_data = pd.DataFrame([(ip_src, ip_dst, port, timestamp.hour)], columns=['ip_src', 'ip_dst', 'port', 'hour'])
            transformed_data['ip_src'] = label_encoder.transform(transformed_data['ip_src'])
            transformed_data['ip_dst'] = label_encoder.transform(transformed_data['ip_dst'])

            prediction = model.predict(transformed_data[['ip_src', 'ip_dst', 'port', 'hour']])[0]

            if prediction == 0:
                send_alert(ip_src, ip_dst, port, timestamp)
                store_network_log(ip_src, ip_dst, port, timestamp)

# Function to send an alert email for abnormal communication
def send_alert(ip_src, ip_dst, port, timestamp):
    subject = "System Breach: Emergency Action Needed"
    body = f"""
    An abnormal communication has been detected:
    Source IP: {ip_src}
    Destination IP: {ip_dst}
    Port: {port}
    Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
    Immediate action is required to investigate and mitigate this threat.
    """

    send_email(subject, body)

# Function to store network logs in the database
def store_network_log(ip_src, ip_dst, port, timestamp):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''
        INSERT INTO network_logs (ip_src, ip_dst, port, timestamp, is_known_good)
        VALUES (?, ?, ?, ?, ?)
    ''', (ip_src, ip_dst, port, timestamp, 0))

    conn.commit()
    conn.close()

# Function to send a weekly report email with log files
def weekly_report():
    today = datetime.now().strftime("%Y-%m-%d")
    archive_filename = f"system_logs_{today}.log"

    # Compress the log file
    with zipfile.ZipFile(archive_filename, 'w') as zipf:
        zipf.write(log_file, os.path.basename(log_file))

    subject = "Weekly System Report"
    body = f"""
    Weekly System Report for {today}:

    Normal Communications and Abnormal Communications are detailed in the attached log file.
    """

    send_email(subject, body, attachment_path=archive_filename)

# Main function
def main():
    install_dependencies()
    configure_iptables()

    # Fetch and update known good IPs every hour
    ip_fetch_thread = threading.Thread(target=fetch_ips_periodically)
    ip_fetch_thread.daemon = True
    ip_fetch_thread.start()

    # Initialize the database for persistent storage of logs
    initialize_database()

    # Start the network monitoring background thread
    network_monitor_thread = threading.Thread(target=monitor_network)
    network_monitor_thread.daemon = True
    network_monitor_thread.start()

if __name__ == "__main__":
    main()
