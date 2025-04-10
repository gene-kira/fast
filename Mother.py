import os
import json
import re
import requests
from bs4 import BeautifulSoup
import hashlib
import psutil
import subprocess
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from keras.models import load_model, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
from tensorflow_model_optimization.quantization.keras import quantize_model
import joblib
from kafka import KafkaConsumer, KafkaProducer
import schedule
import threading
import cProfile
import autopep8
import ast
from pylint import epylint

# Constants and Paths
CONFIG_FILE = 'config.json'
AD_SERVERS_FILE = 'ad_servers.txt'
THREAT_INTELLIGENCE_FILE = 'threat_intelligence.csv'
KNOWN_SAFE_HASHES_FILE = 'known_safe_hashes.txt'
MODEL_PATH = 'optimized_model.h5'
FEEDBACK_TOPIC = 'user_feedback'
CLEAN_DATA_TOPIC = 'clean_data'

# Load configuration
def load_configuration():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

# Setup Kafka Consumers and Producers
def setup_kafka_consumer(topic, group_id):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id=group_id
    )
    return consumer

def setup_kafka_producer():
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
    return producer

# Load the trained model
def load_model():
    try:
        from keras.models import load_model
        return load_model(MODEL_PATH)
    except Exception as e:
        logging.error(f"Error loading machine learning model: {e}")

# Preprocess data
def preprocess_data(message):
    data = json.loads(message.value.decode('utf-8'))
    data['cleaned_content'] = clean_data(data['content'])
    return data

def clean_data(content):
    # Remove HTML tags and URLs
    cleaned_content = re.sub(r'<.*?>', '', content)
    cleaned_content = re.sub(r'http\S+', '', cleaned_content)
    
    # Remove special characters and numbers
    cleaned_content = re.sub(r'[^a-zA-Z\s]', '', cleaned_content)
    
    # Convert to lowercase
    cleaned_content = cleaned_content.lower()
    
    # Tokenize and remove stop words
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(cleaned_content)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)

# Train the machine learning model using threat intelligence data
def train_model(threat_intelligence_file):
    # Collect behavioral data
    behavioral_data = collect_behavioral_data()

    # Convert the collected data into a format suitable for machine learning
    X = []
    y = []

    for data in behavioral_data:
        features = [
            int('python' in data['name'] or 'java' in data['name']),
            int(data['cpu_usage'] > 50),
            int(data['memory_usage'] > 50),
            int(len(data['open_ports']) > 10)
        ]
        X.append(features)
        y.append(int(data['is_malicious']))

    X = np.array(X)
    y = np.array(y)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

    # Train the model
    best_model = IsolationForest(contamination=0.05, n_estimators=100, max_samples='auto', max_features=1.0)
    best_model.fit(X_pca)

    return best_model, X_pca, pca

# Scrape threat intelligence from multiple sources
def scrape_threat_intelligence(initial_urls, interval):
    def fetch_intelligence(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                threats = [threat.strip() for threat in soup.find_all('p')]
                return threats
            else:
                logging.warning(f"Failed to fetch intelligence from {url}: Status code {response.status_code}")
        except Exception as e:
            logging.error(f"Error fetching intelligence from {url}: {e}")

    def update_threat_intelligence(threats, file):
        with open(file, 'a') as f:
            for threat in threats:
                f.write(f"{threat}\n")

    for url in initial_urls:
        threats = fetch_intelligence(url)
        if threats:
            update_threat_intelligence(threats, THREAT_INTELLIGENCE_FILE)

    schedule.every(interval).hours.do(scrape_threat_intelligence, initial_urls=initial_urls, interval=interval)

# Load ad servers
def load_ad_servers():
    ad_servers = []
    with open(AD_SERVERS_FILE, 'r') as file:
        for line in file:
            ad_servers.append(line.strip())
    return ad_servers

# Main function to initialize and start monitoring
def main():
    config = load_configuration()
    os = get_os()
    commands = get_os_specific_commands(os)

    # Define paths for threat intelligence and known safe hashes
    threat_intelligence_file = config['threat_intelligence_file']
    known_safe_hashes_file = config['known_safe_hashes_file']

    # Load the machine learning model
    global model
    try:
        model = load_model()
    except Exception as e:
        logging.error(f"Error loading machine learning model: {e}")

    # Scrape threat intelligence from multiple sources
    initial_urls = ['https://example.com/threats', 'https://threatintelligence.net']
    scrape_threat_intelligence(initial_urls, config['scrape_interval'])

    # Train the machine learning model using threat intelligence data
    train_model(threat_intelligence_file)

    # Update the list of known safe hashes
    update_known_safe_hashes()

    # Start monitoring processes and files
    monitor_processes(config['monitor_interval'])

# Load OS-specific commands
def get_os():
    return platform.system()

def get_os_specific_commands(os):
    if os == 'Windows':
        return {
            'psutil': psutil,
            'requests': requests,
            'BeautifulSoup': BeautifulSoup,
            'hashlib': hashlib
        }
    elif os in ['Linux', 'Darwin']:
        return {
            'psutil': psutil,
            'requests': requests,
            'BeautifulSoup': BeautifulSoup,
            'hashlib': hashlib
        }

# Train the model using threat intelligence data
def train_model(threat_intelligence_file):
    # Collect behavioral data
    behavioral_data = collect_behavioral_data()

    # Convert the collected data into a format suitable for machine learning
    X = []
    y = []

    for data in behavioral_data:
        features = [
            int('python' in data['name'] or 'java' in data['name']),
            int(data['cpu_usage'] > 50),
            int(data['memory_usage'] > 50),
            int(len(data['open_ports']) > 10)
        ]
        X.append(features)
        y.append(int(data['is_malicious']))

    X = np.array(X)
    y = np.array(y)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

    # Train the model
    best_model = IsolationForest(contamination=0.05, n_estimators=100, max_samples='auto', max_features=1.0)
    best_model.fit(X_pca)

    return best_model, X_pca, pca

# Collect behavioral data from system processes
def collect_behavioral_data():
    processes = []
    for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
        try:
            ports = [conn.laddr.port for conn in proc.connections() if conn.status == 'LISTEN']
            processes.append({
                'pid': proc.pid,
                'name': proc.name(),
                'cpu_usage': proc.cpu_percent(),
                'memory_usage': proc.memory_percent(),
                'open_ports': ports,
                'is_malicious': 0  # Initial assumption
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return processes

# Monitor system for anomalies using the trained model
def monitor_processes(interval):
    ad_servers = load_ad_servers()
    
    def detect_and_terminate_anomalies(open_ports, model, unique_ports, pca):
        X = [1 if port in open_ports else 0 for port in unique_ports]
        X = np.array([X])
        X_pca = pca.transform(StandardScaler().fit_transform(X))
        
        anomaly_score = model.decision_function(X_pca)
        if anomaly_score < -0.5:  # Threshold for anomaly detection
            print("Anomaly detected. Terminating...")
            terminate_anomalous_ports(open_ports)

    def terminate_anomalous_ports(anomalous_ports):
        for port in anomalous_ports:
            try:
                subprocess.run(['iptables', '-A', 'OUTPUT', '-p', 'tcp', '--dport', str(port), '-j', 'DROP'])
                print(f"Terminated anomalous port: {port}")
            except Exception as e:
                print(f"Failed to terminate port {port}: {e}")

    # Schedule the monitoring task
    schedule.every(interval).seconds.do(monitor_system)

def monitor_system():
    processes = collect_behavioral_data()
    
    open_ports = []
    for proc in processes:
        if len(proc['open_ports']) > 0:
            open_ports.extend(proc['open_ports'])

    unique_ports = list(set(open_ports))
    
    # Load the trained model and PCA
    model, X_pca, pca = train_model(THREAT_INTELLIGENCE_FILE)
    
    # Detect anomalies in real-time
    detect_and_terminate_anomalies(open_ports, model, unique_ports, pca)

# Initialize Kafka consumers and producers
def initialize_kafka():
    global consumer_feedback, producer_clean_data

    consumer_feedback = setup_kafka_consumer(FEEDBACK_TOPIC, 'feedback_group')
    producer_clean_data = setup_kafka_producer()

# Process feedback for continuous improvement
def process_feedback():
    for message in consumer_feedback:
        data = preprocess_data(message)
        if data['cleaned_content'] not in known_safe_hashes:
            ad_servers.append(data['cleaned_content'])
            with open(AD_SERVERS_FILE, 'a') as file:
                file.write(f"{data['cleaned_content']}\n")

# Main function to initialize and start monitoring
def main():
    config = load_configuration()
    os = get_os()
    commands = get_os_specific_commands(os)

    # Initialize Kafka consumers and producers
    initialize_kafka()

    # Load the machine learning model
    global model
    try:
        model = load_model()
    except Exception as e:
        logging.error(f"Error loading machine learning model: {e}")

    # Scrape threat intelligence from multiple sources
    initial_urls = ['https://example.com/threats', 'https://threatintelligence.net']
    scrape_threat_intelligence(initial_urls, config['scrape_interval'])

    # Train the machine learning model using threat intelligence data
    train_model(THREAT_INTELLIGENCE_FILE)

    # Update the list of known safe hashes
    update_known_safe_hashes()

    # Start monitoring processes and files
    monitor_processes(config['monitor_interval'])

# Run the main function to start the monitoring process
if __name__ == "__main__":
    main()
