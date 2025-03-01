import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import requests
import time
import concurrent.futures
import logging
from prometheus_client import Gauge, start_http_server
from dotenv import load_dotenv
from flask import Flask, render_template

# Load environment variables from .env file
load_dotenv()

EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER'),
    'smtp_port': int(os.getenv('SMTP_PORT')),
    'from_addr': os.getenv('FROM_EMAIL'),
    'password': os.getenv('EMAIL_PASSWORD'),
    'to_addr': os.getenv('TO_EMAIL')
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProcessAnomalyDetector:
    def __init__(self):
        self.model = None  # The anomaly detection model
        self.process_stats = []  # Store the latest process statistics
    
    def simulate_process_stats(self, num_processes=10, time_step=5):
        """Simulate process statistics for training or testing."""
        stats = []
        current_time = pd.Timestamp.now()
        
        for i in range(num_processes):
            start_time = current_time + timedelta(seconds=i * time_step)
            end_time = start_time + timedelta(seconds=time_step)
            
            cpu_usage = np.random.uniform(0.1, 1.0)  # Random CPU usage between 10% and 100%
            mem_usage_mb = (np.random.randint(256, 4096)) / 1024  # Memory in MB
            process_name = f"Process_{i}"
            user = "user"
            
            stats.append({
                'process_id': f"Process_{i}",
                'start_time': start_time,
                'end_time': end_time,
                'cpu_usage': cpu_usage,
                'memory_usage_mb': mem_usage_mb,
                'process_name': process_name,
                'user': user
            })
            
        return stats
    
    def fetch_process_stats_from_api(self, api_url='https://example.com/process-stats', max_retries=3):
        """Fetch process statistics from an API with retry logic."""
        for attempt in range(max_retries):
            try:
                response = requests.get(api_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    stats = []
                    for entry in data.get('process_stats', []):
                        process_id = entry.get('id', 'unknown')
                        start_time = pd.Timestamp(entry.get('start', datetime.now().isoformat()))
                        end_time = pd.Timestamp(entry.get('end', datetime.now().isoformat()))
                        
                        cpu_usage = float(entry.get('cpu_usage', 0.0))
                        mem_usage_mb = (float(entry.get('memory_usage_kb', 256)) / 1024)
                        process_name = entry.get('process_name', 'unknown')
                        user = entry.get('user', 'unknown')
                        
                        stats.append({
                            'process_id': f"Process_{process_id}",
                            'start_time': start_time,
                            'end_time': end_time,
                            'cpu_usage': cpu_usage,
                            'memory_usage_mb': mem_usage_mb,
                            'process_name': process_name,
                            'user': user
                        })
                    self.process_stats = stats
                    return stats
                else:
                    logging.error(f"API request failed with status code {response.status_code}")
                    if attempt == max_retries - 1:
                        raise Exception("Maximum retries exceeded.")
                    time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                logging.error(f"API request failed (Attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
        return []
    
    def extract_features(self, process_stats):
        """Extract features for anomaly detection."""
        try:
            features = []
            for entry in process_stats:
                if isinstance(entry, dict) and 'start_time' in entry and 'end_time' in entry:
                    start_time = pd.to_datetime(entry['start_time'])
                    end_time = pd.to_datetime(entry['end_time'])
                    duration = (end_time - start_time).total_seconds()
                    cpu_usage = entry.get('cpu_usage', 0.0)
                    mem_usage_mb = entry.get('memory_usage_mb', 0.0)
                    
                    features.append([duration, cpu_usage, mem_usage_mb])
            return np.array(features)
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            raise

    def train_model(self, process_stats):
        """Train an IsolationForest model to detect anomalies."""
        self.model = IsolationForest(random_state=42, contamination=0.1)
        
        features = self.extract_features(process_stats)
        if len(features) > 0:
            try:
                # Standardize the features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                # Train the model
                self.model.fit(features_scaled)
                logging.info("Model training completed.")
            except Exception as e:
                logging.error(f"Error training model: {e}")
                raise
        else:
            logging.warning("No valid data to train the model.")
    
    def detect_anomalies(self, process_stats=None):
        """Detect anomalies in process statistics."""
        if not self.model:
            raise ValueError("Model has not been trained yet.")
            
        if process_stats is None:
            process_stats = self.process_stats
            
        features = self.extract_features(process_stats)
        
        try:
            # Standardize the features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Predict anomalies
            labels = self.model.predict(features_scaled)
            return labels  # -1 for anomaly, 1 for normal
        except Exception as e:
            logging.error(f"Error detecting anomalies: {e}")
            raise
# Define the directory path (Q: or D: drive)
        stats_dir = os.path.join('C:', 'data-csv')  # Change to 'D:' if you prefer

    def log_results(self, process_stats, labels=None):
        """Log the detected anomalies in CSV files."""
        stats_df = pd.DataFrame(process_stats)
        
        if labels is not None:
            stats_df['is_anomaly'] = [1 if label == -1 else 0 for label in labels]
            
        # Create directory if it doesn't exist
        stats_dir = 'stats'
        os.makedirs(stats_dir, exist_ok=True)
        
        # Save the logs
        current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"process_stats_{current_time}"
        
        stats_file = os.path.join(stats_dir, f"{filename_base}.csv")
        backup_file = os.path.join(stats_dir, f"{filename_base}_backup.csv")
        
        # Try to save the file, handling potential exceptions
        try:
            original_filename = stats_df.to_csv(stats_file, index=False)
            
            # Create a backup with timestamp in filename
            backup_filename = stats_file.replace('.csv', f'_{current_time}.csv')
            stats_df.to_csv(backup_filename, index=False)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save statistics file: {e}")
            if os.path.exists(stats_file):
                # Try to recover from backup
                try:
                    os.replace(backup_file, stats_file)
                    logging.info("Recovered from backup file.")
                except Exception as e_recover:
                    logging.error(f"Backup recovery failed: {e_recover}")
            return False

    def start_monitoring(self):
        """Start monitoring with Prometheus."""
        gauge = Gauge('process_anomaly_detector', 'Anomalies detected in process statistics')
        
        def update_metrics():
            while True:
                if self.process_stats:
                    labels = self.detect_anomalies()
                    num_anomalies = sum(1 for label in labels if label == -1)
                    gauge.set(num_anomalies)
                time.sleep(60)  # Update every minute
        
        start_http_server(8000)
        logging.info("Prometheus monitoring started on port 8000")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(update_metrics)

    def fetch_and_detect(self, api_urls):
        """Fetch process stats from multiple APIs and detect anomalies."""
        all_stats = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(self.fetch_process_stats_from_api, url): url for url in api_urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    stats = future.result()
                    if stats:
                        all_stats.extend(stats)
                except Exception as e:
                    logging.error(f"Error fetching data from {url}: {e}")
        
        if all_stats:
            labels = self.detect_anomalies(all_stats)
            success = self.log_results(all_stats, labels)
            if success:
                logging.info("Results logged successfully.")
            else:
                logging.error("Failed to log the statistics. Please check permissions and disk space.")

def main():
    detector = ProcessAnomalyDetector()
    
    # Generate and train model if necessary
    if not detector.model:
        simulated_data = detector.simulate_process_stats(10)
        logging.info("Generating sample data for training the model...")
        detector.train_model(simulated_data)
        
    current_stats = detector.simulate_process_stats(5)
    logging.info("\nDetecting anomalies in new process statistics.")
    
    try:
        labels = detector.detect_anomalies(current_stats)
        logging.info(f"Detected {sum(1 for label in labels if label == -1)} anomalies out of {len(labels)} processes.")
        
        # Log the results
        success = detector.log_results(current_stats, labels)
        if success:
            logging.info("Results logged successfully.")
        else:
            logging.error("Failed to log the statistics. Please check permissions and disk space.")
            
    except ValueError as ve:
        logging.error(f"Model not trained: {ve}")
    except Exception as e:
        logging.error(f"Unexpected error during processing: {e}")

if __name__ == "__main__":
    main()

# Example usage with multiple APIs
if __name__ == "__main__":
    detector = ProcessAnomalyDetector()
    
    # Generate and train model if necessary
    if not detector.model:
        simulated_data = detector.simulate_process_stats(10)
        logging.info("Generating sample data for training the model...")
        detector.train_model(simulated_data)
        
    api_urls = [
        'https://example.com/process-stats',
        'https://another-example.com/process-stats'
    ]
    
    # Start monitoring
    detector.start_monitoring()
    
    # Fetch and detect anomalies from multiple APIs
    detector.fetch_and_detect(api_urls)
