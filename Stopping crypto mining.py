import os
import psutil
import time
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Known signatures of crypto mining software
KNOWN_MINERS = [
    "ethminer",
    "claymore",
    "ccminer",
    "xmrig",
    "nicehash",
    "zecwallet"
]

def load_libraries():
    try:
        global psutil, time, Counter, pd, np, RandomForestClassifier, train_test_split, accuracy_score
        import os
        import psutil
        from collections import Counter
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError as e:
        print(f"Failed to import libraries: {e}")

def get_running_processes():
    """Get a list of all running processes with relevant features."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'connections']):
        try:
            process_info = {
                'pid': proc.info['pid'],
                'name': proc.info['name'],
                'cpu_percent': proc.info['cpu_percent'],
                'memory_usage': proc.info['memory_info'].rss,
                'network_connections': len(proc.info['connections'])
            }
            processes.append(process_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

def extract_features(processes):
    """Extract features for machine learning."""
    features = []
    for proc in processes:
        feature_vector = [
            1 if proc['name'].lower() in KNOWN_MINERS else 0,
            proc['cpu_percent'],
            proc['memory_usage'] / (1024 * 1024),  # Convert to MB
            proc['network_connections']
        ]
        features.append(feature_vector)
    return np.array(features)

def train_model(X, y):
    """Train a RandomForestClassifier model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model

def is_crypto_miner(proc, model):
    """Use the trained model to classify a process as a crypto miner."""
    feature_vector = np.array([
        1 if proc['name'].lower() in KNOWN_MINERS else 0,
        proc['cpu_percent'],
        proc['memory_usage'] / (1024 * 1024),  # Convert to MB
        proc['network_connections']
    ]).reshape(1, -1)
    
    prediction = model.predict(feature_vector)
    return bool(prediction[0])

def terminate_crypto_miner(proc):
    """Terminate the identified crypto mining process."""
    try:
        psutil.Process(proc['pid']).terminate()
        print(f"Terminated process: {proc['name']} (PID: {proc['pid']})")
    except psutil.NoSuchProcess:
        pass  # Process already terminated

def monitor_and_terminate(model):
    while True:
        processes = get_running_processes()
        features = extract_features(processes)
        suspect_counter = Counter()

        for proc, feature_vector in zip(processes, features):
            if is_crypto_miner(proc, model):
                suspect_counter[proc['name']] += 1
                terminate_crypto_miner(proc)

        # Print a summary of terminated processes
        if suspect_counter:
            print("Summary of terminated processes:")
            for name, count in suspect_counter.items():
                print(f" - {name}: {count} instances")

        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    load_libraries()
    
    # Collect training data
    known_data = {
        'name': ['ethminer', 'claymore', 'ccminer', 'xmrig', 'nicehash', 'zecwallet', 'chrome', 'explorer'],
        'cpu_percent': [90, 85, 80, 75, 70, 65, 10, 5],
        'memory_usage': [200 * (1024 * 1024), 150 * (1024 * 1024), 100 * (1024 * 1024), 75 * (1024 * 1024), 50 * (1024 * 1024), 25 * (1024 * 1024), 10 * (1024 * 1024), 5 * (1024 * 1024)],
        'network_connections': [10, 9, 8, 7, 6, 5, 3, 2],
        'is_miner': [1, 1, 1, 1, 1, 1, 0, 0]
    }
    
    df = pd.DataFrame(known_data)
    X = df[['name', 'cpu_percent', 'memory_usage', 'network_connections']]
    y = df['is_miner']

    # Convert categorical data to numerical
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[['name']]), columns=encoder.get_feature_names_out(['name']))
    X = pd.concat([X.drop('name', axis=1), X_encoded], axis=1)

    # Train the model
    model = train_model(X.values, y.values)
    
    monitor_and_terminate(model)
