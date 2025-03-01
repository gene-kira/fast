import os
import joblib
import time
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    'benign_files_path': 'path/to/benign/files',
    'malicious_files_path': 'path/to/malicious/files',
    'phishing_emails_csv': 'path/to/phishing/emails.csv',
    'normal_traffic_csv': 'path/to/normal/traffic.csv',
    'malicious_traffic_csv': 'path/to/malicious/traffic.csv',
    'model_save_path': 'models'
}

# Ensure directories exist
os.makedirs(CONFIG['model_save_path'], exist_ok=True)

def collect_samples(path, label):
    samples = []
    for filename in os.listdir(path):
        try:
            with open(os.path.join(path, filename), 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                samples.append((content, label))
        except Exception as e:
            logging.error(f"Error reading file {filename}: {e}")
    return pd.DataFrame(samples, columns=['text', 'label'])

def load_email_data(normal_csv, malicious_csv):
    try:
        normal_df = pd.read_csv(normal_csv)
        malicious_df = pd.read_csv(malicious_csv)
        
        if not all(col in normal_df.columns for col in ['text']):
            logging.error("Normal email CSV missing required columns")
            return None
        if not all(col in malicious_df.columns for col in ['text']):
            logging.error("Malicious email CSV missing required columns")
            return None
        
        normal_df['label'] = 0
        malicious_df['label'] = 1
        combined_df = pd.concat([normal_df, malicious_df], ignore_index=True)
        
        # Advanced text preprocessing
        combined_df['text'] = combined_df['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True).fillna('')
        
        return combined_df
    except Exception as e:
        logging.error(f"Error loading email data: {e}")
        return None

def load_traffic_data(normal_csv, malicious_csv):
    try:
        normal_df = pd.read_csv(normal_csv)
        malicious_df = pd.read_csv(malicious_csv)
        
        if not all(col in normal_df.columns for col in ['feature1', 'feature2']):
            logging.error("Normal traffic CSV missing required columns")
            return None
        if not all(col in malicious_df.columns for col in ['feature1', 'feature2']):
            logging.error("Malicious traffic CSV missing required columns")
            return None
        
        normal_df['label'] = 0
        malicious_df['label'] = 1
        combined_df = pd.concat([normal_df, malicious_df], ignore_index=True)
        
        # Advanced feature preprocessing
        combined_df.fillna(0, inplace=True)
        return combined_df
    except Exception as e:
        logging.error(f"Error loading traffic data: {e}")
        return None

def extract_features(df):
    if 'text' in df.columns:
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        text_features = vectorizer.fit_transform(df['text'])
        feature_df = pd.DataFrame(text_features.toarray())
        return pd.concat([feature_df, df.drop(columns=['text'])], axis=1)
    else:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df.drop(columns='label'))
        feature_df = pd.DataFrame(scaled_features)
        return pd.concat([feature_df, df[['label']]], axis=1)

def objective(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
    }
    
    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    return score

def train_model(X_train, y_train, model_type):
    if model_type == 'xgb':
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)
        best_params = study.best_params
        logging.info(f"Best parameters for XGB: {best_params}")
        
        model = XGBClassifier(**best_params)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    else:
        logging.error(f"Unknown model type: {model_type}")
        return None
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    logging.info(f"F1 Score: {f1}")
    logging.info(f"Classification Report:\n{report}")
    return f1

def save_model(model, filename):
    try:
        joblib.dump(model, os.path.join(CONFIG['model_save_path'], filename))
        logging.info(f"Model saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def load_model(filename):
    try:
        model = joblib.load(os.path.join(CONFIG['model_save_path'], filename))
        logging.info(f"Model loaded from {filename}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def display_alerts(alerts):
    if alerts:
        logging.warning("Alerts detected:")
        for alert in alerts:
            logging.warning(f"- {alert}")
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(alerts)), [1] * len(alerts), tick_label=alerts)
        plt.title('Detected Alerts')
        plt.xlabel('Alerts')
        plt.ylabel('Count')
        plt.show()
    else:
        logging.info("No alerts detected.")

def real_time_monitoring(model_malware, model_phishing, model_intrusion):
    while True:
        try:
            # Collect new data (simulated here)
            benign_samples = collect_samples(CONFIG['benign_files_path'], 0)
            malicious_samples = collect_samples(CONFIG['malicious_files_path'], 1)
            email_data = load_email_data(CONFIG['phishing_emails_csv'], CONFIG['malicious_emails_csv'])
            traffic_data = load_traffic_data(CONFIG['normal_traffic_csv'], CONFIG['malicious_traffic_csv'])

            if email_data is None or traffic_data is None:
                logging.error("Data loading failed. Skipping this iteration.")
                continue

            # Feature extraction
            malware_features = extract_features(benign_samples.append(malicious_samples, ignore_index=True))
            email_features = extract_features(email_data)
            traffic_features = extract_features(traffic_data)

            # Split data into training and test sets
            X_malware_test = malware_features.drop(columns=['label'])
            y_malware_test = malware_features['label']
            X_email_test = email_features.drop(columns=['label'])
            y_email_test = email_features['label']
            X_traffic_test = traffic_features.drop(columns=['label'])
            y_traffic_test = traffic_features['label']

            # Evaluate models
            evaluate_model(model_malware, X_malware_test, y_malware_test)
            evaluate_model(model_phishing, X_email_test, y_email_test)
            evaluate_model(model_intrusion, X_traffic_test, y_traffic_test)

            # Make predictions and generate alerts
            malware_predictions = model_malware.predict(X_malware_test)
            email_predictions = model_phishing.predict(X_email_test)
            traffic_predictions = model_intrusion.predict(X_traffic_test)

            alerts = []
            if any(malware_predictions):
                alerts.append("Suspicious malware detected")
            if any(email_predictions):
                alerts.append("Phishing attempt detected")
            if any(traffic_predictions):
                alerts.append("Suspicious network traffic detected")

            display_alerts(alerts)

        except Exception as e:
            logging.error(f"Error during real-time monitoring: {e}")

        # Sleep for a while before the next iteration
        time.sleep(3600)  # Check every hour

def main():
    try:
        # Collect and load data
        benign_samples = collect_samples(CONFIG['benign_files_path'], 0)
        malicious_samples = collect_samples(CONFIG['malicious_files_path'], 1)
        email_data = load_email_data(CONFIG['phishing_emails_csv'], CONFIG['malicious_emails_csv'])
        traffic_data = load_traffic_data(CONFIG['normal_traffic_csv'], CONFIG['malicious_traffic_csv'])

        if email_data is None or traffic_data is None:
            logging.error("Data loading failed. Exiting...")
            return

        # Feature extraction
        malware_features = extract_features(benign_samples.append(malicious_samples, ignore_index=True))
        email_features = extract_features(email_data)
        traffic_features = extract_features(traffic_data)

        # Split data into training and test sets
        X_malware_train, X_malware_test, y_malware_train, y_malware_test = train_test_split(
            malware_features.drop(columns=['label']), malware_features['label'], test_size=0.2, random_state=42)
        X_email_train, X_email_test, y_email_train, y_email_test = train_test_split(
            email_features.drop(columns=['label']), email_features['label'], test_size=0.2, random_state=42)
        X_traffic_train, X_traffic_test, y_traffic_train, y_traffic_test = train_test_split(
            traffic_features.drop(columns=['label']), traffic_features['label'], test_size=0.2, random_state=42)

        # Train models
        model_malware = train_model(X_malware_train, y_malware_train, 'xgb')
        model_phishing = train_model(X_email_train, y_email_train, 'xgb')
        model_intrusion = train_model(X_traffic_train, y_traffic_train, 'xgb')

        if model_malware is None or model_phishing is None or model_intrusion is None:
            logging.error("Model training failed. Exiting...")
            return

        # Evaluate models
        evaluate_model(model_malware, X_malware_test, y_malware_test)
        evaluate_model(model_phishing, X_email_test, y_email_test)
        evaluate_model(model_intrusion, X_traffic_test, y_traffic_test)

        # Save models
        save_model(model_malware, 'malware_detection_model.pkl')
        save_model(model_phishing, 'phishing_detection_model.pkl')
        save_model(model_intrusion, 'traffic_detection_model.pkl')

        # Start real-time monitoring
        real_time_monitoring(model_malware, model_phishing, model_intrusion)

    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
