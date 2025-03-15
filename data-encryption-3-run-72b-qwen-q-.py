import os
import sys
import logging
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from optuna.integration.tensorflow_keras import TFKerasPruningCallback
import optuna
import tflite_runtime.interpreter as tflite
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def install_dependencies():
    dependencies = [
        "tensorflow",
        "optuna",
        "tflite-runtime",
        "requests"
    ]
    for dependency in dependencies:
        os.system(f"pip install {dependency}")

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

def gather_data(config, df):
    # Gather data from other sources
    additional_data = []
    for source in config.get('data_sources', []):
        if 'file' in source:
            additional_df = load_data(source['file'])
            additional_data.append(additional_df)
        elif 'url' in source:
            response = requests.get(source['url'])
            additional_df = pd.read_csv(response.text)
            additional_data.append(additional_df)
    
    df = pd.concat([df] + additional_data, ignore_index=True)
    logging.info("Data gathered from all sources.")
    return df

def preprocess_data(df):
    # Example preprocessing: fill missing values and convert categorical variables
    df.fillna(0, inplace=True)
    logging.info("Data preprocessed successfully.")
    return df

def detect_anomalies(data):
    z_scores = (data - data.mean()) / data.std()
    return (z_scores.abs() > 3).any(axis=1)

def handle_anomalies(data, anomalies):
    data.drop(data[anomalies].index, inplace=True)
    logging.info("Anomalies handled successfully.")

def augment_data(X, y):
    n_augment = 5
    X_augmented = []
    y_augmented = []
    
    for i in range(n_augment):
        noise = np.random.normal(0, 0.1, X.shape)
        X_augmented.append(X + noise)
        y_augmented.extend(y)
    
    return np.vstack(X_augmented), np.array(y_augmented)

def create_model(trial):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(trial.suggest_int('units', 32, 128), activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(trial.suggest_float('learning_rate', 0.001, 0.1)),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def objective(trial):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2)
    
    model = create_model(trial)
    
    history = model.fit(
        X_train_split,
        y_train_split,
        validation_data=(X_val_split, y_val_split),
        epochs=10,
        batch_size=32,
        callbacks=[TFKerasPruningCallback(trial, 'val_accuracy')]
    )
    
    return history.history['val_accuracy'][-1]

def train_model_with_tuning(X_train, y_train, X_val, y_val):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    best_params = study.best_params
    best_model = create_model(study.best_trial)
    
    best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    logging.info("Model trained successfully.")
    return best_model

def convert_to_tflite(model, input_shape):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    os.system('edgetpu_compiler -s model.tflite')
    logging.info("Model converted to TFLite and optimized for Edge TPU.")
    return 'model_edgetpu.tflite'

def run_tflite_model(tflite_model_path, X_val_reshaped):
    interpreter = tflite.Interpreter(model_path=tflite_model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_data = X_val_reshaped.astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    logging.info("Inference completed on Edge TPU.")
    return output_data

def main(config_path):
    install_dependencies()
    
    config = load_config(config_path)
    
    file_path = config['data_file']
    df = load_data(file_path)
    
    df = gather_data(config, df)
    
    df = preprocess_data(df)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    anomalies = detect_anomalies(X)
    handle_anomalies(X, anomalies)
    handle_anomalies(y, anomalies)
    
    X, y = augment_data(X.values, y.values)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_model = train_model_with_tuning(X_train, y_train, X_val, y_val)
    
    tflite_model_path = convert_to_tflite(best_model, (1, -1))
    
    predictions = run_tflite_model(tflite_model_path, X_val)
    accuracy = np.mean(np.round(predictions.squeeze()) == y_val)
    logging.info(f"Model validation accuracy with Edge TPU: {accuracy:.4f}")
    
    best_model.save('best_model.h5')
    logging.info("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and deploy a model using TensorFlow and Edge TPU.")
    parser.add_argument("config_path", help="Path to the configuration file.")
    args = parser.parse_args()
    
    main(args.config_path)
