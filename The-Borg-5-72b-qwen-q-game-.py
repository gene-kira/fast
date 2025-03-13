import os
import sys
import logging
import argparse
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import sqlalchemy as sa
from pymongo import MongoClient
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_dependencies():
    try:
        os.system("pip install pandas numpy requests beautifulsoup4 sqlalchemy pymongo scikit-learn")
        logging.info("Dependencies installed successfully.")
    except Exception as e:
        logging.error(f"Error installing dependencies: {e}")
        sys.exit(1)

def load_data_from_file(file_path, data_format):
    try:
        if data_format == 'csv':
            return pd.read_csv(file_path)
        elif data_format == 'json':
            return pd.read_json(file_path)
        elif data_format == 'excel':
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    except Exception as e:
        logging.error(f"Error loading data from file {file_path}: {e}")
        sys.exit(1)

def load_data_from_api(api_url):
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            raise Exception(f"Failed to fetch data from API: {api_url}")
    except Exception as e:
        logging.error(f"Error loading data from API {api_url}: {e}")
        sys.exit(1)

def load_data_from_web(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')  # Example: Extracting data from a table
        return pd.read_html(str(table))[0]
    except Exception as e:
        logging.error(f"Error loading data from web {url}: {e}")
        sys.exit(1)

def load_data_from_database(db_config):
    try:
        engine = sa.create_engine(f"{db_config['dialect']}://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
        query = db_config.get('query', 'SELECT * FROM table_name')
        return pd.read_sql_query(query, engine)
    except Exception as e:
        logging.error(f"Error loading data from database: {e}")
        sys.exit(1)

def load_data_from_mongodb(mongo_config):
    try:
        client = MongoClient(mongo_config['host'], mongo_config['port'])
        db = client[mongo_config['database']]
        collection = db[mongo_config['collection']]
        return pd.DataFrame(list(collection.find()))
    except Exception as e:
        logging.error(f"Error loading data from MongoDB: {e}")
        sys.exit(1)

def preprocess_data(df, schema=None):
    try:
        if schema:
            for column, dtype in schema.items():
                df[column] = df[column].astype(dtype)
        
        # Handle missing values
        df.fillna(method='ffill', inplace=True)
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        logging.info("Data preprocessing completed.")
        return df
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        sys.exit(1)

def gather_data(config):
    try:
        data_sources = config.get('data_sources', [])
        game_data_path = config.get('game_data_file', None)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for source in data_sources:
                if source['type'] == 'file':
                    futures.append(executor.submit(load_data_from_file, source['file_path'], source.get('format', 'csv')))
                elif source['type'] == 'api':
                    futures.append(executor.submit(load_data_from_api, source['url']))
                elif source['type'] == 'web':
                    futures.append(executor.submit(load_data_from_web, source['url']))
                elif source['type'] == 'database':
                    futures.append(executor.submit(load_data_from_database, source))
                elif source['type'] == 'mongodb':
                    futures.append(executor.submit(load_data_from_mongodb, source))
            
            if game_data_path:
                futures.append(executor.submit(load_data_from_file, game_data_path, 'csv'))
            
            results = [future.result() for future in futures]
        
        df = pd.concat(results, ignore_index=True)
        logging.info("Data gathering completed.")
        return df
    except Exception as e:
        logging.error(f"Error during data gathering: {e}")
        sys.exit(1)

def integrate_game_data(df, game_df):
    try:
        if 'player_id' in df.columns and 'player_id' in game_df.columns:
            df = pd.merge(df, game_df, on='player_id', how='left')
        else:
            logging.warning("No common key found for merging main data with game data.")
        
        logging.info("Game data integrated successfully.")
        return df
    except Exception as e:
        logging.error(f"Error integrating game data: {e}")
        sys.exit(1)

def create_model(trial):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(trial.suggest_int('units', 64, 256), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(trial.suggest_int('units', 32, 128), activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(trial.suggest_float('learning_rate', 0.001, 0.1)),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model_with_tuning(X_train, y_train, X_val, y_val):
    def objective(trial):
        model = create_model(trial)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)
        val_loss = history.history['val_loss'][-1]
        return val_loss
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    
    best_params = study.best_params
    best_model = create_model(study.best_trial)
    best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)
    
    return best_model

def convert_to_tflite(model, model_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    logging.info(f"TFLite model saved to {model_path}")
    return model_path

def run_tflite_model(model_path, X):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_data = np.array(X, dtype=np.float32).reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def main(config_path):
    install_dependencies()
    
    config = load_config(config_path)
    
    df = gather_data(config)
    df = preprocess_data(df, schema=config.get('data_schema'))
    
    game_data_path = config.get('game_data_file', None)
    if game_data_path:
        game_df = load_game_data(game_data_path, 'csv')
        game_df = preprocess_data(game_df, schema=config.get('game_data_schema'))
        df = integrate_game_data(df, game_df)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    anomalies = detect_anomalies(X)
    handle_anomalies(X, anomalies)
    handle_anomalies(y, anomalies)
    
    X, y = augment_data(X.values, y.values)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_model = train_model_with_tuning(X_train, y_train, X_val, y_val)
    
    tflite_model_path = convert_to_tflite(best_model, 'model.tflite')
    
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
