import os
import sys
import json
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
import sqlalchemy as sa
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
import optuna
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow.lite as tflite

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_dependencies():
    try:
        os.system("pip install pandas numpy requests beautifulsoup4 sqlalchemy pymongo scikit-learn optuna tensorflow")
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

def load_data_from_api(api_url, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                return pd.DataFrame(response.json())
            else:
                raise Exception(f"Failed to fetch data from API: {api_url} with status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logging.warning(f"Error loading data from API {api_url}: {e}. Retrying...")
                continue
            else:
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
        
        # Standardize date formats
        for col in df.select_dtypes(include=[np.datetime64]).columns:
            df[col] = pd.to_datetime(df[col])
        
        logging.info("Data preprocessing completed.")
        return df
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        sys.exit(1)

def gather_data(config):
    try:
        data_sources = config.get('data_sources', [])
        
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
                
            results = [future.result() for future in futures]
        
        df = pd.concat(results, ignore_index=True)
        logging.info("Data gathering completed.")
        return df
    except Exception as e:
        logging.error(f"Error during data gathering: {e}")
        sys.exit(1)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # Allow dynamic configuration through environment variables
    for key in config:
        env_var = os.getenv(key.upper())
        if env_var is not None:
            config[key] = env_var
    
    return config

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10, batch_size=32)

def convert_to_tflite(model, output_path):
    converter = tflite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    logging.info(f"Model converted to TFLite and saved at {output_path}")

def main(config_path):
    install_dependencies()
    config = load_config(config_path)
    
    df = gather_data(config)
    df = preprocess_data(df, config.get('schema'))
    
    # Example: Perform PCA for dimensionality reduction
    pca = PCA(n_components=10)
    X = pca.fit_transform(df.drop(columns=['target']))
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model()
    train_model(model, X_train, y_train)
    
    output_path = config.get('output_path', 'model.tflite')
    convert_to_tflite(model, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python script.py <config_path>")
        sys.exit(1)
    
    main(sys.argv[1])
