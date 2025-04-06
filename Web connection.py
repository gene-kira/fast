import requests
from bs4 import BeautifulSoup
from collections import deque
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import csv

# Set up logging
logging.basicConfig(level=logging.INFO)

def explore_links(initial_url, max_depth, data_file):
    # Initialize the queue and visited set
    queue = deque([(initial_url, 0)])
    visited = set([initial_url])
    
    with open(data_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['url', 'content'])  # Define the CSV columns
        
        while queue:
            url, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Process the current page
                    content = process_page(soup)
                    writer.writerow([url, content])
                    
                    # Find all links on the current page
                    for link in soup.find_all('a', href=True):
                        next_url = link['href']
                        
                        # Ensure the URL is absolute
                        if not next_url.startswith('http'):
                            next_url = f"{url.rstrip('/')}/{next_url.lstrip('/')}"
                        
                        if next_url not in visited:
                            queue.append((next_url, depth + 1))
                            visited.add(next_url)
                else:
                    logging.warning(f"Failed to fetch {url} with status code {response.status_code}")
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")

def process_page(soup):
    # Example: Extract all text content from the page
    paragraphs = soup.find_all('p')
    return ' '.join([paragraph.get_text() for paragraph in paragraphs])

def update_model(model_path, data_file):
    # Load the existing model
    model = load_model(model_path)
    
    # Fetch new data from the CSV file
    new_data = pd.read_csv(data_file)
    X_new = new_data['content'].values
    y_new = new_data['label'].values  # Assuming you have labels in a column named 'label'
    
    # Preprocess new data if necessary (e.g., normalization, reshaping)
    X_new = preprocess_data(X_new)
    
    # Define a learning rate that is lower than the initial training to ensure fine-tuning
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model on the new data with a smaller number of epochs and batch size
    history = model.fit(X_new, y_new, epochs=5, batch_size=32, verbose=1)
    
    # Save the updated model
    model.save(model_path)
    logging.info(f"Model updated and saved at {model_path}")

def preprocess_data(data):
    # Example preprocessing: Tokenization and padding for text data
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    X_new = pad_sequences(sequences, maxlen=500)
    
    return X_new

# Define initial URL and maximum depth for exploration
initial_url = "https://example.com"
max_depth = 3
data_file = "collected_data.csv"

# Define the path to your model
model_path = "path/to/your/model.h5"

# Explore links and collect data
explore_links(initial_url, max_depth, data_file)

# Update the machine learning model with the collected data
update_model(model_path, data_file)
