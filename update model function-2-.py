import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import logging

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
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    X_new = pad_sequences(sequences, maxlen=500)
    
    return X_new
