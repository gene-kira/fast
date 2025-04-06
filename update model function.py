import tensorflow as tf
from tensorflow.keras.models import load_model

def update_model(model_path, X_new, y_new):
    # Load the existing model
    model = load_model(model_path)
    
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
