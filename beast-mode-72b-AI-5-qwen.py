import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, log_loss, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import optuna
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up logging
logging.basicConfig(filename='model_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables to store training and validation data
X_train = None
X_val = None
y_train = None
y_val = None

# Function to load and preprocess data using tf.data API
def load_and_preprocess_data(seq_length=3, num_samples=1000, height=64, width=64, channels=3):
    # Generate random sequences of images for demonstration
    X_seq = np.random.rand(num_samples, seq_length, height, width, channels)
    y_seq = np.random.randint(0, 2, num_samples)

    # Convert to TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((X_seq, y_seq))
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))  # Normalize images

    return dataset

# Function to split the dataset into training and validation sets
def split_dataset(dataset, test_size=0.2):
    num_samples = len(list(dataset))
    val_size = int(num_samples * test_size)
    train_size = num_samples - val_size

    train_dataset = dataset.take(train_size).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = dataset.skip(train_size).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset

# Function to plot and save evaluation metrics
def evaluate_and_plot(model, val_dataset):
    # Evaluate the model on the validation set
    y_true = []
    y_pred = []

    for x, y in val_dataset:
        predictions = model.predict(x)
        y_true.extend(y.numpy())
        y_pred.extend(predictions)

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)

    # Calculate evaluation metrics
    evaluate_model(model, y_true, y_pred)

# Function to calculate and log evaluation metrics
def evaluate_model(model, y_true, y_pred):
    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(y_true, y_pred, verbose=1)
    
    logging.info(f"Validation Loss: {val_loss}")
    logging.info(f"Validation Accuracy: {val_accuracy}")

    # Additional metrics
    cohen_kappa = cohen_kappa_score(y_true, (y_pred > 0.5).astype(int))
    mcc = matthews_corrcoef(y_true, (y_pred > 0.5).astype(int))
    balanced_acc = balanced_accuracy_score(y_true, (y_pred > 0.5).astype(int))
    log_loss_value = log_loss(y_true, y_pred)
    
    logging.info(f"Cohen's Kappa: {cohen_kappa}")
    logging.info(f"Matthews Correlation Coefficient (MCC): {mcc}")
    logging.info(f"Balanced Accuracy: {balanced_acc}")
    logging.info(f"Log-Loss: {log_loss_value}")

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    
    # Build and compile the model
    input_shape = (3, 64, 64, 3)  # Example input shape
    num_classes = 1  # Binary classification for demonstration
    model = build_temporal_cnn(input_shape, num_classes)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # Train the model using k-fold cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_idx, val_idx in kfold.split(X_train, y_train):
        X_train_kf, y_train_kf = X_train[train_idx], y_train[train_idx]
        X_val_kf, y_val_kf = X_train[val_idx], y_train[val_idx]
        
        # Convert to TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_kf, y_train_kf))
        train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val_kf, y_val_kf))
        val_dataset = val_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        
        history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, verbose=1)
        
        val_loss, val_accuracy = model.evaluate(val_dataset, verbose=1)
        fold_scores.append(val_accuracy)
    
    mean_val_accuracy = np.mean(fold_scores)
    return mean_val_accuracy

# Function to build a temporal CNN model
def build_temporal_cnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model

# Function to save the model with versioning
def save_model(model, version):
    model.save(f'model_v{version}.h5')
    logging.info(f"Model saved as model_v{version}.h5")

# Main function
def main():
    global X_train, y_train, X_val, y_val
    
    # Load and preprocess data
    dataset = load_and_preprocess_data()
    
    # Split data into training and validation sets
    train_dataset, val_dataset = split_dataset(dataset)
    
    # Convert datasets to numpy arrays for k-fold cross-validation
    X_train, y_train = next(iter(train_dataset.unbatch().batch(len(train_dataset))))
    X_val, y_val = next(iter(val_dataset.unbatch().batch(len(val_dataset))))

    # Optimize hyperparameters using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    best_params = study.best_params
    logging.info(f"Best Hyperparameters: {best_params}")
    
    # Build and compile the final model with best hyperparameters
    input_shape = (3, 64, 64, 3)  # Example input shape
    num_classes = 1  # Binary classification for demonstration
    final_model = build_temporal_cnn(input_shape, num_classes)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    final_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # Train the final model
    history = final_model.fit(train_dataset, epochs=30, validation_data=val_dataset, verbose=1)
    
    # Evaluate the final model
    evaluate_and_plot(final_model, val_dataset)
    
    # Save the final model with versioning
    save_model(final_model, 1)
    
    # Prune and quantize the model
    pruned_model = sparsity.prune_low_magnitude(final_model)
    pruned_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    pruned_model.fit(train_dataset, epochs=10, validation_data=val_dataset, verbose=1)
    
    sparsity.strip_pruning(pruned_model)
    
    # Evaluate pruned model with sophisticated metrics
    evaluate_and_plot(pruned_model, val_dataset)
    
    # Quantize the pruned model
    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    tflite_model = converter.convert()
    
    # Save the quantized model to a file
    with open('pruned_quantized_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    logging.info("Quantized and pruned model saved as pruned_quantized_model.tflite")

if __name__ == "__main__":
    main()
