# Step 1: Automatic installation of required libraries

import subprocess
import sys
import os
from pathlib import Path

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    'librosa', 
    'numpy', 
    'tensorflow', 
    'scipy', 
    'matplotlib',
    'tqdm'
]

for package in required_packages:
    install(package)

# Step 2: Import necessary libraries

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Activation
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Step 3: Data collection (synthetic data for demonstration)

def generate_noisy_signal(signal, noise_ratio=0.5):
    noise = np.random.normal(0, noise_ratio * np.std(signal), signal.shape)
    return signal + noise

def create_synthetic_data(num_samples=1000, sample_length=8000, sr=8000):
    X_clean = []
    X_noisy = []

    for _ in range(num_samples):
        t = np.linspace(0, 1, sample_length)
        freq = np.random.uniform(200, 300)  # Random frequency between 200 and 300 Hz
        signal = np.sin(2 * np.pi * freq * t)  # Generate a sine wave

        noisy_signal = generate_noisy_signal(signal, noise_ratio=0.5)
        
        X_clean.append(signal.reshape(-1, 1))
        X_noisy.append(noisy_signal.reshape(-1, 1))

    return np.array(X_clean), np.array(X_noisy)

X_clean, X_noisy = create_synthetic_data()

# Step 4: Data Augmentation

def augment_data(X_clean, X_noisy):
    augmented_X_clean = []
    augmented_X_noisy = []

    for clean, noisy in zip(X_clean, X_noisy):
        # Add time-shift augmentation
        shift_amount = np.random.randint(0, len(clean) // 4)
        shifted_clean = np.roll(clean, shift_amount)
        shifted_noisy = np.roll(noisy, shift_amount)

        augmented_X_clean.append(shifted_clean.reshape(-1, 1))
        augmented_X_noisy.append(shifted_noisy.reshape(-1, 1))

        # Add pitch-shift augmentation
        pitch_factor = np.random.uniform(0.9, 1.1)
        shifted_clean = librosa.effects.pitch_shift(clean.flatten(), sr=8000, n_steps=pitch_factor)
        shifted_noisy = librosa.effects.pitch_shift(noisy.flatten(), sr=8000, n_steps=pitch_factor)

        augmented_X_clean.append(shifted_clean.reshape(-1, 1))
        augmented_X_noisy.append(shifted_noisy.reshape(-1, 1))

    return np.array(augmented_X_clean), np.array(augmented_X_noisy)

X_clean_aug, X_noisy_aug = augment_data(X_clean, X_noisy)
X_clean = np.concatenate([X_clean, X_clean_aug])
X_noisy = np.concatenate([X_noisy, X_noisy_aug])

# Step 5: Preprocessing

def preprocess_data(X_clean, X_noisy, sr=8000, n_fft=512, hop_length=256):
    X_clean_spectrogram = []
    X_noisy_spectrogram = []

    for clean, noisy in tqdm(zip(X_clean, X_noisy), total=len(X_clean)):
        clean_spec = np.abs(librosa.stft(clean.flatten(), n_fft=n_fft, hop_length=hop_length))
        noisy_spec = np.abs(librosa.stft(noisy.flatten(), n_fft=n_fft, hop_length=hop_length))

        # Normalize spectrograms
        clean_spec = librosa.util.normalize(clean_spec)
        noisy_spec = librosa.util.normalize(noisy_spec)

        X_clean_spectrogram.append(clean_spec.reshape(*clean_spec.shape, 1))
        X_noisy_spectrogram.append(noisy_spec.reshape(*noisy_spec.shape, 1))

    return np.array(X_clean_spectrogram), np.array(X_noisy_spectrogram)

X_clean_spec, X_noisy_spec = preprocess_data(X_clean, X_noisy)

# Step 6: Model selection and training

def build_unet(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.2)(conv3)

    # Decoder
    up4 = UpSampling2D((2, 2))(conv3)
    merge4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up4)
    merge4 = BatchNormalization()(merge4)
    merge4 = Dropout(0.2)(merge4)

    up5 = UpSampling2D((2, 2))(merge4)
    merge5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up5)
    merge5 = BatchNormalization()(merge5)
    merge5 = Dropout(0.2)(merge5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(merge5)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

input_shape = X_noisy_spec.shape[1:]
model = build_unet(input_shape)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_noisy_spec, X_clean_spec, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Step 7: Evaluation

def plot_history(history):
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)

# Step 8: Demonstration of Sound Enhancement

def enhance_sound(model, noisy_signal, sr=8000, n_fft=512, hop_length=256):
    noisy_spec = np.abs(librosa.stft(noisy_signal.flatten(), n_fft=n_fft, hop_length=hop_length))
    noisy_spec = librosa.util.normalize(noisy_spec)
    noisy_spec = noisy_spec.reshape(1, *noisy_spec.shape, 1)

    enhanced_spec = model.predict(noisy_spec)[0]
    enhanced_spec = np.squeeze(enhanced_spec)

    # Convert back to time domain
    enhanced_signal = librosa.istft(enhanced_spec, hop_length=hop_length)
    
    return enhanced_signal

# Select a sample for demonstration
sample_index = 0
noisy_sample = X_noisy[sample_index].flatten()
clean_sample = X_clean[sample_index].flatten()

# Enhance the noisy sample
enhanced_sample = enhance_sound(model, noisy_sample)

# Plot and listen to the signals
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(noisy_sample)
plt.title('Noisy Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(clean_sample)
plt.title('Clean Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(enhanced_sample)
plt.title('Enhanced Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Optionally, save the enhanced signal to a WAV file
wavfile.write('enhanced_signal.wav', sr, np.int16(enhanced_sample * 32767))

# Step 9: Real-Time Performance Optimization

import tensorflow_model_optimization as tfmot

# Quantize the model for real-time performance
quantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
    num_bits=8, per_axis=False, symmetric=True
)

def apply_quantizer(layer):
    if isinstance(layer, (Conv2D,)):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, quantizer)
    return layer

quantized_model = tfmot.quantization.keras.quantize_apply(model, apply_quantizer)
quantized_model.compile(optimizer='adam', loss='mean_squared_error')
quantized_model.summary()

# Save the quantized model
quantized_model.save('quantized_denoising_model.h5')

print("Quantized model saved as 'quantized_denoising_model.h5'")
