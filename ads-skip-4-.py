import cv2
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.resnet50 import ResNet50

# Function to scan ports for active streams
def scan_ports_for_streams(start_port=8000, end_port=9000):
    active_streams = []
    for port in range(start_port, end_port + 1):
        cap = cv2.VideoCapture(f"udp://@:{port}")
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                active_streams.append((port, len(frame)))
            cap.release()
    return active_streams

# Function to select the best stream based on signal quality
def select_best_stream(active_streams):
    if not active_streams:
        return None
    best_stream = max(active_streams, key=lambda x: x[1])
    return f"udp://@:{best_stream[0]}"

# Function to extract frames from a live stream
def extract_frames_from_stream(stream_url, frame_rate=24):
    cap = cv2.VideoCapture(stream_url)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) > 100:  # Limit the number of frames to avoid memory issues
            break
        time.sleep(1 / frame_rate)
    cap.release()
    return frames

# Function to extract audio from a live stream
def extract_audio_from_stream(stream_url, sr=44100):
    y, _ = librosa.load(stream_url, sr=sr)
    return y

# Function to preprocess and buffer frames and audio
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def preprocess_audio(y, sr=44100):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = cv2.resize(mel_spectrogram, (128, 128))
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=(0, -1))
    return mel_spectrogram

# Function to create the image model
def create_image_model():
    model = Sequential()
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Function to create the audio model
def create_audio_model():
    model = Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# Function to create the ensemble model
def create_ensemble_model(image_model, audio_model):
    image_input = Input(shape=(224, 224, 3))
    audio_input = Input(shape=(128, 128, 1))

    image_output = image_model(image_input)
    audio_output = Flatten()(audio_model(audio_input))
    
    combined = concatenate([image_output, audio_output])
    final_output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[image_input, audio_input], outputs=final_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to detect yellow line in a frame
def detect_yellow_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_line_detected = cv2.countNonZero(mask) > 0
    return yellow_line_detected

# Function to play live stream and handle user feedback with mouse click and yellow line detection
def play_live_stream_and_handle_feedback(stream_url, ensemble_model):
    cap = cv2.VideoCapture(stream_url)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    ad_segments = []
    user_feedback = []

    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            start_time = current_time - (1000 / frame_rate)
            end_time = current_time + (1000 / frame_rate)
            user_feedback.append((start_time, end_time))

    cv2.namedWindow('Live Stream')
    cv2.setMouseCallback('Live Stream', on_mouse_click)

    sliding_window_size = 5
    image_buffer = []
    audio_buffer = []
    ad_timer = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow('Live Stream', frame)

        # Detect yellow line
        yellow_line_detected = detect_yellow_line(frame)
        if yellow_line_detected:
            ad_timer = 10 * frame_rate  # Set timer to skip 10 seconds of ads

        if ad_timer > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + (1000 / frame_rate))
            ad_timer -= 1
            continue

        # Preprocess and buffer frames and audio
        image_frame = preprocess_frame(frame)
        image_buffer.append(image_frame)

        y, sr = extract_audio_from_stream(stream_url, sr=44100)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        audio_frame = preprocess_audio(mel_spectrogram)
        audio_buffer.append(audio_frame)

        if len(image_buffer) >= sliding_window_size:
            image_input = np.concatenate(image_buffer[-sliding_window_size:], axis=0)
            audio_input = np.array([audio_buffer[-1]])

            # Make prediction using the ensemble model
            prediction = ensemble_model.predict([image_input, audio_input])
            ad_detected = prediction[0][0] > 0.5

            # Skip if an ad is detected
            if ad_detected:
                cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + (1000 / frame_rate))

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return user_feedback

# Mock evaluation functions
def mock_image_data(num_samples):
    return np.random.rand(num_samples, 224, 224, 3), np.random.randint(0, 2, num_samples)

def mock_audio_data(num_samples):
    return np.random.rand(num_samples, 128, 128, 1), np.random.randint(0, 2, num_samples)

# Train the models with mock data
image_model = create_image_model()
audio_model = create_audio_model()

X_img, y_img = mock_image_data(100)
X_aud, y_aud = mock_audio_data(100)

image_model.fit(X_img, y_img, epochs=10, batch_size=32)
audio_model.fit(np.expand_dims(X_aud, axis=1), y_aud, epochs=10, batch_size=32)

# Create the ensemble model
ensemble_model = create_ensemble_model(image_model, audio_model)

# Scan for active streams and select the best one
active_streams = scan_ports_for_streams()
best_stream_url = select_best_stream(active_streams)

if best_stream_url:
    user_feedback = play_live_stream_and_handle_feedback(best_stream_url, ensemble_model)
    print("User feedback:", user_feedback)
else:
    print("No active streams found.")
