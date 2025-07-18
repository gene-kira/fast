import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk

# Install libraries if missing
def install_libraries_gui():
    required_libs = [
        "numpy", "scipy", "tensorflow", "skopt",
        "scikit-image", "tensorflow_model_optimization"
    ]
    for lib in required_libs:
        try:
            __import__(lib)
            status.set(f"✓ {lib} already installed")
        except ImportError:
            status.set(f"Installing {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# GUI setup
app = tk.Tk()
app.title("🔮 MagicBox Ritual Launcher")
status = tk.StringVar()
status.set("Ready to launch ritual...")

ttk.Label(app, text="MagicBox Ritual Launcher", font=("Segoe UI", 14)).pack(pady=10)

# Sliders for hyperparameters
ttk.Label(app, text="Learning Rate:").pack()
lr_slider = ttk.Scale(app, from_=1e-5, to=1e-3, orient="horizontal", length=300)
lr_slider.set(1e-4)
lr_slider.pack()

ttk.Label(app, text="Dropout Rate:").pack()
dropout_slider = ttk.Scale(app, from_=0.1, to=0.9, orient="horizontal", length=300)
dropout_slider.set(0.5)
dropout_slider.pack()

# Toggles
augment_var = tk.BooleanVar(value=True)
prune_var = tk.BooleanVar(value=True)
ttk.Checkbutton(app, text="Enable Data Augmentation", variable=augment_var).pack(pady=4)
ttk.Checkbutton(app, text="Enable Model Pruning", variable=prune_var).pack(pady=4)

# Launch button
ttk.Button(app, text="Start Magic", command=lambda: threading.Thread(target=lambda: [install_libraries_gui(), run_magic_pipeline()]).start()).pack(pady=10)

# Status display
ttk.Label(app, textvariable=status, wraplength=400, font=("Segoe UI", 10)).pack(pady=10)

def run_magic_pipeline():
    status.set("🔮 Starting the MagicBox ritual...")

    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import AdamW
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    import tensorflow_model_optimization as tfmot

    # Gather GUI inputs
    best_lr = lr_slider.get()
    best_dropout = dropout_slider.get()
    enable_aug = augment_var.get()
    enable_prune = prune_var.get()

    def load_data():
        X = np.random.rand(1000, 32, 32, 3) * 255.0
        y = np.random.randint(0, 10, size=(1000,))
        return X, y

    def create_data_augmentor():
        return ImageDataGenerator(
            rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
            horizontal_flip=True, vertical_flip=True, brightness_range=[0.8, 1.2],
            zoom_range=0.2, shear_range=0.2, channel_shift_range=30, fill_mode='nearest'
        )

    def resnet_bottleneck_block(x, filters, stride=1):
        shortcut = x
        x = Conv2D(filters//4, (1,1), strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Conv2D(filters//4, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, (1,1), strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(shortcut)
            shortcut = BatchNormalization()(shortcut)
        return Activation('relu')(Add()([x, shortcut]))

    def build_resnet(input_shape, num_classes, dropout_rate):
        inputs = Input(shape=input_shape)
        x = Conv2D(64, (7,7), strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = MaxPooling2D((3,3), strides=2)(x)
        for filters in [64, 128, 256, 512]:
            stride = 2 if filters != 64 else 1
            x = resnet_bottleneck_block(x, filters, stride)
            x = resnet_bottleneck_block(x, filters)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout_rate)(x)
        return Model(inputs=inputs, outputs=Dense(num_classes, activation='softmax')(x))

    # Load and preprocess data
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = build_resnet((32, 32, 3), 10, best_dropout)
        datagen = create_data_augmentor() if enable_aug else ImageDataGenerator()
        datagen.fit(X_train)
        optimizer = AdamW(learning_rate=best_lr, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint('best_model_final.h5', save_best_only=True)
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(datagen.flow(X_train, y_train, batch_size=32),
                  epochs=20, validation_data=(X_val, y_val),
                  callbacks=[checkpoint, early_stopping])

        if enable_prune:
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            pruned_model = prune_low_magnitude(model)
            pruned_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            pruned_model.fit(datagen.flow(X_train, y_train, batch_size=32),
                             epochs=5, validation_data=(X_val, y_val),
                             callbacks=[checkpoint, early_stopping])
            final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
            final_model.save('final_pruned_model.h5')
            status.set("✨ Ritual complete! Pruned model saved as final_pruned_model.h5")
        else:
            model.save('final_model.h5')
            status.set("✨ Ritual complete! Model saved as final_model.h5")

# Keep GUI running
app.mainloop()

