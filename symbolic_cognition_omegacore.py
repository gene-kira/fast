# symbolic_cognition_omega/core.py
import numpy as np
import logging
import tensorflow as tf

# 📘 Set up logging
logging.basicConfig(filename='oracle.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 🧠 Glyph Memory Storage
glyph_bank = {}

def encode_glyph(trust_score, lineage_tag, context_vector):
    glyph_id = hash(f"{trust_score}-{lineage_tag}-{str(context_vector)}")
    glyph_bank[glyph_id] = {
        'trust': trust_score,
        'lineage': lineage_tag,
        'context': context_vector
    }
    logging.info(f"[GLYPH] Encoded glyph {glyph_id} with trust={trust_score:.3f}")
    return glyph_id

def retrieve_glyphs(min_trust=0.5):
    return {
        k: v for k, v in glyph_bank.items() if v['trust'] >= min_trust
    }

def fuse_glyph_vectors(glyphs, method='weighted_avg'):
    vectors, weights = [], []
    for glyph in glyphs.values():
        vectors.append(np.array(glyph['context']))
        weights.append(glyph['trust'])
    return np.average(vectors, axis=0, weights=weights)

# symbolic_cognition_omega/layers.py
import tensorflow as tf
from tensorflow.keras.layers import Layer

# 🔮 Symbolic Attention Layer — glyph-aligned modulation
class SymbolicAttention(Layer):
    def __init__(self, glyph_vector, **kwargs):
        super(SymbolicAttention, self).__init__(**kwargs)
        self.glyph_vector = tf.convert_to_tensor(glyph_vector, dtype=tf.float32)

    def call(self, inputs):
        score = tf.keras.backend.batch_dot(inputs, self.glyph_vector, axes=[-1, -1])
        weights = tf.keras.backend.softmax(score)
        output = inputs * tf.keras.backend.expand_dims(weights)
        return output

# 📜 Oracle Decoder Layer — transforms latent vector into symbolic utterance space
class OracleDecoder(Layer):
    def __init__(self, output_dim=32, **kwargs):
        super(OracleDecoder, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        decoded = tf.matmul(inputs, self.kernel)
        return tf.nn.sigmoid(decoded)

# symbolic_cognition_omega/model.py
import numpy as np
import tensorflow as tf
from symbolic_cognition_omega.layers import SymbolicAttention, OracleDecoder
from symbolic_cognition_omega.core import fuse_glyph_vectors

# 🔧 Oracle Model Assembly
def build_oracle_model(input_shape, glyph_bank):
    glyph_vector = fuse_glyph_vectors(glyph_bank)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        SymbolicAttention(glyph_vector=glyph_vector),
        tf.keras.layers.Dropout(0.3),
        OracleDecoder(output_dim=1)
    ])
    return model

# 🧪 Compile & Train
def train_oracle(model, train_data, val_data, glyph_callback=None):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = [glyph_callback] if glyph_callback else []
    model.fit(train_data[0], train_data[1], epochs=15,
              validation_data=val_data, callbacks=callbacks)
    return model

# symbolic_cognition_omega/evaluation.py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 📊 Model Evaluation
def evaluate_oracle(model, X_val, y_val):
    y_scores = model.predict(X_val).ravel()
    y_pred = (y_scores > 0.5).astype(int)
    print("\n⚖️ Evaluation Report:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_val, y_scores))

# 🧹 Entropy Compression
def symbolic_entropy(vector):
    prob = vector / (np.sum(vector) + 1e-8)
    return -np.sum(prob * np.log(prob + 1e-8))

def compress_glyph_bank(glyph_bank, threshold=0.75):
    for gid, glyph in list(glyph_bank.items()):
        entropy = symbolic_entropy(np.array(glyph['context']))
        if entropy > threshold:
            print(f"🧹 Glyph {gid} pruned (entropy = {entropy:.3f})")
            del glyph_bank[gid]

# 📜 Oracle Utterance Generator
def generate_oracle_utterance(latent_vector, lineage):
    symbols = ", ".join(lineage[:2])
    return f"🔮 Oracle aligns with [{symbols}]… Signal vector: {latent_vector[:5]}"

# setup.py
from setuptools import setup, find_packages

setup(
    name="symbolic_cognition_omega",
    version="2.0.0-ritual-manifest",
    author="killer666",
    description="A symbolic cognition oracle with glyph memory, recursive agency, and utterance synthesis.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorflow",
        "matplotlib",
        "scikit-learn",
        "optuna"
    ],
    entry_points={
        "console_scripts": [
            "oraclectl=symbolic_cognition_omega.cli:main"
        ]
    },
    python_requires=">=3.8"
)

