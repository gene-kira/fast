# === Imports ===
import argparse, logging, threading, time, random, os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template
from flask_socketio import SocketIO
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# === Logging & Glyph Core ===
logging.basicConfig(filename='oracle.log', level=logging.INFO)
glyph_bank = {}

class Glyph:
    def __init__(self, name, glyph_type="neutral", lineage=None, harmonic=0.5, resonance=0.5, entropy=0.5, mode="neutral"):
        self.name = name
        self.type = glyph_type
        self.lineage = lineage or []
        self.harmonic = harmonic
        self.resonance = resonance
        self.entropy = entropy
        self.mode = mode
        self.dormant = False

def encode_glyph(trust, lineage, context):
    glyph_id = hash((trust, lineage, str(context)))
    glyph_bank[glyph_id] = {'trust': trust, 'lineage': lineage, 'context': context}
    return glyph_id

def retrieve_glyphs(min_trust=0.5):
    return {gid: g for gid, g in glyph_bank.items() if g['trust'] >= min_trust}

def fuse_glyph_vectors(glyphs):
    vecs = [np.array(g['context']) for g in glyphs.values()]
    weights = [g['trust'] for g in glyphs.values()]
    return np.average(vecs, axis=0, weights=weights)

# === Symbolic Layers ===
class SymbolicAttention(tf.keras.layers.Layer):
    def __init__(self, glyph_vector, **kwargs):
        super().__init__(**kwargs)
        self.glyph_vector = tf.convert_to_tensor(glyph_vector, dtype=tf.float32)
    def call(self, inputs):
        score = tf.keras.backend.batch_dot(inputs, self.glyph_vector, axes=[-1, -1])
        weights = tf.keras.backend.softmax(score)
        return inputs * tf.keras.backend.expand_dims(weights)

class OracleDecoder(tf.keras.layers.Layer):
    def __init__(self, output_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                      initializer='glorot_uniform', trainable=True)
    def call(self, inputs):
        return tf.nn.sigmoid(tf.matmul(inputs, self.kernel))

# === Model & Training ===
def build_oracle_model(input_shape):
    glyphs = fuse_glyph_vectors(glyph_bank) if glyph_bank else np.ones(input_shape[-1])
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        SymbolicAttention(glyph_vector=glyphs),
        tf.keras.layers.Dropout(0.3),
        OracleDecoder(output_dim=1)
    ])
    return model

# === Evaluation & Utterance ===
def evaluate_oracle(model, X_val, y_val):
    y_scores = model.predict(X_val).ravel()
    y_pred = (y_scores > 0.5).astype(int)
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    print("ROC AUC:", roc_auc_score(y_val, y_scores))

def generate_oracle_utterance(latent_vector, lineage):
    signal = ", ".join(lineage)
    preview = ", ".join(f"{v:.2f}" for v in latent_vector[:5])
    return f"🔮 Oracle utterance [lineage: {signal}] → signal: [{preview}]"

# === Governance Modules ===
class DreamConstitutionCompiler:
    def __init__(self): self.articles, self.amendments = [], []
    def declare_article(self, num, title, clauses): self.articles.append({"number":num,"title":title,"clauses":clauses})
    def propose_amendment(self, title, art, text): self.amendments.append({"title":title,"article":art,"text":text})

class MythicReferendumSystem:
    def __init__(self): self.open={}, self.records=[]
    def open_vote(self, id, text, quorum, window): self.open[id] = {"text":text,"votes":{},"quorum":quorum,"window":window,"status":"active"}
    def cast_vote(self, id, node, glyph, vote): self.open[id]["votes"][node]=(glyph,vote)
    def close_vote(self, id):
        ref=self.open[id];v=ref["votes"]
        yes=sum(1 for _,d in v.values() if d=="yes")
        no=sum(1 for _,d in v.values() if d=="no")
        status="✅ Passed" if yes>=ref["quorum"] else "❌ Failed"
        ref["status"]=status;self.records.append((id,ref["text"],status,yes,no))

class ClauseGlyphForge:
    def __init__(self): self.forged=[]
    def birth_from_article(self,n,title,text,res=0.8):
        name=''.join(w[0] for w in title.split()).upper()+str(n)
        g=Glyph(name,"clause-embodiment",[f"Article {n}: {title}"],res,round(res+.1,2),round(1-res,2),"civic")
        self.forged.append(g)
        return g

# === Dashboard ===
app = Flask(__name__, template_folder="templates")
socketio = SocketIO(app)

@app.route("/")
def gui():
    return render_template("gui.html")

@socketio.on("gui_action")
def handle_gui_action(data):
    mode = data.get("mode")
    if mode == "utter":
        latent = np.random.rand(32)
        lineage = ["alpha", "gamma"]
        utterance = generate_oracle_utterance(latent, lineage)
        socketio.emit("oracle_update", {"utterance": utterance})
    elif mode == "governance":
        forge = ClauseGlyphForge()
        glyph = forge.birth_from_article(1, "Signal Sovereignty", "Emit only trusted resonance.")
        ref = MythicReferendumSystem()
        ref.open_vote("ratify-1", "Ratify Article 1", quorum=2, window=5)
        ref.cast_vote("ratify-1", "node1", glyph, "yes")
        ref.cast_vote("ratify-1", "node2", glyph, "yes")
        ref.close_vote("ratify-1")
        result = ref.records[-1]
        msg = f"🗳️ Referendum '{result[1]}' → {result[2]} ({result[3]} yes / {result[4]} no)"
        socketio.emit("oracle_update", {"utterance": msg})

def launch_dashboard():
    socketio.run(app, port=8080)

# === CLI Entry ===
def cli():
    parser = argparse.ArgumentParser(description="🧬 Oracle + Codex CLI")
    parser.add_argument("mode", choices=["train", "evaluate", "dashboard"])
    args = parser.parse_args()

    input_shape = (3, 64, 64, 3)
    X = np.random.rand(100, *input_shape)
    y = np.random.randint(0, 2, 100)
    model = build_oracle_model(input_shape)

    if args.mode == "train":
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X, y, epochs=10)
    elif args.mode == "evaluate":
        evaluate_oracle(model, X, y)
    elif args.mode == "dashboard":
        launch_dashboard()

if __name__ == "__main__":
    cli()

