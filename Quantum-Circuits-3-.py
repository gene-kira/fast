import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliSumOp, StateFn, CircuitStateFn
from qiskit.algorithms.optimizers import COBYLA
from qiskit.aqua.algorithms import VQE
import re
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
import os

# Install necessary libraries
os.system("pip install qiskit scikit-learn keras numpy pandas")

# Define a function to preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text
    tokens = text.split()
    
    # Remove stop words (optional, depending on your dataset)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Load the dataset
def load_dataset(file_path):
    # Assuming the dataset is a CSV file with columns 'text' and 'label'
    df = pd.read_csv(file_path)
    
    # Preprocess the text data
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    return df

# Split the dataset into training and testing sets
def split_data(df, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Convert text data to numerical features using TF-IDF
def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=1024)  # Adjust max_features as needed
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    
    # Normalize the data to [0, 1]
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train_tfidf)
    X_test_normalized = scaler.transform(X_test_tfidf)
    
    return X_train_normalized, X_test_normalized

# Create a feature map and an ansatz for the quantum circuit
def create_quantum_circuit(num_qubits):
    feature_map = TwoLocal(num_qubits, ['ry', 'rz'], 'cz', reps=1)
    ansatz = TwoLocal(num_qubits, ['ry', 'rz'], 'cz', reps=2)
    
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)  # Apply Hadamard gates to create superposition
    
    return qc

# Define the VQE algorithm
def define_vqe(qc, num_qubits):
    from qiskit.utils import algorithm_globals
    
    # Initialize random seed for reproducibility
    algorithm_globals.random_seed = 42
    
    hamiltonian = PauliSumOp.from_list([("Z" * num_qubits, 1)])
    
    optimizer = COBYLA(maxiter=100)  # Adjust maxiter based on your computational resources and dataset size
    
    quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
    
    vqe = VQE(ansatz=qc, optimizer=optimizer, quantum_instance=quantum_instance)
    
    return vqe

# Function to encode input features into quantum states
def encode_features(qc, input_data):
    for i in range(len(input_data)):
        qc.ry(input_data[i], i)
    return qc

# Train the VQE model using the training data
def train_vqe(vqe, X_train_normalized, y_train):
    from qiskit.opflow import StateFn
    
    # Convert labels to binary encoding (0 or 1)
    y_train_binary = np.where(y_train == 1, 1, -1)  # Assuming 1 is the positive class
    
    def cost_function(params):
        total_cost = 0
        for i in range(X_train_normalized.shape[0]):
            qc = create_quantum_circuit(num_qubits)
            encoded_qc = encode_features(qc.copy(), X_train_normalized[i])
            expectation_value = StateFn(encoded_qc).eval(hamiltonian).real
            cost = (expectation_value - y_train_binary[i])**2
            total_cost += cost
        return total_cost / X_train_normalized.shape[0]
    
    vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian, initial_point=params)
    return vqe_result

# Evaluate the model using test data
def evaluate_vqe(vqe, X_test_normalized, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    predictions = []
    
    for i in range(X_test_normalized.shape[0]):
        qc = create_quantum_circuit(num_qubits)
        encoded_qc = encode_features(qc.copy(), X_test_normalized[i])
        expectation_value = StateFn(encoded_qc).eval(hamiltonian).real
        prediction = 1 if expectation_value > 0 else -1
        predictions.append(prediction)
    
    y_pred_binary = np.where(np.array(predictions) == 1, 1, 0)
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    
    return accuracy, precision, recall, f1

# Set up a Flask web application to serve the trained model
app = Flask(__name__)

@app.route('/detect-threat', methods=['POST'])
def detect_threat():
    data = request.json
    text = data.get('text', '')
    
    # Preprocess the input text
    clean_text = preprocess_text(text)
    X_tfidf = vectorizer.transform([clean_text]).toarray()
    X_normalized = scaler.transform(X_tfidf)
    
    # Predict using the trained VQE model
    qc = create_quantum_circuit(num_qubits)
    encoded_qc = encode_features(qc.copy(), X_normalized[0])
    expectation_value = StateFn(encoded_qc).eval(hamiltonian).real
    prediction = 1 if expectation_value > 0 else -1
    
    if prediction == 1:
        threat_level = "Phishing Email"
    elif prediction == -1:
        threat_level = "Malware Link"
    else:
        threat_level = "No Threat"

    return jsonify({'threat_level': threat_level})

if __name__ == '__main__':
    # Load the dataset
    file_path = 'path_to_your_dataset.csv'
    df = load_dataset(file_path)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Vectorize and normalize the data
    num_qubits = 10  # Adjust based on your dataset size
    X_train_normalized, X_test_normalized = vectorize_data(X_train, X_test)
    
    # Create the quantum circuit
    qc = create_quantum_circuit(num_qubits)
    
    # Define the VQE algorithm
    vqe = define_vqe(qc, num_qubits)
    initial_params = [0.1] * qc.num_parameters
    
    # Train the model
    vqe_result = train_vqe(vqe, X_train_normalized, y_train)
    
    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_vqe(vqe, X_test_normalized, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Set up and run the Flask application
    app.run(host='0.0.0.0', port=5000)
