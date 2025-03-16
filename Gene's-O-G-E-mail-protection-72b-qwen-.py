import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention
from imbox import Imbox
import nltk
nltk.download('words')
import time
import psutil

# Constants
EMAIL_HOST = 'imap.example.com'
EMAIL_USER = 'your-email@example.com'
EMAIL_PASSWORD = 'your-password'
SPAM_FOLDER = 'spam'
SANDBOX_PATH = './sandbox'

# Function to fetch the user's contact list
def get_contact_list():
    # This is a placeholder function. In a real scenario, you would fetch the contact list from an email provider or a file.
    return {
        'contact1@example.com': 'Contact 1',
        'contact2@example.com': 'Contact 2'
    }

# Function to check if the sender is in the user's contact list
def verify_contact(sender_email, contact_list):
    return sender_email in contact_list

# Function to fetch emails from the inbox
def fetch_emails():
    with Imbox(EMAIL_HOST, username=EMAIL_USER, password=EMAIL_PASSWORD, ssl=True) as imbox:
        messages = imbox.messages(unread=True)
        return [msg for uid, msg in messages]

# Function to move an email to the spam folder
def move_to_spam(uid, imbox):
    imbox.move(uid, SPAM_FOLDER)

# Function to delete an email
def delete_email(uid, imbox):
    imbox.delete(uid)

# Function to extract features from an email
def extract_features(email):
    sender_email = email['from'][0]['email']
    subject = email['subject']
    body = email['body']['plain'][0] if 'plain' in email['body'] else ''
    attachments = email['attachments']
    
    # Extract text and URLs
    text_content = f"{subject} {body}"
    urls = []
    for part in email['body'].get('html', []):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(part, 'html.parser')
        for link in soup.find_all('a'):
            url = link.get('href')
            if url:
                urls.append(url)
    
    return {
        'sender_email': sender_email,
        'text_content': text_content,
        'urls': urls,
        'attachments': attachments
    }

# Function to check if the sender's name is correctly spelled
def verify_sender(sender_name):
    english_words = set(nltk.corpus.words.words())
    words_in_name = sender_name.split()
    for word in words_in_name:
        if word.lower() not in english_words:
            return False
    return True

# Function to pre-scan emails for malicious content
def pre_scan_email(features, contact_list):
    # Check if the sender is in the contact list
    if not verify_contact(features['sender_email'], contact_list):
        return True
    
    # Check for known malicious URLs
    known_malicious_urls = ['malicious-url1.com', 'malicious-url2.com']
    for url in features['urls']:
        if any(mal_url in url for mal_url in known_malicious_urls):
            return True
    
    # Verify sender's name
    if not verify_sender(features['sender_name']):
        return True
    
    # Check for unexpected patterns (temporal anomalies)
    if len(features['urls']) > 5 or len(features['attachments']) > 3:
        return True
    
    return False

# Function to set up a sandbox environment
def setup_sandbox():
    os.makedirs(SANDBOX_PATH, exist_ok=True)

# Function to open emails in a sandbox
def open_email_in_sandbox(features):
    # Ensure the sandbox is clean
    for file in os.listdir(SANDBOX_PATH):
        os.remove(os.path.join(SANDBOX_PATH, file))
    
    # Save email content and attachments to the sandbox
    with open(os.path.join(SANDBOX_PATH, 'email.txt'), 'w') as f:
        f.write(features['text_content'])
    
    for attachment in features['attachments']:
        with open(os.path.join(SANDBOX_PATH, attachment['filename']), 'wb') as f:
            f.write(attachment['content'])

# Function to train models with enhanced hyperparameter tuning
def train_models(X_train, y_train):
    # Random Forest
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5)
    rf_grid.fit(X_train, y_train)
    
    # SVM
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm_grid = GridSearchCV(SVC(), svm_params, cv=5)
    svm_grid.fit(X_train, y_train)
    
    # LSTM with Attention
    input_text = Input(shape=(X_train.shape[1], 1))
    lstm_output = LSTM(64, return_sequences=True)(input_text)
    attention_output = Attention()([lstm_output, lstm_output])
    dense_output = Dense(1, activation='sigmoid')(attention_output)
    
    lstm_model = Model(inputs=input_text, outputs=dense_output)
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    return rf_grid.best_estimator_, svm_grid.best_estimator_, lstm_model

# Function to load models
def load_models():
    rf_model = joblib.load('rf_model.pkl') if os.path.exists('rf_model.pkl') else None
    svm_model = joblib.load('svm_model.pkl') if os.path.exists('svm_model.pkl') else None
    lstm_model = load_lstm_model() if os.path.exists('lstm_model.h5') else None
    return rf_model, svm_model, lstm_model

# Function to save models
def save_models(rf_model, svm_model, lstm_model):
    joblib.dump(rf_model, 'rf_model.pkl')
    joblib.dump(svm_model, 'svm_model.pkl')
    lstm_model.save('lstm_model.h5')

# Function to load LSTM model
def load_lstm_model():
    from tensorflow.keras.models import load_model
    return load_model('lstm_model.h5')

# Function to predict using models
def predict_with_models(rf_model, svm_model, lstm_model, X):
    if rf_model is not None:
        rf_pred = rf_model.predict(X)
    else:
        rf_pred = np.zeros(len(X))
    
    if svm_model is not None:
        svm_pred = svm_model.predict(X)
    else:
        svm_pred = np.zeros(len(X))
    
    if lstm_model is not None:
        lstm_pred = (lstm_model.predict(X) > 0.5).astype(int).flatten()
    else:
        lstm_pred = np.zeros(len(X))
    
    return rf_pred, svm_pred, lstm_pred

# Function to handle email processing
def process_email(email, imbox):
    features = extract_features(email)
    
    if pre_scan_email(features, get_contact_list()):
        print(f"Email from {features['sender_email']} is suspicious. Deleting.")
        delete_email(email.message_id, imbox)
        return
    
    rf_model, svm_model, lstm_model = load_models()
    
    # Prepare data for prediction
    X = [features['text_content']]
    
    # Predict using models
    rf_pred, svm_pred, lstm_pred = predict_with_models(rf_model, svm_model, lstm_model, X)
    
    if any([rf_pred[0], svm_pred[0], lstm_pred[0]]):
        print(f"Email from {features['sender_email']} is identified as malicious. Deleting.")
        delete_email(email.message_id, imbox)
    else:
        open_email_in_sandbox(features)

# Function to handle email processing in a safe manner
def open_email_in_sandbox(features):
    try:
        setup_sandbox()
        open_email_in_sandbox(features)
        print(f"Email from {features['sender_email']} opened safely.")
    except Exception as e:
        print(f"Error opening email: {e}")
        delete_email(email.message_id, imbox)

# Main function to process emails
def main():
    while True:
        try:
            with Imbox(EMAIL_HOST, username=EMAIL_USER, password=EMAIL_PASSWORD, ssl=True) as imbox:
                emails = fetch_emails()
                for email in emails:
                    process_email(email, imbox)
        except Exception as e:
            print(f"Error processing emails: {e}")
        
        time.sleep(60)  # Check for new emails every minute

if __name__ == "__main__":
    main()
