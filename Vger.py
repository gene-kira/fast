import requests
from bs4 import BeautifulSoup
from transformers import pipeline, BertForSequenceClassification, AdamW, Trainer, TrainingArguments
import torch
from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Configuration
data_url = "https://example.com"  # Replace with the URL of the data source
db_uri = 'sqlite:///knowledge.db'  # SQLite database for storing collected data
model_name = 'bert-base-uncased'
feedback_file = 'user_feedback.csv'

# Initialize components
Base = declarative_base()
engine = create_engine(db_uri)
Session = sessionmaker(bind=engine)

class DataEntry(Base):
    __tablename__ = 'data_entries'
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    sentiment_score = Column(Float, default=None)
    classification_label = Column(String, default=None)

# Create the database tables
Base.metadata.create_all(engine)

def crawl_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            return [p.get_text() for p in paragraphs]
        else:
            print(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error while crawling data: {e}")
        return []

def save_data(data):
    session = Session()
    try:
        for text in data:
            entry = DataEntry(text=text)
            session.add(entry)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error while saving data: {e}")
    finally:
        session.close()

def train_model(data, model_name):
    classifier = pipeline('sentiment-analysis')
    tokenizer = classifier.tokenizer
    model = BertForSequenceClassification.from_pretrained(model_name)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01
    )

    def compute_metrics(p):
        pred, labels = p
        return {'accuracy': (pred == labels).mean()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

def load_data_for_training():
    session = Session()
    try:
        data_entries = session.query(DataEntry).all()
        texts = [entry.text for entry in data_entries]
        sentiments = [entry.sentiment_score for entry in data_entries if entry.sentiment_score is not None]
        labels = [1 if score > 0 else 0 for score in sentiments]

        df = pd.DataFrame({'text': texts, 'label': labels})
        return df
    except Exception as e:
        print(f"Error while loading data: {e}")
    finally:
        session.close()

def update_sentiment_scores():
    session = Session()
    try:
        entries = session.query(DataEntry).filter(DataEntry.sentiment_score == None).all()
        classifier = pipeline('sentiment-analysis')

        for entry in entries:
            result = classifier(entry.text)
            entry.sentiment_score = result[0]['score']
        
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error while updating sentiment scores: {e}")
    finally:
        session.close()

def main():
    # Crawl data from the web
    data = crawl_data(data_url)
    save_data(data)

    # Load existing data for training
    train_df = load_data_for_training()

    if not train_df.empty:
        # Train the model
        tokenizer = pipeline('sentiment-analysis').tokenizer
        train_model(train_df, model_name)

        # Update sentiment scores for new entries
        update_sentiment_scores()

if __name__ == "__main__":
    main()
