import os
import sys
import time
import json
from flask import Flask, request, render_template_string, jsonify
from kafka import KafkaProducer, KafkaConsumer
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import requests
import spacy
import networkx as nx
from transformers import pipeline
from elasticsearch import Elasticsearch
import logging
from logstash_formatter import LogstashFormatterV1

# Initialize Flask app
app = Flask(__name__)

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Set up Kafka producer and consumer
producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda x: json.dumps(x).encode('utf-8'))
consumer = KafkaConsumer('result_topic', bootstrap_servers='localhost:9092', auto_offset_reset='earliest', enable_auto_commit=True, group_id='my-group', value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Set up Elasticsearch for logging
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Configure logging with Logstash formatter
log_format = LogstashFormatterV1()
logger = logging.getLogger('app')
handler = logging.StreamHandler()
handler.setFormatter(log_format)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Load Hugging Face QA pipeline
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

# Google Custom Search API key and CX (Custom Search Engine ID)
GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY'
CUSTOM_SEARCH_ENGINE_ID = 'YOUR_CUSTOM_SEARCH_ENGINE_ID'

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Thinking and Reasoning Model</title>
    <style>
        body { font-family: Arial, sans-serif; }
        #result { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Ask a Question</h1>
    <form id="question-form">
        <input type="text" id="question-input" placeholder="Enter your question here..." required>
        <button type="submit">Ask</button>
    </form>
    <div id="result"></div>

    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#question-form').on('submit', function(event) {
                event.preventDefault();
                var question = $('#question-input').val().trim();

                if (question === '') {
                    alert("Please enter a valid question.");
                    return;
                }

                $.ajax({
                    url: '/ask',
                    type: 'POST',
                    data: { question: question },
                    success: function(response) {
                        $('#result').html("<p>" + response.message + "</p>");
                        checkResult(question);
                    },
                    error: function(xhr, status, error) {
                        $('#result').html("<p>Error: " + error + "</p>");
                    }
                });
            });

            function checkResult(question) {
                setTimeout(function() {
                    $.ajax({
                        url: '/result',
                        type: 'POST',
                        data: { question: question },
                        success: function(response) {
                            $('#result').html("<h2>Answer:</h2><p>" + response.answer + "</p>");
                            $('#result').append("<h3>Knowledge Graph:</h3><ul>");
                            response.knowledge_graph.forEach(function(edge) {
                                $('#result').append("<li>" + edge.join(" -> ") + "</li>");
                            });
                            $('#result').append("</ul>");
                        },
                        error: function(xhr, status, error) {
                            checkResult(question);
                        }
                    });
                }, 1000); // Check every second
            }
        });
    </script>
</body>
</html>
"""

# Route to render the web interface
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# Route to handle question submission
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
    producer.send('question_topic', {'question': question})
    logger.info(f"Question asked: {question}")
    return jsonify({'message': 'Processing your question...'})

# Route to check for results
@app.route('/result', methods=['POST'])
def get_result():
    question = request.form['question']
    for message in consumer:
        if message.value['question'] == question:
            logger.info(f"Result received: {message.value}")
            return jsonify(message.value)

# Function to fetch web content using Google Custom Search API
def fetch_web_content(question):
    service = build('customsearch', 'v1', developerKey=GOOGLE_API_KEY)
    res = service.cse().list(q=question, cx=CUSTOM_SEARCH_ENGINE_ID).execute()
    results = []
    for item in res['items']:
        url = item['link']
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = '\n'.join([para.get_text() for para in paragraphs])
            results.append(text)
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
    return results

# Function to build knowledge graph
def build_knowledge_graph(text):
    doc = nlp(text)
    G = nx.Graph()
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    for i in range(len(entities) - 1):
        G.add_edge(entities[i][0], entities[i + 1][0])
    return G

# Worker function to process questions
def worker():
    consumer = KafkaConsumer('question_topic', bootstrap_servers='localhost:9092', auto_offset_reset='earliest', enable_auto_commit=True, group_id='worker-group', value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    for message in consumer:
        question = message.value['question']
        logger.info(f"Processing question: {question}")
        try:
            content = fetch_web_content(question)
            aggregated_text = ' '.join(content)
            result = qa_pipeline(question=question, context=aggregated_text)
            knowledge_graph = build_knowledge_graph(aggregated_text)
            edges = list(knowledge_graph.edges())
            producer.send('result_topic', {'question': question, 'answer': result['answer'], 'knowledge_graph': edges})
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            producer.send('result_topic', {'question': question, 'answer': 'No answer found.', 'knowledge_graph': []})

# Run Flask app
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        worker()
    else:
        app.run(debug=True)
