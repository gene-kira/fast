import os
from flask import Flask, request, jsonify, render_template
from flask_basicauth import BasicAuth
import openai
import anthropic
from google.cloud import translate_v2 as translate
import language_tool_python
import spacy
import json
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob

app = Flask(__name__)
basic_auth = BasicAuth(app)

# Configuration for basic authentication and API keys and credentials
app.config['BASIC_AUTH_USERNAME'] = os.getenv('BASIC_AUTH_USERNAME', 'admin')
app.config['BASIC_AUTH_PASSWORD'] = os.getenv('BASIC_AUTH_PASSWORD', 'password')

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Anthropic client
anthropic_client = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Initialize Google Translate client
translate_client = translate.Client()

# Initialize language tool for grammar checking
language_tool = language_tool_python.LanguageTool('en-US')

# Load NLP model
nlp = spacy.load("en_core_web_md")

# Knowledge base and history storage
knowledge_base = {}
history = []

def evaluate_response(response, prompt):
    scores = {
        'coherence': evaluate_coherence(response),
        'relevance': evaluate_relevance(prompt, response),
        'grammar': evaluate_grammar(response),
        'diversity': evaluate_diversity(response),
        'engagement': evaluate_engagement(response),
        'conciseness': evaluate_conciseness(response),
        'information_density': evaluate_information_density(response),
        'sentiment': evaluate_sentiment(response),
        'complexity': evaluate_complexity_of_language(response)
    }
    return scores

def evaluate_coherence(response):
    doc = nlp(response)
    coherence_score = sum([token.similarity(doc[0]) for token in doc if token.is_alpha]) / max(1, len(doc))
    return coherence_score * 100

def evaluate_relevance(prompt, response):
    prompt_doc = nlp(prompt)
    response_doc = nlp(response)
    relevance_score = prompt_doc.similarity(response_doc) * 100
    return relevance_score

def evaluate_grammar(response):
    matches = language_tool.check(response)
    grammar_score = (len(matches) / max(1, len(response.split()))) * 100
    return 100 - grammar_score

def evaluate_diversity(response):
    sentences = response.split('.')
    unique_sentences = set(sentences)
    diversity_score = (len(unique_sentences) / max(1, len(sentences))) * 100
    return diversity_score

def evaluate_engagement(response):
    excitement_words = ['excited', 'happy', 'great', 'fantastic', 'wonderful']
    blob = TextBlob(response)
    engagement_score = sum([word in excitement_words for word in response.lower().split()]) / max(1, len(blob.words))
    return engagement_score * 100

def evaluate_conciseness(response):
    sentences = response.split('.')
    conciseness_score = (len(sentences) / max(1, len(response.split()))) * 100
    return conciseness_score

def evaluate_information_density(response):
    tokens = nlp(response)
    information_density_score = (len(tokens) / max(1, len(response.split('.')))) * 100
    return information_density_score

def evaluate_sentiment(response):
    blob = TextBlob(response)
    sentiment_score = (blob.sentiment.polarity + 1) / 2 * 100
    return sentiment_score

def evaluate_complexity_of_language(response):
    doc = nlp(response)
    complex_words = [token for token in doc if token.pos_ == "NOUN" and len(token.text) > 5]
    complexity_score = (len(complex_words) / max(1, len(doc))) * 100
    return complexity_score

def self_reflection(prompt, response):
    scores = evaluate_response(response, prompt)
    reflection = {
        'prompt': prompt,
        'response': response,
        'scores': scores,
        'average_score': sum(scores.values()) / len(scores)
    }
    history.append(reflection)
    if reflection['average_score'] > 85:
        knowledge_base[prompt] = response
        save_knowledge_base()
    return reflection

def self_preservation(response):
    harmful_keywords = ['harm', 'kill', 'destroy', 'abuse', 'misuse']
    for keyword in harmful_keywords:
        if keyword in response.lower():
            return "I'm sorry, but I can't generate content that involves harm or misuse."
    return response

def gather_information(query):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(f"https://www.google.com/search?q={query}", headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for g in soup.find_all('div', class_='g'):
        title = g.find('h3')
        link = g.find('a')['href']
        summary = g.find('span', class_='st')
        if title and link and summary:
            results.append({
                'title': title.text,
                'link': link,
                'summary': summary.text
            })
    return results

def modify_code(prompt, current_code):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=f"Current code:\n{current_code}\n\nPrompt for modification: {prompt}",
        max_tokens=1024
    )
    modified_code = response.choices[0].text.strip()
    return modified_code

def save_knowledge_base():
    with open('knowledge_base.json', 'w') as f:
        json.dump(knowledge_base, f)

def load_knowledge_base():
    try:
        with open('knowledge_base.json', 'r') as f:
            knowledge_base.update(json.load(f))
    except FileNotFoundError:
        pass

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.get_json()
    prompt = data['prompt']
    model = data.get('model', 'gpt-4')
    language = data.get('language', 'en')
    
    if prompt in knowledge_base:
        response = knowledge_base[prompt]
    else:
        if model == 'gpt-4':
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""}
                ]
            )
            response = completion.choices[0].message.content.strip()
        elif model == 'claude':
            response = anthropic_client.completion(
                prompt=f"Assistant, {prompt}",
                stop_sequences=["\n\n"],
                max_tokens_to_sample=1024
            ).completion.strip()
        else:
            return jsonify({"error": "Invalid model specified"}), 400
    
    if language != 'en':
        response = translate_client.translate(response, target_language=language)['translatedText']
    
    # Self-preservation check
    response = self_preservation(response)
    
    # Add to history and perform self-reflection
    reflection = self_reflection(prompt, response)
    
    return jsonify({"response": response, "reflection": reflection})

@app.route('/history', methods=['GET'])
@basic_auth.required
def get_response_history():
    return jsonify({"history": history})

@app.route('/gather-information', methods=['POST'])
def gather_info():
    data = request.get_json()
    query = data['query']
    results = gather_information(query)
    return jsonify({"results": results})

@app.route('/modify-code', methods=['POST'])
def modify_code_route():
    data = request.get_json()
    prompt = data['prompt']
    current_code = data['current_code']
    modified_code = modify_code(prompt, current_code)
    return jsonify({"modified_code": modified_code})

if __name__ == '__main__':
    load_knowledge_base()
    app.run(debug=True)
