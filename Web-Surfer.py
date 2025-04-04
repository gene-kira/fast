import os
import random
import string
import subprocess
import threading
from datetime import datetime, timedelta

# Check and install required packages
required_packages = [
    'flask', 'flask-wtf', 'flask-login', 'sqlalchemy', 'flask-talisman',
    'werkzeug', 'requests', 'beautifulsoup4', 'sentence-transformers'
]

installed_packages = {pkg.key for pkg in subprocess.run(
    [sys.executable, '-m', 'pip', 'list'], stdout=subprocess.PIPE
).stdout.decode().split('\n') if pkg}

missing_packages = set(required_packages) - installed_packages

if missing_packages:
    print(f"Missing packages: {missing_packages}. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + list(missing_packages))

# Import libraries
from flask import Flask, request, render_template_string, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from flask_wtf.csrf import CSRFProtect, CSRFTokenField
from wtforms import Form, StringField, PasswordField, validators
import requests
from bs4 import BeautifulSoup
import imaplib
import email
from sentence_transformers import SentenceTransformer, util
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_talisman import Talisman
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL', 'sqlite:///site.db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
csrf = CSRFProtect(app)
talisman = Talisman(app)

# User model for database
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Create database tables
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Registration form
class RegistrationForm(Form):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Confirm Password')

# Login form
class LoginForm(Form):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    password = PasswordField('Password', [validators.DataRequired()])

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm(request.form)
    if request.method == 'POST' and form.validate():
        user = User(username=form.username.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template_string('''
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <div class="container mt-5">
            <h2>Register</h2>
            <form method="post">
                {{ form.hidden_tag() }}
                <div class="form-group">
                    {{ form.username.label }}<br>
                    {{ form.username(size=32) }}<br>
                    {% for error in form.username.errors %}
                        <span style="color: red;">[{{ error }}]</span>
                    {% endfor %}
                </div>
                <div class="form-group">
                    {{ form.password.label }}<br>
                    {{ form.password() }}<br>
                    {% for error in form.password.errors %}
                        <span style="color: red;">[{{ error }}]</span>
                    {% endfor %}
                </div>
                <div class="form-group">
                    {{ form.confirm.label }}<br>
                    {{ form.confirm() }}<br>
                    {% for error in form.confirm.errors %}
                        <span style="color: red;">[{{ error }}]</span>
                    {% endfor %}
                </div>
                <input type="submit" class="btn btn-primary" value="Register">
            </form>
        </div>
    ''', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm(request.form)
    if request.method == 'POST' and form.validate():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template_string('''
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <div class="container mt-5">
            <h2>Login</h2>
            <form method="post">
                {{ form.hidden_tag() }}
                <div class="form-group">
                    {{ form.username.label }}<br>
                    {{ form.username(size=32) }}<br>
                    {% for error in form.username.errors %}
                        <span style="color: red;">[{{ error }}]</span>
                    {% endfor %}
                </div>
                <div class="form-group">
                    {{ form.password.label }}<br>
                    {{ form.password() }}<br>
                    {% for error in form.password.errors %}
                        <span style="color: red;">[{{ error }}]</span>
                    {% endfor %}
                </div>
                <input type="submit" class="btn btn-primary" value="Login">
            </form>
        </div>
    ''', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template_string('''
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <div class="container mt-5">
            <h2>Welcome to the Web Surfing Assistant</h2>
            <a href="{{ url_for('scrape') }}" class="btn btn-primary">Scrape Website</a>
            <a href="{{ url_for('search') }}" class="btn btn-secondary">Search Documents</a>
            <a href="{{ url_for('email_protection') }}" class="btn btn-danger">Email Protection</a>
        </div>
    ''')

@app.route('/scrape', methods=['GET', 'POST'])
@login_required
def scrape():
    form = ScrapeForm(request.form)
    if request.method == 'POST' and form.validate():
        url = form.url.data
        if url in scraped_cache and scraped_cache[url]['timestamp'] + timedelta(hours=1) > datetime.utcnow():
            content = scraped_cache[url]['content']
        else:
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = [p.get_text() for p in soup.find_all('p')[:5]]
                content = '\n'.join(paragraphs)
                scraped_cache[url] = {'content': content, 'timestamp': datetime.utcnow()}
            except requests.RequestException as e:
                flash(f'Error fetching the URL: {e}', 'danger')
                return redirect(url_for('scrape'))
        return render_template_string('''
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
            <div class="container mt-5">
                <h2>Scraped Content</h2>
                <pre>{{ content }}</pre>
                <a href="{{ url_for('scrape') }}" class="btn btn-secondary">Back to Scrape</a>
            </div>
        ''', content=content)
    return render_template_string('''
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <div class="container mt-5">
            <h2>Scrape Website</h2>
            <form method="post">
                {{ form.hidden_tag() }}
                <div class="form-group">
                    {{ form.url.label }}<br>
                    {{ form.url(size=32) }}<br>
                    {% for error in form.url.errors %}
                        <span style="color: red;">[{{ error }}]</span>
                    {% endfor %}
                </div>
                <input type="submit" class="btn btn-primary" value="Scrape">
            </form>
        </div>
    ''', form=form)

class ScrapeForm(Form):
    url = StringField('URL', [validators.URL(), validators.Length(min=4, max=256)])

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    form = SearchForm(request.form)
    if request.method == 'POST' and form.validate():
        query = form.query.data.lower()
        results = []
        for doc in documents:
            if query in doc['content'].lower():
                results.append(doc)
        return render_template_string('''
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
            <div class="container mt-5">
                <h2>Search Results</h2>
                {% if results %}
                    {% for result in results %}
                        <div class="card mb-3">
                            <div class="card-body">
                                <h5 class="card-title">{{ result['title'] }}</h5>
                                <p class="card-text">{{ result['content'][:100] }}...</p>
                            </div>
                        </div>
                    {% endfor %}
                {% else %}
                    <p>No results found.</p>
                {% endif %}
                <a href="{{ url_for('search') }}" class="btn btn-secondary">Back to Search</a>
            </div>
        ''', results=results)
    return render_template_string('''
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <div class="container mt-5">
            <h2>Search Documents</h2>
            <form method="post">
                {{ form.hidden_tag() }}
                <div class="form-group">
                    {{ form.query.label }}<br>
                    {{ form.query(size=32) }}<br>
                    {% for error in form.query.errors %}
                        <span style="color: red;">[{{ error }}]</span>
                    {% endfor %}
                </div>
                <input type="submit" class="btn btn-primary" value="Search">
            </form>
        </div>
    ''', form=form)

class SearchForm(Form):
    query = StringField('Query', [validators.Length(min=1, max=256)])

@app.route('/email_protection')
@login_required
def email_protection():
    emails = fetch_emails()
    return render_template_string('''
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <div class="container mt-5">
            <h2>Email Protection</h2>
            {% if emails %}
                {% for email in emails %}
                    <div class="card mb-3">
                        <div class="card-body">
                            <h5 class="card-title">{{ email['subject'] }}</h5>
                            <p class="card-text">{{ email['content'][:100] }}...</p>
                            <a href="{{ url_for('mark_safe', email_id=email['id']) }}" class="btn btn-success">Mark as Safe</a>
                        </div>
                    </div>
                {% endfor %}
            {% else %}
                <p>No emails found.</p>
            {% endif %}
        </div>
    ''', emails=emails)

@app.route('/mark_safe/<int:email_id>')
@login_required
def mark_safe(email_id):
    # Implement email marking logic here
    flash(f'Email {email_id} marked as safe.', 'success')
    return redirect(url_for('email_protection'))

# Document data for search functionality
documents = [
    {'id': 1, 'title': 'Document 1', 'content': 'This is the content of document one.'},
    {'id': 2, 'title': 'Document 2', 'content': 'This document contains important information.'}
]

# Email data for email protection functionality
email_cache = {
    1: {'id': 1, 'subject': 'Promotion', 'content': 'Congratulations! You have won a prize.', 'safe': False},
    2: {'id': 2, 'subject': 'Meeting Reminder', 'content': 'Reminder about tomorrow\'s meeting.', 'safe': True}
}

scraped_cache = {}

def fetch_emails():
    # This function should fetch emails from an actual email service
    # For demonstration purposes, we use a mock cache
    return [email for email in email_cache.values() if not email['safe']]

if __name__ == '__main__':
    app.run(debug=True)
