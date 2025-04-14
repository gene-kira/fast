import os
import subprocess
from flask import Flask, request, jsonify, redirect, url_for, render_template_string, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, validators
import requests
from bs4 import BeautifulSoup
from scrapy import Spider, Request
import logging
import datetime
from datadog_api_client import api_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Dynamically install and import required libraries
def install_and_import(module_name: str, package_name: str = None):
    try:
        return __import__(module_name)
    except ModuleNotFoundError as e:
        logger.info(f"Module {module_name} not found. Installing {package_name or module_name}.")
        subprocess.check_call(['pip', 'install', package_name or module_name])
        return __import__(module_name)

# Install and import required libraries
BeautifulSoup = install_and_import('bs4').BeautifulSoup
requests = install_and_import('requests')
FlaskForm = install_and_import('flask_wtf').FlaskForm
StringField = install_and_import('wtforms').StringField
PasswordField = install_and_import('wtforms').PasswordField
SubmitField = install_and_import('wtforms').SubmitField
validators = install_and_import('wtforms.validators')

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(25), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# User Loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Registration Form
class RegistrationForm(FlaskForm):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Confirm Password')
    submit = SubmitField('Register')

# Login Form
class LoginForm(FlaskForm):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    password = PasswordField('Password', [validators.DataRequired()])
    submit = SubmitField('Login')

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
    return render_template_string('''<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <div class="container mt-5">
            <h2>Register</h2>
            <form method="post">
                {{ form.hidden_tag() }}
                <div class="form-group">
                    {{ form.username.label }}<br>
                    {{ form.username(class_="form-control") }}<br>
                    {{ form.password.label }}<br>
                    {{ form.password(class_="form-control") }}<br>
                    {{ form.confirm.label }}<br>
                    {{ form.confirm(class_="form-control") }}
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
            flash('You have been logged in.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template_string('''<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <div class="container mt-5">
            <h2>Login</h2>
            <form method="post">
                {{ form.hidden_tag() }}
                <div class="form-group">
                    {{ form.username.label }}<br>
                    {{ form.username(class_="form-control") }}<br>
                    {{ form.password.label }}<br>
                    {{ form.password(class_="form-control") }}
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
    return render_template_string('''<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
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
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.get_text()
            return jsonify({'content': content})
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            flash('Failed to scrape the website', 'danger')
    return render_template_string('''<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <div class="container mt-5">
            <h2>Scrape Website</h2>
            <form method="post">
                {{ form.hidden_tag() }}
                <div class="form-group">
                    {{ form.url.label }}<br>
                    {{ form.url(class_="form-control") }}
                </div>
                <input type="submit" class="btn btn-primary" value="Scrape">
            </form>
        </div>
    ''', form=form)

# Define the Scrape Form
class ScrapeForm(FlaskForm):
    url = StringField('URL', [validators.URL()])
    submit = SubmitField('Scrape')

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    form = SearchForm(request.form)
    if request.method == 'POST' and form.validate():
        query = form.query.data
        try:
            results = perform_web_search(query)
            return jsonify({'results': results})
        except Exception as e:
            logger.error(f"Failed to search for {query}: {e}")
            flash('Failed to search the web', 'danger')
    return render_template_string('''<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <div class="container mt-5">
            <h2>Search Documents</h2>
            <form method="post">
                {{ form.hidden_tag() }}
                <div class="form-group">
                    {{ form.query.label }}<br>
                    {{ form.query(class_="form-control") }}
                </div>
                <input type="submit" class="btn btn-primary" value="Search">
            </form>
        </div>
    ''', form=form)

# Define the Search Form
class SearchForm(FlaskForm):
    query = StringField('Query', [validators.DataRequired()])
    submit = SubmitField('Search')

def perform_web_search(query):
    # Use Google Custom Search API
    api_key = 'your_google_api_key'
    cx = 'your_custom_search_id'
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}"
    response = requests.get(url)
    results = response.json().get('items', [])
    return [item['link'] for item in results]

# Datadog Monitoring
def setup_datadog():
    from datadog_api_client import Configuration, api_client

    configuration = Configuration()
    configuration.api_key['apiKeyAuth'] = 'your_datadog_api_key'
    configuration.api_key['appKeyAuth'] = 'your_datadog_app_key'

    with api_client.ApiClient(configuration) as api_client:
        from datadog_api_client.v1 import MetricsApi, Series

        metrics_api = MetricsApi(api_client)
        series = [
            Series.Series(
                metric="web_assistant_request",
                type="count",
                interval=60,
                points=[[datetime.datetime.now().timestamp(), 1]]
            )
        ]
        metrics_api.submit_metrics(body={"series": series})

@app.route('/monitor')
@login_required
def monitor():
    setup_datadog()
    return "Monitoring is set up."

if __name__ == '__main__':
    app.run(debug=True)
