import requests
from bs4 import BeautifulSoup
from collections import deque
import logging
import csv
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import smtplib
from email.mime.text import MIMEText

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def explore_links(initial_urls, max_depth, data_file):
    # Initialize the queue and visited set
    queue = deque([(url, 0) for url in initial_urls])
    visited = set(initial_urls)
    
    with open(data_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['url', 'name', 'price', 'rating', 'availability'])  # Define the CSV columns
        
        while queue:
            url, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Process the current page
                    products = process_page(soup)
                    for product in products:
                        writer.writerow([url, product['name'], product['price'], product['rating'], product['availability']])
                    
                    # Find all links on the current page and handle pagination
                    next_urls = handle_pagination(soup, url)
                    for next_url in next_urls:
                        if next_url not in visited:
                            queue.append((next_url, depth + 1))
                            visited.add(next_url)
                else:
                    logging.warning(f"Failed to fetch {url} with status code {response.status_code}")
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")

def process_page(soup):
    # Extract all product elements
    products = []
    for item in soup.find_all('div', class_='product'):
        name = item.find('h2').get_text() if item.find('h2') else 'N/A'
        price = item.find('span', class_='price').get_text() if item.find('span', class_='price') else 'N/A'
        rating = item.find('span', class_='rating').get_text() if item.find('span', class_='rating') else 'N/A'
        availability = item.find('span', class_='availability').get_text() if item.find('span', class_='availability') else 'N/A'
        products.append({
            'name': name,
            'price': price,
            'rating': rating,
            'availability': availability
        })
    return products

def handle_pagination(soup, base_url):
    # Find next page links
    next_page_links = [link['href'] for link in soup.find_all('a', href=True) if 'next' in link.text.lower()]
    return [f"{base_url.rstrip('/')}/{next_link.lstrip('/')}" for next_link in next_page_links]

def send_email(to_email, subject, message):
    from_email = "your-email@example.com"
    password = "your-password"

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())

def notify_users(data_file):
    # Load the collected data
    new_data = pd.read_csv(data_file)
    
    # Filter out new or better deals (for simplicity, we assume 'better' means a lower price)
    current_best_deals = load_current_best_deals()  # Implement this function to load your existing best deals
    new_best_deals = []

    for _, row in new_data.iterrows():
        if is_better_deal(row, current_best_deals):
            new_best_deals.append(row)

    # Send notifications for the new or better deals
    for deal in new_best_deals:
        subject = f"New Deal Found: {deal['name']}"
        message = f"A new deal has been found for {deal['name']}:\nPrice: {deal['price']}\nRating: {deal['rating']}\nAvailability: {deal['availability']}"
        send_email("user-email@example.com", subject, message)

def is_better_deal(new_deal, current_best_deals):
    if new_deal['name'] not in current_best_deals:
        return True
    else:
        current_price = float(current_best_deals[new_deal['name']]['price'].replace('$', ''))
        new_price = float(new_deal['price'].replace('$', ''))
        return new_price < current_price

def load_current_best_deals():
    # Implement this to load your existing best deals from a file or database
    pass  # Placeholder for loading current best deals

def schedule_exploration(initial_urls, max_depth, data_file):
    scheduler = BackgroundScheduler()
    
    def explore_and_notify():
        explore_links(initial_urls, max_depth, data_file)
        notify_users(data_file)
    
    # Schedule the exploration and notification to run every 24 hours
    scheduler.add_job(explore_and_notify, 'interval', hours=24)
    scheduler.start()

# Define initial URLs and maximum depth for exploration
initial_urls = ["https://example.com", "https://another-deal-site.com"]
max_depth = 3
data_file = "collected_data.csv"

# Schedule the bot to run every day at midnight
schedule_exploration(initial_urls, max_depth, data_file)
