import requests
from bs4 import BeautifulSoup
from collections import deque
import logging
import csv

def explore_links(initial_url, max_depth, data_file):
    # Initialize the queue and visited set
    queue = deque([(initial_url, 0)])
    visited = set([initial_url])
    
    with open(data_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['url', 'content'])  # Define the CSV columns
        
        while queue:
            url, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Process the current page
                    content = process_page(soup)
                    writer.writerow([url, content])
                    
                    # Find all links on the current page
                    for link in soup.find_all('a', href=True):
                        next_url = link['href']
                        
                        # Ensure the URL is absolute
                        if not next_url.startswith('http'):
                            next_url = f"{url.rstrip('/')}/{next_url.lstrip('/')}"
                        
                        if next_url not in visited:
                            queue.append((next_url, depth + 1))
                            visited.add(next_url)
                else:
                    logging.warning(f"Failed to fetch {url} with status code {response.status_code}")
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")

def process_page(soup):
    # Example: Extract all text content from the page
    paragraphs = soup.find_all('p')
    return ' '.join([paragraph.get_text() for paragraph in paragraphs])
