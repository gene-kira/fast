import requests
from bs4 import BeautifulSoup
from collections import deque
import logging

def explore_links(initial_url, max_depth):
    # Initialize the queue and visited set
    queue = deque([(initial_url, 0)])
    visited = set([initial_url])
    
    while queue:
        url, depth = queue.popleft()
        
        if depth > max_depth:
            continue
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Process the current page (e.g., extract data or perform actions)
                process_page(soup)
                
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
    for paragraph in soup.find_all('p'):
        print(paragraph.get_text())
    
    # Additional processing logic can be added here, such as extracting specific data points or images.
