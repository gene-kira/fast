import requests
from urllib.parse import urlparse
import time
import json
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up custom headers to make it harder for websites to track you
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://example.com/',
    'Accept-Language': '*'
}

# Toggle for HTTPS enforcement
enforce_https = True

def toggle_https():
    global enforce_https
    enforce_https = not enforce_https
    print(f"HTTPS Enforcement: {'Enabled' if enforce_https else 'Disabled'}")

def is_malicious(url, api_key="your_api_key"):
    """Check if the URL points to a known malicious domain using VirusTotal API."""
    parsed = urlparse(url)
    params = {
        "apikey": api_key,
        "url": url
    }
    
    try:
        response = requests.post("https://www.virustotal.com/api/v3/urls/analyse", params=params, headers=headers)
        if response.status_code == 200:
            data = json.loads(response.text)
            if data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {}).get('malicious') or False:
                return True
    except Exception as e:
        print(f"Error checking VirusTotal: {e}")
    
    # Fallback to local malicious domains list
    malicious_domains = [
        'example-malware.com',
        'fake-phishing-site.net'
    ]
    for domain in malicious_domains:
        if parsed.hostname == domain:
            return True
    return False

def enforce_https(url):
    """Enforce HTTPS connections by replacing HTTP with HTTPS."""
    if url.startswith('http:') and enforce_https:
        return url.replace('http:', 'https:')
    return url

def block_trackers(response_text):
    """Block common trackers like Google Analytics or Facebook Pixel."""
    blocked_domains = [
        'google-analytics.com',
        'facebook.com'
    ]
    for domain in blocked_domains:
        if domain in response_text:
            print(f"Blocked tracker: {domain}")
            # Remove the tracker code from the page
            response_text = response_text.replace(domain, '')
    return response_text

def get_tracker_list():
    """Fetch a comprehensive list of trackers to block."""
    try:
        response = requests.get("https://easylist.to/api/v1/trackers", headers=headers)
        if response.status_code == 200:
            data = json.loads(response.text)
            return data.get('domains', [])
    except Exception as e:
        print(f"Error fetching tracker list: {e}")
        return []

def main():
    """Main function that checks and protects your browser."""
    global enforce_https
    print("Browser Protector")
    print("-----------------")
    
    while True:
        print("\n1. Visit a URL")
        print("2. Toggle HTTPS Enforcement")
        print("3. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            url = input("Enter the URL you want to visit: ")
            
            # Enforce HTTPS
            safe_url = enforce_https(url)
            print(f"Visiting: {safe_url}")
            
            try:
                session = requests.Session()
                response = session.get(safe_url, headers=headers, timeout=5)
                
                if is_malicious(response.url):
                    print("Warning! This URL is known to be malicious!")
                    continue
                
                # Execute JavaScript using Selenium
                options = Options()
                options.headless = True
                driver = webdriver.Firefox(options=options)
                driver.get(safe_url)
                html = driver.page_source
                driver.quit()
                
                # Block trackers
                blocked_domains = get_tracker_list()
                for domain in blocked_domains:
                    if domain in html:
                        print(f"Blocked tracker: {domain}")
                        html = html.replace(domain, '')
                
                # Print the response text
                print("\n--- Response Content ---\n")
                print(html[:500])  # Show only the first 500 characters
                
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
        elif choice == '2':
            toggle_https()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
