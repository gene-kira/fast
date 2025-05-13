import os
import sys
import csv
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

# Auto-loader for necessary libraries
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
except ImportError as e:
    print(f"Missing library: {e}")
    sys.exit(1)

# Initialize the browser
def init_browser():
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    service = ChromeService(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

# Log actions to a CSV file
def log_action(action_type, url, details):
    log_file = "actions_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Action Type", "URL", "Details"])
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([action_type, url, details])

# Check for broken links
def check_broken_links(driver, url):
    driver.get(url)
    links = driver.find_elements(By.TAG_NAME, "a")
    for link in links:
        href = link.get_attribute("href")
        if href and not href.startswith("#"):
            try:
                response = requests.head(href, timeout=5)
                if response.status_code >= 400:
                    log_action("Broken Link", url, f"Link: {href}, Status Code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                log_action("Broken Link", url, f"Link: {href}, Error: {e}")

# Fix broken links (example: replace with a placeholder)
def fix_broken_links(driver, url):
    driver.get(url)
    links = driver.find_elements(By.TAG_NAME, "a")
    for link in links:
        href = link.get_attribute("href")
        if href and not href.startswith("#"):
            try:
                response = requests.head(href, timeout=5)
                if response.status_code >= 400:
                    new_href = f"{url}/placeholder"
                    driver.execute_script(f'arguments[0].setAttribute("href", "{new_href}");', link)
                    log_action("Fixed Link", url, f"Link: {href} replaced with {new_href}")
            except requests.exceptions.RequestException as e:
                log_action("Broken Link", url, f"Link: {href}, Error: {e}")

# Check for security vulnerabilities (example: simple check for XSS)
def check_security_vulnerabilities(driver, url):
    driver.get(url)
    inputs = driver.find_elements(By.TAG_NAME, "input")
    for input_field in inputs:
        try:
            input_field.send_keys("<script>alert('XSS')</script>")
            input_field.submit()
            if "<script>alert('XSS')</script>" in driver.page_source:
                log_action("Security Vulnerability", url, "Potential XSS vulnerability detected")
        except Exception as e:
            log_action("Security Check", url, f"Error: {e}")

# Main function
def main():
    url = "https://example.com"
    driver = init_browser()
    
    try:
        # Check for broken links
        check_broken_links(driver, url)
        
        # Fix broken links
        fix_broken_links(driver, url)
        
        # Check for security vulnerabilities
        check_security_vulnerabilities(driver, url)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
