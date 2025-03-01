```python
import requests
import time

def fetch_data(url):
    """
    Fetch data from the given URL, handling various exceptions and providing detailed logs.

    Args:
        url (str): The web address to fetch data from.

    Returns:
        str: The plain text content of the webpage.

    Raises:
        requests.exceptions.RequestException: For general HTTP errors with more details.

    Logs:
        Informational messages about successful and failed attempts, including error details.
    """
    try:
        response = requests.get(url, timeout=10)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {str(e)}")
        raise

def fetch_data_async(urls, max_retries=3):
    """
    Fetch data from multiple URLs concurrently with retry logic.

    Args:
        urls (list): List of URLs to fetch data from.
        max_retries (int): Maximum number of retries for each URL. Defaults to 3.

    Returns:
        dict: A dictionary mapping URLs to their fetched content, with failed attempts logged.
    """
    results = {}
    for url in urls:
        for retry_num in range(max_retries):
            print(f"Retrying URL {url} attempt {retry_num + 1}/{max_retries}")
            try:
                response = requests.get(url, timeout=10)
                results[url] = response.text
                break
            except requests.exceptions.RequestException as e:
                if retry_num == max_retries - 1:
                    raise
                print(f"Failed to fetch {url} (Attempt {retry_num + 1}/{max_retries})")
                time.sleep(2 ** retry_num)  # Exponential backoff

    return results

# Example usage
urls = ["www.example.com", "another.example.com"]
data = fetch_data_async(urls)
print(data)
```