from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Path to your WebDriver executable
driver_path = "path/to/your/webdriver"

# Create Chrome options with enhanced security and privacy settings
options = Options()
options.add_argument("--enable-secure-after")
options.add_argument("--disable-extensions")
options.add_argument("--block-unsafe-security-features")
options.add_experimental_option("excludeSwitches", ["enable-logging"])

# Initialize the WebDriver with specific options
driver = webdriver.Chrome(options=options, executable_path=driver_path)

try:
    # Open the webpage
    driver.get("https://www.example.com/form")

    # Check if form fields are present
    first_name_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "firstName"))
    )
    last_name_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "lastName"))
    )
    email_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "email"))
    )

    # Get user inputs with placeholders
    first_name = input("Enter your first name: ").strip()
    last_name = input("Enter your last name: ").strip()
    email = input("Enter your email: ").strip()

    if not (first_name and last_name and email):
        raise ValueError("All fields are required")

    # Fill in the form
    first_name_field.send_keys(first_name)
    last_name_field.send_keys(last_name)
    email_field.send_keys(email)

    # Find submit button and ensure it is clickable
    try:
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.NAME, "submit"))
        )
        submit_button.click()
    except Exception as e:
        print(f"Submit button not found or not clickable: {str(e)}")
        raise

    # Verify form submission was successful
    if driver.current_url != "https://www.example.com/form":
        print("Form submitted successfully!")
    else:
        print("Form submission may have failed. Please check the page.")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise

finally:
    # Clean up
    try:
        driver.quit()
        print("Browser session ended.")
    except NameError:
        print("No active browser session found.")
