import requests
from pystray import Icon, Menu, MenuItem
import time
import json

def get_country():
    try:
        response = requests.get('http://ip-api.com/json')
        if response.status_code == 200:
            data = response.json()
            return data['country_name']
    except Exception as e:
        print(f"Error: {e}")
        return None

def create_icon(icon_path):
    icon = Icon("World Server Status")
    menu = Menu()
    item = MenuItem()
    icon.icon = icon_path
    return icon

def change_icon(icon, icon_path):
    icon.icon = icon_path

if __name__ == "__main__":
    home_icon = "green.ico"  # Replace with your green icon path
    away_icon = "red.ico"    # Replace with your red icon path
    
    icon = create_icon(home_icon)
    
    while True:
        country = get_country()
        if country == 'United States':
            change_icon(icon, home_icon)
        else:
            change_icon(icon, away_icon)
        
        time.sleep(60)  # Update every minute
