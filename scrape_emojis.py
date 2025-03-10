# scrape_emojis.py
import requests
from bs4 import BeautifulSoup
import re
import json
import os

def create_directories():
    """Create necessary directories for the project"""
    os.makedirs("raw_emojis", exist_ok=True)
    os.makedirs("processed_emojis", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)

def fetch_emoji_list():
    """Fetch the list of emojis from Emojigraph"""
    url = 'https://emojigraph.org/apple/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    all_emoji_data = []
    
    # Process all category divs
    for div_num in range(7, 16):
        selector = f'#category__first > div > div > div.col-12.col-lg-8 > div:nth-child({div_num})'
        category_div = soup.select_one(selector)
        
        if category_div:
            # Process images in this div
            for img in category_div.find_all('img'):
                if 'src' in img.attrs:
                    path = img['src']
                    
                    # Extract emoji name
                    name_match = re.search(r'/([^/]+)_[^/]+\.png$', path)
                    if name_match:
                        name = name_match.group(1)
                        
                        # Process URL
                        processed_url = path.replace('/72/', '/')
                        full_url = f"https://emojigraph.org{processed_url}"
                        
                        # Create processed name
                        processed_name = name.replace('-', ' ')
                        
                        all_emoji_data.append({
                            'link': full_url,
                            'name': name,
                            'processed': processed_name,
                        })
    
    # Save the emoji data
    with open('emojis.json', 'w', encoding='utf-8') as f:
        json.dump(all_emoji_data, f, ensure_ascii=False, indent=2)
    
    print(f"Collected {len(all_emoji_data)} emojis")
    return all_emoji_data

if __name__ == '__main__':
    create_directories()
    fetch_emoji_list()