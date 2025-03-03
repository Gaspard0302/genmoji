# process_emojis.py
import json
import requests
from PIL import Image
from io import BytesIO
import os

def add_white_background(image_data):
    """Add a white background to transparent PNG"""
    img = Image.open(BytesIO(image_data))
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, "WHITE")
        background.paste(img, mask=img)
        return background
    return img

def download_and_process_emojis(limit=None):
    """Download and process emoji images"""
    with open("emojis.json", "r") as f:
        emoji_data = json.load(f)
    
    if limit:
        emoji_data = emoji_data[:limit]
    
    total = len(emoji_data)
    print(f"Processing {total} emojis...")
    
    for i, emoji in enumerate(emoji_data):
        print(f"Processing {i+1}/{total}: {emoji['processed']}")
        
        # Download the emoji
        response = requests.get(emoji['link'])
        if response.status_code == 200:
            # Save raw version
            raw_path = os.path.join("raw_emojis", f"{emoji['name']}.png")
            with open(raw_path, "wb") as f:
                f.write(response.content)
            
            # Process and save white background version
            img_with_background = add_white_background(response.content)
            processed_path = os.path.join("processed_emojis", f"{emoji['name']}.png")
            img_with_background.save(processed_path, "PNG")
            
            # Create training pair
            img_number = i + 1
            img_path = os.path.join("training_data", f"img{img_number}.png")
            txt_path = os.path.join("training_data", f"img{img_number}.txt")
            
            # Save processed image to training folder
            img_with_background.save(img_path, "PNG")
            
            # Save text description
            with open(txt_path, "w") as f:
                f.write(f"emoji of {emoji['processed']}")
        else:
            print(f"Failed to download {emoji['processed']}: Status {response.status_code}")

if __name__ == "__main__":
    download_and_process_emojis()