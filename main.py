from dotenv import load_dotenv, find_dotenv
import requests
import os
import io
import streamlit as st

from PIL import Image
from datetime import datetime

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {"Authorization": "Bearer token"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def text2image(prompt: str):
    image_bytes = query({
        "inputs": prompt,
    })

    try:
        image = Image.open(io.BytesIO(image_bytes))
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}.jpg"
        image.save(filename)
        return filename
    except Exception as e:
        print("Error:", e)
        print("Image bytes:", image_bytes)
        return None

def main():
    st.set_page_config(page_title="Text2Image", page_icon="")
    st.title("Text-to-Image Generator")
    
    with st.form(key="my_form"):
        query = st.text_area(
            label="Enter prompt for the image..",
            help="Enter a prompt for the image here.",
            key="query",
            max_chars=50,
        )
        
        submit_button = st.form_submit_button(label="Submit")
 
    if query and submit_button:
        with st.spinner(text="Generating image in progress..."):
            img_file = text2image(prompt=query)
            
        st.subheader("Your Image")
        st.image(f"./{img_file}", caption=query)
                
if __name__ == "__main__":
    main()
