# --- Worker 1 & Worker 2 Assembly Line ---
from rembg import remove
from PIL import Image
import google.generativeai as genai

# Tell the script how to access Gemini
# IMPORTANT: Put your real Gemini API key inside the quotes below!
genai.configure(api_key="AQ.Ab8RN6IOXyzyBKGXsWIxLfiRpH89qIpHg_0zy7XAurrx4ayQ_Q")

def process_image(input_image_path, output_image_path):
    print("Worker 1 (The Cutter) is removing the background...")
    raw_photo = Image.open(input_image_path)
    clean_sticker = remove(raw_photo)
    clean_sticker.save(output_image_path)
    print(f"-> Worker 1 saved: {output_image_path}\n")
    return output_image_path

def inspect_image(image_path):
    print("Worker 2 (The Inspector) is reviewing the final sticker...")
    
    # 1. Open the new sticker image
    sticker_image = Image.open(image_path)
    
    # 2. Wake up the Gemini AI model
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # 3. Give the Inspector its strict guardrail instructions
    prompt = """
    You are a strict Visual Quality Inspector for an e-commerce studio.
    Look at this product sticker image. 
    Verify two things:
    1. Is the background completely removed?
    2. Does the product look properly oriented (not rotated upside down or distorted)?
    Respond with exactly "PASS" or "FAIL", followed by a short 1-sentence explanation.
    """
    
    # 4. Ask Gemini to look at the prompt AND the image together
    response = model.generate_content([prompt, sticker_image])
    print(f"Inspector's Verdict: {response.text}")

# --- Start the Factory Line! ---
# Step 1: Cut the background
final_image = process_image('raw_lamp.jpg', 'perfect_sticker.png')

# Step 2: Inspect the result
inspect_image(final_image)