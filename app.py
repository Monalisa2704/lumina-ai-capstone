from flask import Flask, render_template, request, jsonify
import os
import uuid
import re
import joblib
from textblob import TextBlob
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rembg import remove
from PIL import Image, ImageStat

# --- 1. INITIALIZE ENVIRONMENT & AI CLIENT ---
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- 2. SETUP FLASK APP ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- 3. LOAD THE ML BRAIN (Predictive Model) ---
try:
    # Ensure you have run the training script first to generate this file!
    ml_model = joblib.load('engagement_model.pkl')
    print("✅ Predictive Model Loaded Successfully!")
except Exception as e:
    print("⚠️ Warning: engagement_model.pkl not found. Predictions will be skipped.")
    ml_model = None

# --- 4. HELPER FUNCTIONS ---
def calculate_image_metrics(img_path):
    """Calculates real brightness and contrast from the uploaded image."""
    try:
        with Image.open(img_path) as img:
            gray_img = img.convert('L')
            stat = ImageStat.Stat(gray_img)
            
            brightness = stat.mean[0] # Average pixel intensity (0-255)
            contrast = stat.stddev[0] # Standard deviation acts as contrast
            
            # Scale contrast to match the synthetic dataset range (~0.5 to 2.0)
            scaled_contrast = max(0.5, min(2.0, contrast / 50.0))
            
            return brightness, scaled_contrast
    except Exception as e:
        print(f"Error calculating visual metrics: {e}")
        return 120, 1.2 # Fallback defaults

def extract_features(text, brightness, contrast):
    """Translates raw caption text into numbers for the ML model."""
    word_count = len(text.split())
    hashtag_count = len(re.findall(r'#\w+', text))
    
    # Simple regex hack to count emojis (non-alphanumeric/punctuation characters)
    emoji_count = len(re.findall(r'[^\w\s,\.!\?#\'"-]', text)) 
    
    sentiment = TextBlob(text).sentiment.polarity
    reading_level = max(4, min(12, int(word_count / 10))) 
    
    # Must match the exact column order from the synthetic dataset
    return [word_count, emoji_count, hashtag_count, sentiment, reading_level, brightness, contrast]


# --- 5. UI ROUTES ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/marketer', methods=['GET'])
def marketer():
    image_url = request.args.get('image', '')
    return render_template('marketer.html', image_url=image_url)


# --- 6. API ENDPOINTS ---
@app.route('/api/remove-background', methods=['POST'])
def remove_background_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        print("\n--- NEW ASSET UPLOADED ---")
        unique_id = str(uuid.uuid4())
        
        # Save Raw Image
        raw_filename = f'{unique_id}_raw.jpg'
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
        file.save(raw_path)

        # Worker 1: Remove Background
        print("Worker 1 is cutting the background...")
        raw_photo = Image.open(raw_path)
        clean_sticker = remove(raw_photo)
        
        sticker_filename = f'{unique_id}_sticker.png'
        sticker_path = os.path.join(app.config['UPLOAD_FOLDER'], sticker_filename)
        clean_sticker.save(sticker_path)
        
        return jsonify({
            "success": True,
            "sticker_url": f"/{sticker_path}"
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-marketing', methods=['POST'])
def generate_marketing_api():
    data = request.json
    image_url = data.get('image_url')
    user_prompt = data.get('prompt')

    if not image_url or not user_prompt:
        return jsonify({"error": "Missing image or prompt"}), 400

    try:
        # Safely locate the image file
        filename = image_url.split('/')[-1]
        local_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo = Image.open(local_image_path)
        
        # 1. Dynamically extract visual features
        img_brightness, img_contrast = calculate_image_metrics(local_image_path)
        
        system_instruction = """
        You are the Marketing Manager Agent. You must generate 3 distinct caption options.
        Adhere strictly to the brand guidelines: 
        - Tone must be Gen-Z / Millennial.
        - Never use the word 'cheap', use 'affordable' or 'investment'.
        - Keep hashtags to exactly 5 relevant tags.
        Output format MUST be:
        OPTION 1: [Text]
        OPTION 2: [Text]
        OPTION 3: [Text]
        """

        print("Worker 3 is writing copy (using Google Search for trends)...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_prompt, photo],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[{"google_search": {}}]
            )
        )

        # 2. Score and Rank Options
        if ml_model:
            print("Evaluator Agent is scoring the generated options...")
            
            # Split the Gemini response by the "OPTION X:" headers
            raw_options = re.split(r'OPTION\s*[1-3]:', response.text)
            clean_options = [opt.strip() for opt in raw_options if opt.strip()]
            
            scored_options = []
            for opt in clean_options:
                # Pass text AND image metrics to the ML feature extractor
                features = extract_features(opt, brightness=img_brightness, contrast=img_contrast)
                predicted_score = ml_model.predict([features])[0]
                scored_options.append({'text': opt, 'score': predicted_score})
            
            # Sort the list so the highest predicted score is at the top
            scored_options.sort(key=lambda x: x['score'], reverse=True)
            
            # Reconstruct the string for the frontend UI
            final_reply = ""
            for i, opt in enumerate(scored_options):
                win_rate = round(opt['score'] * 100, 2)
                final_reply += f"OPTION {i+1}: [Predicted Engagement: {win_rate}% 📈]\n{opt['text']}\n\n"
        else:
            # Fallback if ML model is missing
            final_reply = response.text

        return jsonify({
            "success": True,
            "reply": final_reply
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)