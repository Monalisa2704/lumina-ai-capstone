import pandas as pd
import numpy as np
import os

def create_synthetic_dataset(output_filename="marketing_campaigns.csv"):
    print("Generating 5,000 rows of synthetic e-commerce data...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    n_rows = 5000

    # 1. Generate Base Features
    caption_word_count = np.random.randint(10, 150, n_rows)
    emoji_count = np.random.randint(0, 10, n_rows)
    hashtag_count = np.random.randint(0, 20, n_rows)
    sentiment_score = np.random.uniform(-0.5, 1.0, n_rows) # Skewed positive
    reading_level = np.random.randint(4, 12, n_rows)
    
    # Visual features from Editor Agent context
    image_brightness = np.random.randint(50, 255, n_rows)
    image_contrast = np.random.uniform(0.5, 2.0, n_rows)

    # 2. Build the DataFrame
    df = pd.DataFrame({
        'caption_word_count': caption_word_count,
        'emoji_count': emoji_count,
        'hashtag_count': hashtag_count,
        'sentiment_score': sentiment_score,
        'reading_level': reading_level,
        'image_brightness': image_brightness,
        'image_contrast': image_contrast
    })

    # 3. Create "Rules" for Engagement (This makes your ML model look smart!)
    # Rule A: Exactly 5 hashtags is the 2026 optimal limit (Boosts score)
    hashtag_boost = np.where(df['hashtag_count'] == 5, 0.2, -0.05 * np.abs(df['hashtag_count'] - 5))
    
    # Rule B: Short, punchy captions + high sentiment perform best
    text_boost = (df['sentiment_score'] * 0.3) - (df['caption_word_count'] / 500)
    
    # Rule C: Good visual contrast boosts clicks
    visual_boost = np.where((df['image_contrast'] > 1.2) & (df['image_brightness'] > 100), 0.15, 0)

    # Calculate Base Engagement Rate (Normal distribution around 3%)
    base_engagement = np.random.normal(0.03, 0.01, n_rows)
    
    # Apply boosts and add some random noise
    final_engagement = base_engagement + hashtag_boost + text_boost + visual_boost + np.random.normal(0, 0.005, n_rows)
    
    # Ensure engagement is between 0% and 15%
    df['engagement_rate'] = np.clip(final_engagement, 0.001, 0.15)
    
    # Monetization clicks scale roughly with engagement and visual contrast
    df['monetization_clicks'] = (df['engagement_rate'] * 10000 * df['image_contrast']).astype(int)

    # 4. Save the dataset
    df.to_csv(output_filename, index=False)
    print(f"✅ Success! Saved {n_rows} rows to {output_filename}")
    print("Features ready for XGBoost or Random Forest training.")

if __name__ == "__main__":
    # Ensure pandas and numpy are installed: pip install pandas numpy
    create_synthetic_dataset()