import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
import json
import ast
# Download stopwords if you haven't already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean input text by converting to lower-case, removing punctuation,
    extra whitespace, and stop words.
    """
    if not isinstance(text, str):
        return text
    # Convert to lower-case
    text = text.lower()
    # Remove punctuation (you can modify the regex to preserve certain punctuation if needed)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stop words (optional)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def parse_and_clean_segments(segments_str):
    """
    1. Parse the Python literal (single-quoted list of dicts) using ast.literal_eval.
    2. Clean each segment's 'text'.
    3. Return as a proper JSON string (with double quotes).
    """
    if not isinstance(segments_str, str):
        return segments_str  # Already a list or invalid
    try:
        # Parse the string as a Python object
        segments = ast.literal_eval(segments_str)
    except Exception as e:
        print(f"Error parsing segments with ast.literal_eval: {e}")
        return segments_str  # Return unmodified if parsing fails

    # Clean each segment's 'text'
    for seg in segments:
        if 'text' in seg:
            seg['text'] = clean_text(seg['text'])

    # Convert back to a JSON string for consistent storage
    return json.dumps(segments)

# --- Main Script ---

# Step 1: Load the master dataset
master_file = '../../data/final/master_dataset_clean.csv'  # Replace with your filename if different
df = pd.read_csv(master_file)

# Step 2: Identify textual columns to clean.
# For this example, we'll clean:
# - 'jd_text'
# - 'overall_scores.explanation'
# - 'Interviewer_Comments'
# You can add more columns as needed.

text_columns = ['jd_text', 'overall_scores.explanation', 'Interviewer_Comments']

for col in text_columns:
    if col in df.columns:
        print(f"Cleaning column: {col}")
        df[col] = df[col].apply(clean_text)

# Step 3: Clean the 'segments' field.
if 'segments' in df.columns:
    print("Cleaning 'segments' field...")
    df['segments'] = df['segments'].apply(parse_and_clean_segments)

# (Optional) If you want to clean any other textual fields, add them to text_columns or process similarly.

# Step 4: Save the cleaned master dataset
cleaned_file = '../../data/final/final_dataset.csv'
df.to_csv(cleaned_file, index=False)
print(f"Cleaned master dataset saved to '{cleaned_file}'")
