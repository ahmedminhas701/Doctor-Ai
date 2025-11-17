import os
from pathlib import Path
import pandas as pd
import torch
import spacy
from sentence_transformers import SentenceTransformer, util
import nltk
import numpy as np
import joblib
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from typing import List, Tuple, Optional, Any
import sys
import re
import random
from collections import defaultdict
from deep_translator import GoogleTranslator
import random
import seaborn as sns
import re
import matplotlib.pyplot as plt
from gemini_api import ask_gemini
import subprocess

os.makedirs("./static", exist_ok=True)
os.makedirs("./reports", exist_ok=True)
os.makedirs("./model", exist_ok=True)
os.makedirs("./data", exist_ok=True)

def extract_single_age(text):
    numbers = re.findall(r'\d+\.?\d*', text)
    return numbers[0] if numbers else None


def is_valid_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_gender(text):
    """
    Extracts and returns 'male' or 'female' from input text.
    Returns None if not valid.
    """
    text = text.strip().lower().replace(".", "").replace(",", "")

    if "male" in text:
        return "male"
    elif "female" in text:
        return "female"
    return None


def load_dataframe(path: Path, expected_columns: list) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {missing}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")


base_path = Path('./data')
model_path = Path('./model')

try:
    df_category_food = load_dataframe(
        base_path / 'category-foods.csv',
        expected_columns=['Category', 'Food', 'Calories', 'Sugar (g)',
                          'Carbs (g)', 'Fat (g)', 'Sodium (mg)', 'Key Nutrients']
    )

    df_category_exercise = load_dataframe(
        base_path / 'category-exercises.csv',
        expected_columns=['Category', 'Exercise', 'Calories Burned (per 30 min)',
                          'Intensity', 'Key Benefits']
    )

    df_category_meds = load_dataframe(
        base_path / 'category-medicine.csv',
        expected_columns=['Category', 'Medicine', 'Adults', 'Children', 'Notes']
    )

    df_category_disease = load_dataframe(
        base_path / 'disease-category.csv',
        expected_columns=['Disease', 'Category']
    )
    df_category_description = load_dataframe(
        base_path / 'category-description.csv',
        expected_columns= ["Category","Description"]
    )


    def category_description(category: str) -> str:
        desc = df_category_description[df_category_description['Category'] == category]
        return desc['Description'].values[0] if not desc.empty else "No description available."


    with open(model_path / "symptom_columns.json") as f:
        symptom_columns = json.load(f)
        if not isinstance(symptom_columns, list):
            raise ValueError("symptom_columns.json should contain a list")

    pipeline = joblib.load(model_path / 'saved_model1.pkl')

    with open(model_path / "class_dict.json") as f:
        class_dict = json.load(f)
        if not isinstance(class_dict, dict):
            raise ValueError("class_dict.json should contain a dictionary")

except Exception as e:
    print(f"Error during initialization: {str(e)}")
    sys.exit(1)


def category_find(disease: str) -> List[str]:
    categories = df_category_disease.loc[df_category_disease.Disease == disease, 'Category']
    return categories.tolist()


def category_med(category: str) -> pd.DataFrame:
    return df_category_meds[df_category_meds.Category == category]


def category_exercise(category: str) -> pd.DataFrame:
    return df_category_exercise[df_category_exercise.Category == category]


def category_food(category: str) -> pd.DataFrame:
    return df_category_food[df_category_food.Category == category]


def category_meds(category: str) -> List:
    return category_med(category)


def plot_category_distribution(user_category: str, save_path: str = None):
    """
    Plot disease category distribution and highlight a given category.

    Args:
        user_category (str): The category to highlight.
        save_path (str, optional): If provided, saves PNG instead of showing.

    Returns:
        str | None: Full path of saved PNG if saved, else None.
    """
    # Count categories from your CSV-backed dataframe
    category_counts = df_category_disease['Category'].value_counts()

    # Handle case where user_category is not present
    if user_category not in category_counts.index:
        print(f"⚠️ Warning: '{user_category}' not found in category list.")

    plt.figure(figsize=(12, 6))

    # Highlight user’s category in pink, others grey
    palette = ["#444444" if cat != user_category else "#ff4da6" for cat in category_counts.index]

    sns.barplot(
        x=category_counts.index,
        y=category_counts.values,
        palette=palette
    )

    # Labels and title
    plt.xlabel("Disease Category", fontsize=13, weight="semibold", color="white")
    plt.ylabel("Count", fontsize=13, weight="semibold", color="white")
    plt.title("Disease Category Distribution", fontsize=18, weight="bold", pad=20, color="#ff4da6")

    # Rotate x-labels
    plt.xticks(rotation=45, ha='right', fontsize=11, color="white")
    plt.yticks(fontsize=11, color="white")

    # Add value labels above bars
    for i, v in enumerate(category_counts.values):
        plt.text(i, v + 0.5, str(v), ha='center', fontsize=10, color="white")

    # Black background
    plt.gca().set_facecolor("black")
    plt.gcf().set_facecolor("black")

    # Remove borders
    sns.despine()
    plt.tight_layout()

    if save_path:
        save_path = str(Path(save_path).resolve())  # ensure full path
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()
        return save_path
    else:
        # plt.show()
        return None




def plot_exercise_recommendations(category, save_path=None):
    """
    Generate a barplot of recommended exercises for a given disease category.

    Parameters:
        category (str): Disease category (e.g., "Cardiovascular")
        save_path (str, optional): If provided, saves the figure as PNG at this path.
                                   If None, just shows the plot.
    Returns:
        str or None: Path to saved image if save_path is provided, else None.
    """
    # Load exercise CSV
    df = pd.read_csv("./static/category-exercises.csv")

    # Convert calorie range → midpoint
    df["Calories Burned (per 30 min)"] = df["Calories Burned (per 30 min)"].str.split("-").apply(
        lambda x: (int(x[0]) + int(x[1])) / 2
    )

    # Filter by category
    df_user = df[df["Category"] == category]

    if df_user.empty:
        print(f"⚠️ Warning: No exercises found for category '{category}'")
        return None

    # ---- Barplot ----
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_user,
        x="Exercise",
        y="Calories Burned (per 30 min)",
        hue="Intensity",
        palette="rocket"
    )

    plt.title(f"Recommended Exercises for {category} Diseases", fontsize=16, weight="bold", color="#ff4da6")
    plt.xlabel("Exercise", fontsize=12, weight="semibold", color="white")
    plt.ylabel("Calories Burned (per 30 min)", fontsize=12, weight="semibold", color="white")
    plt.xticks(rotation=30, ha="right", color="white")
    plt.yticks(color="white")

    plt.gca().set_facecolor("black")
    plt.gcf().set_facecolor("black")
    plt.legend(title="Intensity", labelcolor="white", facecolor="black")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="black")
        # plt.show()
        plt.close()
        return save_path
    else:
        # plt.show()
        return None




def plot_medicine_recommendations(category, save_path=None):
    """
    Plot medicine recommendations by age group for a given disease category.

    Args:
        category (str): The disease category (e.g., "Cardiovascular").
        save_path (str, optional): Path to save the plot. If None, only displays.

    Returns:
        str: Path of the saved plot if save_path is provided, else None.
    """
    # Filter for the given category
    df_user = df_category_meds[df_category_meds["Category"] == category]

    if df_user.empty:
        print(f"No medicine data found for category '{category}'.")
        return None

    # Melt to long format
    df_melted = df_user.melt(
        id_vars=["Medicine"],
        value_vars=["Adults", "Children"],
        var_name="Age Group",
        value_name="Recommended"
    )

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_melted,
        x="Medicine",
        y="Recommended",
        hue="Age Group",
        palette=["#ff4da6", "#4da6ff"]
    )

    plt.title(f"Medicines for {category}", fontsize=15, weight="bold", color="#ff4da6")
    plt.xlabel("Medicine", fontsize=12, weight="semibold", color="white")
    plt.ylabel("Recommended (1=Yes, 0=No)", fontsize=12, weight="semibold", color="white")
    plt.xticks(rotation=25, ha="right", color="white")
    plt.yticks(color="white")
    plt.legend(title="Age Group", labelcolor="white", facecolor="black")

    plt.gca().set_facecolor("black")
    plt.gcf().set_facecolor("black")
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, facecolor="black")
        # plt.show()
        plt.close()
        return save_path
    else:
        # plt.show()
        return None



def install_spacy_model():
    """Install spaCy model if not available"""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

def load_sentence_transformer():
    """Load SentenceTransformer model"""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception as e:
        print(f"Error loading sentence transformer: {e}")
        # The model will be automatically downloaded on first use
        return SentenceTransformer("all-MiniLM-L6-v2")

# Usage in your main code
try:
    nlp = install_spacy_model()
    model = load_sentence_transformer()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")


def get_user_input():
    while True:
        text = input("Input: ").strip()
        print(f"Input was -->  {text}")
        if text:
            return text
        print("⚠️ Please enter at least something.")


def translate_english(text, dest_lang='en', src_lang='auto'):
    translator = GoogleTranslator(source=src_lang, target=dest_lang)
    return translator.translate(text)


def translate(text, dest_lang, src_lang='auto'):
    translator = GoogleTranslator(source=src_lang, target=dest_lang)
    return translator.translate(text)


language_list = {"urdu", "english", "french", "arabic", "hindi",
                 "german", "spanish", "chinese", "japanese"}

language_codes = {
    "english": "en",
    "urdu": "ur",
    "roman urdu": "ur",
    "hindi": "hi",
    "french": "fr",
    "arabic": "ar",
    "german": "de",
    "spanish": "es",
    "chinese": "zh-cn",
    "japanese": "ja"
}


def get_language_code(language_name, default="en"):
    if not language_name:
        return default
    return language_codes.get(language_name.strip().lower(), default)


try:
    from langdetect import detect, DetectorFactory

    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

import re

# Your language configuration
language_list = {"urdu", "english", "french", "arabic", "hindi",
                 "german", "spanish", "chinese", "japanese"}

# Language-specific request patterns with proper Urdu, Hindi, and French patterns
LANGUAGE_REQUEST_PATTERNS = {
    "english": [
        "reply in english", "respond in english", "answer in english",
        "translate to english", "in english", "to english",
        "english mein", "english me"
    ],
    "urdu": [
        "reply in urdu", "respond in urdu", "answer in urdu",
        "translate to urdu", "in urdu", "to urdu",
        "urdu mein", "urdu me",
        "ourdou", "en ourdou", "réponse en ourdou", "réponds en ourdou",
        "اردو میں", "اردو میں جواب", "اردو میں بتائیں",
        "جواب اردو میں", "ترجمہ اردو میں"
    ],
    "hindi": [
        "reply in hindi", "respond in hindi", "answer in hindi",
        "translate to hindi", "in hindi", "to hindi",
        "hindi mein", "hindi me",
        "हिंदी में", "हिन्दी में", "हिंदी में जवाब", "हिंदी में बताएं",
        "जवाब हिंदी में", "अनुवाद हिंदी में"
    ],
    "french": [
        "reply in french","respond in french", "answer in french",
        "translate to french", "in french", "to french",
        "en français", "réponse en français", "réponds en français",
        "traduire en français", "french mein"
    ],
    "arabic": [
        "reply in arabic", "respond in arabic", "answer in arabic",
        "translate to arabic", "in arabic", "to arabic",
        "بالعربية", "في العربية", "arabic mein"
    ],
    "german": [
        "reply in german", "respond in german", "answer in german",
        "translate to german", "in german", "to german",
        "auf deutsch", "in deutsch", "german mein"
    ],
    "spanish": [
        "reply in spanish", "respond in spanish", "answer in spanish",
        "translate to spanish", "in spanish", "to spanish",
        "en español", "spanish mein"
    ],
    "chinese": [
        "reply in chinese", "respond in chinese", "answer in chinese",
        "translate to chinese", "in chinese", "to chinese",
        "用中文", "chinese mein"
    ],
    "japanese": [
        "reply in japanese", "respond in japanese", "answer in japanese",
        "translate to japanese", "in japanese", "to japanese",
        "日本語で", "japanese mein"
    ]
}


# --- helpers ---
def _script_heuristic_language(text):
    """Detect language by checking Unicode ranges for various scripts."""
    arabic_count = 0
    latin_count = 0
    devanagari_count = 0
    chinese_count = 0
    japanese_count = 0

    for ch in text:
        char_code = ord(ch)
        # Arabic script (Urdu, Arabic, Persian)
        if (0x0600 <= char_code <= 0x06FF or 0x0750 <= char_code <= 0x077F or
                0x08A0 <= char_code <= 0x08FF or 0xFB50 <= char_code <= 0xFDFF or
                0xFE70 <= char_code <= 0xFEFF):
            arabic_count += 1
        # Devanagari script (Hindi)
        elif 0x0900 <= char_code <= 0x097F:
            devanagari_count += 1
        # Chinese characters
        elif 0x4E00 <= char_code <= 0x9FFF:
            chinese_count += 1
        # Japanese Hiragana and Katakana
        elif (0x3040 <= char_code <= 0x309F) or (0x30A0 <= char_code <= 0x30FF):
            japanese_count += 1
        # Latin script (English, French, German, Spanish, etc.)
        elif ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'):
            latin_count += 1

    # Return based on script counts
    if arabic_count > max(latin_count, devanagari_count, chinese_count, japanese_count, 1):
        return "urdu"  # or could be arabic - would need more context
    elif devanagari_count > max(arabic_count, latin_count, chinese_count, japanese_count, 1):
        return "hindi"
    elif chinese_count > max(arabic_count, latin_count, devanagari_count, japanese_count, 1):
        return "chinese"
    elif japanese_count > max(arabic_count, latin_count, devanagari_count, chinese_count, 1):
        return "japanese"
    # For Latin script, we can't distinguish between English/French/German/Spanish
    # So return None and let langdetect handle it
    elif latin_count > 0:
        return None

    return None


def detect_and_clean_languages(text):
    if not text or not text.strip():
        return "", "english"

    text = text.strip()
    original_text = text
    print(f"DEBUG: Original input: '{original_text}'")

    # --- PRIORITY 1: Check for explicit language requests ---
    user_lang = None
    patterns_found = []

    for lang, patterns in LANGUAGE_REQUEST_PATTERNS.items():
        for pattern in patterns:
            try:
                if re.search(pattern, original_text, flags=re.IGNORECASE | re.UNICODE):
                    user_lang = lang
                    patterns_found.append(pattern)
                    print(f"DEBUG: Found language request pattern '{pattern}' -> {lang}")
                    break
            except:
                continue
        if user_lang:
            break

    # Clean the text - remove language request patterns if found
    cleaned_text = original_text
    if user_lang:
        print(f"DEBUG: User requested language: {user_lang}")
        # Remove all found patterns
        for lang, patterns in LANGUAGE_REQUEST_PATTERNS.items():
            for pattern in patterns:
                try:
                    cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE | re.UNICODE)
                except:
                    continue

        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        if cleaned_text and cleaned_text[-1] not in ".!?":
            cleaned_text += "."

        print(f"DEBUG: Cleaned text: '{cleaned_text}'")
        print(f"DEBUG: Returning user requested language: {user_lang}")
        return cleaned_text, user_lang

    # --- PRIORITY 2: No explicit request, detect the language of the text itself ---

    # First try script heuristic (good for non-Latin scripts)
    heuristic_lang = _script_heuristic_language(original_text)
    if heuristic_lang:
        print(f"DEBUG: Script heuristic detected: {heuristic_lang}")
        return original_text, heuristic_lang

    # For Latin-script languages, use langdetect
    detected_lang = None
    if _HAS_LANGDETECT:
        try:
            code = detect(original_text)
            langdetect_to_our_lang = {
                'en': 'english',
                'ur': 'urdu',
                'ar': 'arabic',
                'hi': 'hindi',
                'fr': 'french',
                'de': 'german',
                'es': 'spanish',
                'zh-cn': 'chinese',
                'zh-tw': 'chinese',
                'ja': 'japanese'
            }
            detected_lang = langdetect_to_our_lang.get(code)
            print(f"DEBUG: Langdetect detected: {code} -> {detected_lang}")

            if detected_lang:
                return original_text, detected_lang
        except Exception as e:
            print(f"DEBUG: Langdetect error: {e}")

    # Default fallback
    print(f"DEBUG: Falling back to default: english")
    return original_text, "english"


# Test cases
test_cases = [
    "J'ai des nausées, je veux une réponse en ourdou",  # French text, Urdu requested
    "J'ai des nausées",  # Just French
    "I feel sick, reply in urdu",  # English text, Urdu requested
    "मुझे बुखार है",  # Hindi
    "میں بیمار ہوں",  # Urdu
    "I am sick",  # English
    "Estoy enfermo",  # Spanish
    "Ich bin krank",  # German
]

    # for test in test_cases:
    #     cleaned, lang = detect_and_clean_languages(test)
    #     print(f"\nInput: '{test}'")
    #     print(f"Cleaned: '{cleaned}'")
    #     print(f"Language: {lang}")
    #     print("-" * 50)

gratitude_keywords = [
    "thank you", "thanks", "thx", "thank u", "many thanks", "thanks a lot",
    "thanks so much", "thank you very much", "really appreciate it", "appreciate it",
    "i appreciate that", "much appreciated", "i'm grateful", "grateful for that",
    "tysm", "tyvm", "cheers", "thank you kindly", "thank you for your help",
    "thanks buddy", "thanks man", "thanks dear"
]
service_keywords = [
    "can you make me a pdf", "make a pdf report", "generate my report", "give me my report",
    "can you give me a weekly diet plan", "weekly diet plan", "make a diet plan",
    "create a weekly diet", "can you suggest a diet", "custom diet plan", 'what food should I eat',
    "can you give me a weekly exercise plan", "weekly exercise routine", "exercise schedule",
    "can you give me medicine info", "suggest me medicine", "medicine recommendation",
    "what meds should i take", "can you prescribe something", "medicine plan",
    "send me my report", "create my health summary", "download my summary",
    "give me personalized plan", "custom health plan", "can you recommend me diet",
    "can you recommend me good exercises",
    "can you recommend me food", "recommend me medicine"
]
greeting_keywords = [
    "hi", "hello", "hey", "hiya", "yo", "sup", "what's up",
    "good morning", "good afternoon", "good evening",
    "greetings", "hello there", "hey there", "hi there",
    "morning", "evening", "howdy", "nice to meet you",
    "pleased to meet you", "good day"
]
df = pd.read_csv(f'{base_path}/symptoms.csv')
symptom_keywords = df['symptom'].dropna().str.lower().tolist()

gratitude_vectors = model.encode(gratitude_keywords, convert_to_tensor=True)
greeting_vectors = model.encode(greeting_keywords, convert_to_tensor=True)
service_vectors = model.encode(service_keywords, convert_to_tensor=True)
symptom_vectors = model.encode(symptom_keywords, convert_to_tensor=True)

from sentence_transformers import util


def classify_intent_semantic(
        text,
        model,
        greeting_vectors,
        gratitude_vectors,
        symptom_vectors,
        service_vectors,  # Make optional
        threshold=0.6
):
    input_embedding = model.encode(text, convert_to_tensor=True)

    greeting_score = float(util.cos_sim(input_embedding, greeting_vectors).max())
    symptom_score = float(util.cos_sim(input_embedding, symptom_vectors).max())
    gratitude_score = float(util.cos_sim(input_embedding, gratitude_vectors).max())
    service_score = float(util.cos_sim(input_embedding, service_vectors).max())

    scores = {
        "greeting": greeting_score,
        "gratitude": gratitude_score,
        "symptom": symptom_score,
        "service": service_score
    }

    top_intent = max(scores, key=scores.get)

    if scores[top_intent] > threshold:
        return top_intent
    else:
        return "uncertain"


report_keywords = [
    "pdf", "report", "summary", "generate report", "health summary",
    "medical history", "create report", "download report", "export data",
    "send report", "document", "health records", "medical summary"
]

diet_keywords = [
    "diet", "meal plan", "food plan", "eating plan", "nutrition plan",
    "weekly diet", "custom diet", "personalized diet", "diet chart",
    "meal schedule", "diet routine", "food schedule", "diet recommendation",
    "what should i eat", "healthy eating plan", 'food recommendation'
]

exercise_keywords = [
    "exercise", "workout", "fitness", "training", "exercise plan",
    "workout routine", "exercise schedule", "fitness plan",
    "training program", "exercise chart", "workout schedule",
    "personal training", "fitness routine", "gym plan", "home workout"
]

medicine_keywords = [
    "medicine", "meds", "prescription", "drug", "medication",
    "what to take", "treatment", "pills", "pharmacy",
    "recommend medicine", "suggest medication", "drug recommendation",
    "pain relief", "should i take", "dosage", "pharmaceutical"
]

report_vectors = model.encode(report_keywords, convert_to_tensor=True)
diet_vectors = model.encode(diet_keywords, convert_to_tensor=True)
exercise_vectors = model.encode(exercise_keywords, convert_to_tensor=True)
medicine_vectors = model.encode(medicine_keywords, convert_to_tensor=True)


def classify_service(
        text,
        model,
        report_vectors,
        diet_vectors,
        exercise_vectors,
        medicine_vectors,
        threshold=0.3
):
    input_embedding = model.encode(text, convert_to_tensor=True)

    report_score = float(util.cos_sim(input_embedding, report_vectors).max())
    diet_score = float(util.cos_sim(input_embedding, diet_vectors).max())
    exercise_score = float(util.cos_sim(input_embedding, exercise_vectors).max())
    medicine_score = float(util.cos_sim(input_embedding, medicine_vectors).max())

    scores = {
        "report": report_score,
        "diet": diet_score,
        "exercise": exercise_score,
        "medicine": medicine_score
    }

    top_score = max(scores, key=scores.get)
    return top_score


def create_input_vector(user_symptoms, all_symptoms):
    user_symptoms_lower = [s.strip().lower() for s in user_symptoms]
    all_symptoms_lower = [s.strip().lower() for s in all_symptoms]

    # Create dictionary with all symptoms initialized to 0
    input_dict = {symptom: 0 for symptom in all_symptoms}

    for user_symptom in user_symptoms_lower:
        if user_symptom in all_symptoms_lower:

            idx = all_symptoms_lower.index(user_symptom)
            original_case = all_symptoms[idx]
            input_dict[original_case] = 1
        else:
            print(f"Warning: Symptom '{user_symptom}' not found in trained symptoms list")

    return pd.DataFrame([input_dict])[all_symptoms]


def format_food_recommendations(category, sample_size=7):
    df = category_food(category)
    if df.empty:
        return "No specific food recommendations available for this condition."

    # Shuffle & take random subset (min ensures not exceeding available rows)
    sample_size = min(sample_size, len(df))
    df_sample = df.sample(n=sample_size, random_state=random.randint(1, 9999))

    recommendations = ["🍎 Recommended Foods:"]
    for _, row in df_sample.iterrows():
        rec = f"- {row['Food']}"
        if pd.notna(row.get('Calories')):
            rec += f" ({row['Calories']} cal)"
        if pd.notna(row.get('Key Nutrients')):
            rec += f" | Nutrients: {row['Key Nutrients']}"
        recommendations.append(rec)

    return '\n'.join(recommendations)


def format_exercise_recommendations(category):
    df = category_exercise(category)
    if df.empty:
        return "No specific exercise recommendations available for this condition."

    recommendations = ["🏋️ Recommended Exercises:"]
    for _, row in df.iterrows():
        rec = f"\n- {row['Exercise']}"
        if pd.notna(row.get('Intensity')):
            rec += f" ({row['Intensity']})"
        if pd.notna(row.get('Key Benefits')):
            rec += f" | Benefits: {row['Key Benefits']}"
        recommendations.append(rec)

    return '\n'.join(recommendations)


def format_medicine_recommendations(category, is_adult=True):
    df = category_med(category)
    if df.empty:
        return "No specific medicine recommendations available for this condition."

    recommendations = ["💊 Recommended Medicines:"]
    for _, row in df.iterrows():
        dosage = row['Adults'] if is_adult else row['Children']
        if pd.isna(dosage) or dosage in ['', 'N/A']:
            continue

        rec = f"\n- {row['Medicine']}: {dosage}"
        if pd.notna(row.get('Notes')):
            rec += f" | Notes: {row['Notes']}"
        recommendations.append(rec)

    if len(recommendations) == 1:  # Only header was added
        return "No suitable medicines found for your age group. Please consult a doctor."

    return '\n'.join(recommendations)


def generate_pdf_report(user_info, disease, category, nutrients, exercises, meds):
    try:
        os.makedirs("./reports", exist_ok=True)
        filename = f"./reports/{user_info['name'].replace(' ', '_')}_DoctorAI_Report.pdf"

        doc = SimpleDocTemplate(filename, pagesize=A4,
                                rightMargin=40, leftMargin=40,
                                topMargin=50, bottomMargin=40)

        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(name='CenterHeading', alignment=1, fontSize=18, spaceAfter=10, textColor=colors.darkblue))
        styles.add(
            ParagraphStyle(name='SubHeading', fontSize=14, spaceAfter=8, textColor=colors.darkred, underline=True))
        styles.add(ParagraphStyle(name='NormalText', fontSize=11, spaceAfter=6))

        content = []

        # Title
        content.append(Paragraph(" Doctor AI Diagnosis Report", styles['CenterHeading']))
        content.append(Spacer(1, 12))

        # Date
        content.append(Paragraph(f"Date: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles['NormalText']))
        content.append(Spacer(1, 12))

        # Patient Info
        content.append(Paragraph(" Patient Information", styles['SubHeading']))
        patient_table_data = [
            ["Name:", user_info["name"]],
            ["Age:", str(user_info["age"])],
            ["Gender:", user_info["gender"].capitalize()],
            ["Location:", user_info.get("location", "N/A")]
        ]
        patient_table = Table(patient_table_data, hAlign='LEFT', colWidths=[80, 400])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        content.append(patient_table)
        content.append(Spacer(1, 12))

        # Predicted Diagnosis
        content.append(Paragraph(" Predicted Diagnosis", styles['SubHeading']))
        content.append(Paragraph(f"<b>Disease:</b> {disease.title()}", styles['NormalText']))
        content.append(Paragraph(f"<b>Category:</b> {category}", styles['NormalText']))

        try:
            description = category_description(category)
            content.append(Paragraph(f"<b>Description:</b> {description}", styles['NormalText']))
        except Exception:
            content.append(Paragraph("<b>Description:</b> No description available.", styles['NormalText']))

            content.append(Spacer(1, 12))

        # Nutritional Foods (max 5)
        if not nutrients.empty:
            content.append(Paragraph(" Recommended Nutritional Foods", styles['SubHeading']))
            nutri_table_data = [["Food", "Calories", "Sugar (g)", "Key Nutrients"]]
            for _, row in nutrients.head(5).iterrows():  # limit to 5
                nutri_table_data.append([
                    row['Food'],
                    str(row.get('Calories', 'N/A')),
                    str(row.get('Sugar (g)', 'N/A')),
                    row.get('Key Nutrients', 'N/A')
                ])
            nutri_table = Table(nutri_table_data, colWidths=[130, 70, 70, 220])
            nutri_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            content.append(nutri_table)
            content.append(Spacer(1, 14))

        # Exercises (max 5, no benefits)
        if not exercises.empty:
            content.append(Paragraph(" Recommended Exercises", styles['SubHeading']))
            exercise_table_data = [["Exercise", "Intensity", "Calories Burned (30 min)"]]
            for _, row in exercises.head(5).iterrows():  # limit to 5
                exercise_table_data.append([
                    row.get('Exercise', 'N/A'),
                    row.get('Intensity', 'N/A'),
                    row.get('Calories Burned (per 30 min)', 'N/A')
                ])
            exercise_table = Table(exercise_table_data, colWidths=[180, 100, 210])
            exercise_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            content.append(exercise_table)
            content.append(Spacer(1, 14))

        # Medicines (max 5)
        if not meds.empty:
            content.append(Paragraph(" Suggested Medicines", styles['SubHeading']))
            med_table_data = [["Medicine", "Dosage", "Notes"]]
            for _, row in meds.head(5).iterrows():  # limit to 5
                dosage = row.get("Adults", "N/A") if user_info.get("age_group") == 'adult' else row.get("Children",
                                                                                                        "N/A")
                med_table_data.append([
                    row.get("Medicine", "N/A"),
                    dosage,
                    row.get("Notes", "N/A")
                ])
            med_table = Table(med_table_data, colWidths=[130, 100, 260])
            med_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.salmon),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            content.append(med_table)
            content.append(Spacer(1, 12))

        content.append(Spacer(1, 20))
        content.append(Paragraph("🩺 Generated by Doctor AI — Your smart health companion", styles['NormalText']))

        doc.build(content)
        print(f"\n PDF saved successfully: {filename}")
        return True

    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False


def extract_candidate_phrases(text, nlp_model):
    doc = nlp_model(text)
    phrases = set()

    for chunk in doc.noun_chunks:
        phrases.add(chunk.text.lower())
    for token in doc:
        if token.pos_ in ["NOUN", "ADJ"] and not token.is_stop:
            phrases.add(token.text.lower())

    return list(phrases)


def filter_symptom_phrases(phrases, model, symptom_vectors, threshold=0.5):
    if not phrases:
        return []
    phrase_vectors = model.encode(phrases, convert_to_tensor=True)
    scores = util.cos_sim(phrase_vectors, symptom_vectors)
    return [phrases[i] for i in range(len(phrases)) if torch.max(scores[i]) >= threshold]


def remove_subsets(phrases):
    return [p for p in phrases if not any((p != other and p in other) for other in phrases)]


def get_matched_symptoms(filtered_phrases, model, symptom_columns, symptom_vectors, k_extra=2):
    if not filtered_phrases:
        return []

    exact_matches = []
    symptom_columns_lower = [s.lower() for s in symptom_columns]

    for phrase in filtered_phrases:
        phrase_lower = phrase.lower()
        if phrase_lower in symptom_columns_lower:
            idx = symptom_columns_lower.index(phrase_lower)
            exact_matches.append(symptom_columns[idx])

    if exact_matches:
        return exact_matches

    input_vector = model.encode(", ".join(filtered_phrases), convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_vector, symptom_vectors)

    k_display = min(len(filtered_phrases) + k_extra, len(symptom_columns))
    top_results = torch.topk(cosine_scores, k=k_display)

    return [symptom_columns[i] for i in top_results.indices[0]]


class MyBot:
    def __init__(self):
        disease_handler = DiseaseRecognitionHandler()

        service_handler = ServiceHandler(
            model,
            report_vectors,
            diet_vectors,
            exercise_vectors,
            medicine_vectors,
            disease_handler
        )

        self.categories = {
            'greeting': GreetingHandler(),
            'gratitude': GratitudeHandler(),
            'service': service_handler,
            'symptom': disease_handler,
            'uncertain': UncertainHandler(disease_handler)
        }

        self.current_stage = None

    def process(self, input_text):
        # Classify intent
        category = classify_intent_semantic(
            input_text,
            model,
            greeting_vectors,
            gratitude_vectors,
            symptom_vectors,
            service_vectors,
            threshold=0.4
        )

        # Reset disease handler ONLY when switching into it
        if category == "symptom" and not isinstance(self.current_stage, DiseaseRecognitionHandler):
            self.categories["symptom"].reset()

        # Continue current stage if exists
        if self.current_stage:
            response, next_stage = self.current_stage.handle(input_text)
        else:
            handler = self.categories[category]
            response, next_stage = handler.handle(input_text)

        # Update stage
        self.current_stage = next_stage if next_stage else None
        return response


class DiseaseRecognitionHandler:
    def __init__(self, llm_handler=None):
        self.reset()
        self.llm_handler = llm_handler

    def reset(self):
        self.sub_stage = None
        self.identified_disease = None
        self.matched_symptoms = None
        self.category = None
        self.description = None

    def handle(self, input_text):
        try:
            # --- STEP 1: No disease identified yet ---
            if not self.identified_disease:

                # --- CASE A: First time symptom extraction ---
                if self.matched_symptoms is None:
                    raw_phrases = extract_candidate_phrases(input_text, nlp)
                    filtered_phrases = filter_symptom_phrases(raw_phrases, model, symptom_vectors)
                    filtered_phrases = remove_subsets(filtered_phrases)
                    matched_symptoms = get_matched_symptoms(
                        filtered_phrases, model, symptom_columns, symptom_vectors
                    )

                    if not matched_symptoms:
                        # Fallback if nothing found
                        if self.llm_handler:
                            return self.llm_handler.handle(input_text)
                        return (
                            "I'm sorry, I couldn't clearly identify your symptoms. "
                            "Could you please rephrase or describe them again so I can determine the possible condition?",
                            self
                        )

                    # ✅ Store detected symptoms
                    self.matched_symptoms = matched_symptoms
                    symptoms_text = ", ".join(matched_symptoms)

                    return (
                        f"I identified the following symptoms from your input: {symptoms_text}.\n"
                        f"Do these look correct? Please type 'yes' to confirm or rewrite your symptoms.",
                        self  # keep self alive
                    )

                else:
                    # --- CASE B: User is confirming or correcting symptoms ---
                    if input_text.lower().strip() == 'yes':
                        # ✅ user confirmed, use previously stored symptoms
                        confirmed_symptoms = self.matched_symptoms
                    else:
                        # User provided corrections → re-run extraction
                        raw_phrases = extract_candidate_phrases(input_text, nlp)
                        filtered_phrases = filter_symptom_phrases(raw_phrases, model, symptom_vectors)
                        filtered_phrases = remove_subsets(filtered_phrases)
                        confirmed_symptoms = get_matched_symptoms(
                            filtered_phrases, model, symptom_columns, symptom_vectors
                        )
                        # Update stored symptoms with the new ones
                        self.matched_symptoms = confirmed_symptoms

                    # Check if we have valid symptoms after confirmation/correction
                    if not confirmed_symptoms:
                        if self.llm_handler:
                            return self.llm_handler.handle(input_text)
                        return (
                            "I'm sorry, I still couldn't clearly detect your symptoms. "
                            "Please describe them again.",
                            self
                        )

                    # Proceed with disease prediction using confirmed symptoms
                    input_vector = create_input_vector(confirmed_symptoms, symptom_columns)
                    probas = pipeline.predict_proba(input_vector)

                    if probas is None or len(probas[0]) == 0:
                        if self.llm_handler:
                            return self.llm_handler.handle(input_text)
                        return ("Prediction failed. Please try again later.", self)

                    top_classes = np.argsort(probas[0])[::-1][:3]
                    predicted_diseases = [class_dict.get(str(i), "Unknown Disease") for i in top_classes]

                    if not predicted_diseases or predicted_diseases[0] == "Unknown Disease":
                        if self.llm_handler:
                            return self.llm_handler.handle(input_text)
                        return ("I couldn't confidently identify a condition from your symptoms.", self)

                    # ✅ success
                    self.identified_disease = predicted_diseases[0]
                    category_list = category_find(self.identified_disease)
                    self.category = category_list[0] if category_list else "N/A"
                    self.description = category_description(self.category)

                    return self._generate_options_response()

            # --- STEP 2: Handle user requests for recommendations ---
            if "diet" in input_text.lower():
                text_result = format_food_recommendations(self.category)
                file_path = "./static/nutritional_profile.png"
                return ((text_result, file_path), None)

            elif "exercise" in input_text.lower():
                img_path = plot_exercise_recommendations(
                    self.category,
                    save_path="./static/category_ex.png"
                )
                return ((format_exercise_recommendations(self.category), img_path), None)

            elif "medicine" in input_text.lower():
                is_adult = hasattr(self, 'user_info') and self.user_info.get('age_group') == 'adult'
                img_path = plot_medicine_recommendations(
                    self.category,
                    save_path="./static/category_med.png"
                )
                return ((format_medicine_recommendations(self.category, is_adult), img_path), None)

            else:
                return (
                    "I'm DoctorAi. Please let me know what kind of guidance you'd like:\n"
                    "- 🥗 Diet\n"
                    "- 🏃 Exercise\n"
                    "- 💊 Medicine",
                    None
                )

        except Exception as e:
            if self.llm_handler:
                return self.llm_handler.handle(input_text)
            return (f"⚠️ An error occurred: {str(e)}", self)

    def _generate_options_response(self):
        img_path = plot_category_distribution(
            self.category,
            save_path="./static/category_dist.png"
        )

        response_text = (
            f"Based on the symptoms you've shared, I believe the most likely condition is: **{self.identified_disease}**, "
            f"which falls under the **{self.category}** category.\n\n"
            f"ℹ️ {self.description}\n\n"
            f"Here is a visualization of count of diseases per category:\n\n"
            f"Would you like personalized recommendations for:\n"
            f"1️⃣ Diet\n2️⃣ Exercise\n3️⃣ Medicine?"
        )

        return ((response_text, img_path), self)


class GreetingHandler:
    def __init__(self):
        self.stage = None
        self.positive_response_keys = ["good", "fine", "well", "ok", "okay", "great", "better"]
        self.symptom_keywords = ["pain", "ache", "fever", "cough", "nausea", "dizzy", "rash", "headache"]

    def handle(self, input_text):
        if self.stage is None:
            self.stage = 'greet'
            return (
                "👋 Hello! I'm DoctorAi, your virtual health assistant. I’m here to help you identify possible conditions based on your symptoms.\n\nHow are you feeling today?",
                None
            )

        input_lower = input_text.lower()

        if any(word in input_lower for word in self.symptom_keywords):
            self.stage = None
            return (
                "",
                DiseaseRecognitionHandler()
            )

        if any(word in input_lower for word in self.positive_response_keys) and self.stage == 'greet':
            self.stage = 'awaiting_symptoms'
            return (
                "😊 I'm glad to hear you're feeling okay. If you're experiencing any symptoms—even mild ones—feel free to describe them (e.g., headache, fever, nausea), and I’ll do my best to assist.",
                None
            )

        return (
            "🤔 I didn’t quite catch that. Could you let me know if you're feeling well, or describe any symptoms you're having?",
            None
        )


class GratitudeHandler:
    def __init__(self):
        self.stage = None  # Tracks conversation state

    def handle(self, input_text):
        """Handle gratitude expressions with appropriate responses"""
        try:
            if self.stage is None:
                self.stage = 'acknowledge'
                return (
                    "You're very welcome! 😊\n"
                    "Warm regards,\n"
                    "**DoctorAi**",
                    None
                )

            return (
                "Is there anything else I can assist you with today?",
                None
            )

        except Exception as e:
            print(f"Error in GratitudeHandler: {str(e)}")
            return ("I appreciate your thanks! How else can I help?", None)


class UncertainHandler:
    def __init__(self, disease_handler):
        self.attempts = 0
        self.disease_handler = disease_handler
        self.ask_gemini = ask_gemini

    def handle(self, input_text):
        self.attempts += 1

        # Create the system prompt
        system_prompt = (
            "You are a helpful, polite doctor AI assistant. "
            "You take symptoms from the user and suggest a probable disease "
            "and tell them what category the disease is in. "
            "The disease should be **one word only**. "
            "The category must be one of the following:\n"
            "Cardiovascular, Dermatological, Ear Nose and Throat, Endocrine, Gastrointestinal, "
            "Genetic, Hematological, Hepatic, Immunological, Infectious, Metabolic, "
            "Musculoskeletal, Neurological, Oncological, Ophthalmological, Oral, Pediatric, "
            "Reproductive, Respiratory, Toxicology, Urinary, Mental Health, Surgical, Lymphatic.\n\n"
            "Give an easy description of the disease/category. "
            "After suggesting, only ask: 'Do you want a diet, exercise or medicine recommendation?'. "
            "If the user input is off-topic, give him polite reply for his"
            "saying and then again politely ask for symptoms."
        )

        # Combine system prompt with user input
        full_prompt = f"{system_prompt}\n\nUser: {input_text}\n\nAssistant:"

        # Call Gemini API
        response_text = self.ask_gemini(full_prompt)

        # Try extracting structured disease + category from response
        disease, category = self._extract_disease_category(response_text)

        if disease and category:
            # Save into disease handler
            self.disease_handler.identified_disease = disease
            self.disease_handler.category = category
            self.disease_handler.description = f"(From LLM) {response_text}"

            # Switch to DiseaseHandler → return its menu response
            return self.disease_handler._generate_options_response()

        else:
            # Off-topic or uncertain → return response + None so flow loops here again
            return (
                response_text + "\n\nCould you describe your symptoms more clearly?",
                None
            )

    def _extract_disease_category(self, llm_text):
        """
        Extracts (disease, category) from LLM text.
        """
        known_categories = [
            "Cardiovascular", "Dermatological", "Ear Nose and Throat", "Endocrine",
            "Gastrointestinal", "Genetic", "Hematological", "Hepatic", "Immunological",
            "Infectious", "Metabolic", "Musculoskeletal", "Neurological", "Oncological",
            "Ophthalmological", "Oral", "Pediatric", "Reproductive", "Respiratory",
            "Toxicology", "Urinary", "Mental Health", "Surgical", "Lymphatic"
        ]

        found_category = None
        for cat in known_categories:
            if cat.lower() in llm_text.lower():
                found_category = cat
                break

        found_disease = None
        if found_category:
            # Look for "X ... falls under Y" or "X is ... Y category"
            match = re.search(

                "([A-Z][a-zA-Z]+)\s+(?:is|disease|condition|falls|likely).+?" + re.escape(found_category),
                llm_text,
                re.IGNORECASE
            )
            if match:
                found_disease = match.group(1).strip()

        return found_disease, found_category


class ServiceHandler:
    def __init__(self, model, report_vectors, diet_vectors, exercise_vectors, medicine_vectors, disease_handler):
        self.model = model
        self.report_vectors = report_vectors
        self.diet_vectors = diet_vectors
        self.exercise_vectors = exercise_vectors
        self.medicine_vectors = medicine_vectors
        self.disease_handler = disease_handler  # Store the disease handler reference
        self.threshold = 0.3
        self._reset_state()

    def _has_symptoms(self):
        """Check if symptoms were processed and disease identified"""
        return hasattr(self.disease_handler, 'identified_disease') and self.disease_handler.identified_disease

    def _reset_state(self):
        """Reset all state variables"""
        self.service_type = None
        self.user_data = {
            'name': None,
            'age': None,
            'age_group': None,
            'gender': None,
            'location': 'N/A'
        }
        self._info_stage = 0
        if hasattr(self.disease_handler, 'reset_state'):
            self.disease_handler.reset_state()

    def _identify_service_type(self, input_text):
    # "Identify what type of service the user wants"
    # ❌ remove unconditional reset
    # self._reset_state()

     service_type = classify_service(
        text=input_text,
        model=self.model,
        report_vectors=self.report_vectors,
        diet_vectors=self.diet_vectors,
        exercise_vectors=self.exercise_vectors,
        medicine_vectors=self.medicine_vectors,
        threshold=self.threshold
    )

     print(f"DEBUG: Classified '{input_text}' as: {service_type}")
     self.service_type = service_type

    # ✅ Report flow
     if service_type == 'report':
        if not self._has_symptoms():
            return ("⚠️ Please describe your symptoms first so I can generate you a report.", None)
        return ("Sure, I'll prepare a health report for you.\nLet's start with your name:", self)

    # ✅ Diet flow
     if service_type == 'diet':
        if any(word in input_text.lower() for word in ["plan", "schedule", "weekly", "chart"]):
            return self._handle_diet_flow(input_text)  # weekly plan
        else:
            img_path = "./static/nutritional_profile.png"
            return (format_food_recommendations(self.disease_handler.category), img_path), None

    # ✅ Exercise flow
     if service_type == 'exercise':
        if any(word in input_text.lower() for word in ["plan", "schedule", "weekly", "routine", "chart"]):
            return self._handle_exercise_flow(input_text)  # weekly plan
        else:
            img_path = plot_exercise_recommendations(
                self.disease_handler.category,
                save_path="./static/category_ex.png"
            )
            return (format_exercise_recommendations(self.disease_handler.category), img_path), None

    # ✅ Medicine flow
     if service_type == 'medicine':
        if any(word in input_text.lower() for word in ["plan", "schedule", "weekly", "chart"]):
            return self._handle_medicine_flow(input_text)  # medicine plan
        else:
            img_path = plot_medicine_recommendations(
                self.disease_handler.category,
                save_path="./static/category_med.png"
            )
            return (format_medicine_recommendations(self.disease_handler.category, is_adult=True), img_path), None

     return ("Please specify: report, diet plan, exercise routine, or medicine info.", self)

    def handle(self, input_text):
        """Main handler for service requests"""
        try:
            # If we are at the beginning, detect service type
            if self.service_type is None or self._info_stage == 0:
                if self._looks_like_service_request(input_text):
                    self._reset_state()
                    return self._identify_service_type(input_text)

            if self.service_type:
                return self._continue_current_service(input_text)

            return ("Please specify: report, diet plan, exercise routine, or medicine info.", self)

        except Exception as e:
            print(f"Error in ServiceHandler: {str(e)}")
            return ("⚠️ An error occurred. Please try again.", None)

    def _looks_like_service_request(self, text):
        """Improved service request detection"""
        text = text.lower()
        service_keywords = {
            'report': ['report', 'health summary'],
            'diet': ['diet', 'food', 'meal', 'nutrition', 'eat'],
            'exercise': ['exercise', 'workout', 'fitness', 'train'],
            'medicine': ['medicine', 'medication', 'pill', 'drug', 'prescription']
        }

        for service, keywords in service_keywords.items():
            if any(keyword in text for keyword in keywords):
                return True

        return False

    def _continue_current_service(self, input_text):
        """Continue the current service flow"""
        if self.service_type == 'report':
            return self._handle_report_flow(input_text)
        elif self.service_type == 'diet':
            return self._handle_diet_flow(input_text)
        elif self.service_type == 'exercise':
            return self._handle_exercise_flow(input_text)
        elif self.service_type == 'medicine':
            return self._handle_medicine_flow(input_text)
        return ("Please specify what you need help with.", None)

    def _handle_report_flow(self, input_text):
        if not self._has_symptoms():
            return ("⚠️ Please describe your symptoms first so I can generate you a report.",
                    None)
        try:
            if self._info_stage == 0:  # Get name
                self.user_data["name"] = input_text.strip().title()
                self._info_stage += 1
                return ("Great. What's your age?", self)  # Fixed this line

            elif self._info_stage == 1:  # Get age
                age_clean = extract_single_age(input_text)
                if age_clean and is_valid_number(age_clean):
                    age = float(age_clean)
                    self.user_data.update({
                        "age": age,
                        "age_group": "adult" if age >= 18 else "child"
                    })
                    self._info_stage += 1
                    return ("Got it. Are you male or female?", self)
                return ("❌ Please enter a valid age (e.g., 25 or 30.5)", self)

            elif self._info_stage == 2:  # Get gender
                gender = extract_gender(input_text)
                if gender in ['male', 'female']:
                    self.user_data["gender"] = gender
                    self._info_stage += 1
                    return ("✅ Noted. Where are you located? (This is optional)", self)
                return ("❌ Please specify your gender as 'male' or 'female'.", self)

            elif self._info_stage == 3:  # Get location and generate
                self.user_data["location"] = input_text.strip() or "N/A"
                return self._generate_report()

        except Exception as e:
            print(f"Report generation error: {str(e)}", file=sys.stderr)
            return ("⚠️ Error in report generation. Please start over.", None)

    def _generate_report(self):
        """Generate the final PDF report"""
        try:
            if not hasattr(self.disease_handler, 'identified_disease'):
                return ("⚠️ Please describe your symptoms first.", None)

            disease = self.disease_handler.identified_disease
            category = self.disease_handler.category

            success = generate_pdf_report(
                self.user_data,
                disease,
                category,
                category_food(category),
                category_exercise(category),
                category_med(category)
            )

            msg = ("✅ Report generated successfully!" if success
                   else "❌ Failed to generate report")
            return (msg, None)
        except Exception as e:
            print(f"PDF generation error: {str(e)}", file=sys.stderr)
            return ("⚠️ Error generating PDF. Please try again.", None)

    def _handle_diet_flow(self, input_text):
        """Generate diet plan only if symptoms were provided"""
        if not self._has_symptoms():
            return ("⚠️ Please describe your symptoms first so I can recommend appropriate foods.",
                    None)

        try:
            food_df = category_food(self.disease_handler.category)
            if food_df.empty:
                return (self._generate_general_diet_plan(), None)

            food_items = []
            for _, row in food_df.iterrows():
                food_items.append((
                    row['Food'],
                    int(float(row['Calories'])),
                    float(row['Sugar (g)']),
                    float(row['Carbs (g)']),
                    float(row['Fat (g)']),
                    float(row['Sodium (mg)']),
                    row.get('Key Nutrients', 'Various nutrients')
                ))

            return (self._generate_7day_plan(food_items), None)

        except Exception as e:
            print(f"Diet generation error: {str(e)}")
            return (self._generate_general_diet_plan(), None)

    def _generate_7day_plan(self, food_items):
        """
        Generate a balanced 7-day diet plan.
        Each day gets 3 unique foods (breakfast, lunch, dinner).
        Shows macros + daily totals.
        """
        days = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
        plan = ["🍽️ Your Personalized 7-Day Diet Plan:\n"]

        for day in days:
            meals = random.sample(food_items, min(3, len(food_items)))

            daily_cals = daily_carbs = daily_fat = daily_sugar = daily_sodium = 0
            plan.append(f"📅 {day}:")

            meal_names = ["Breakfast", "Lunch", "Dinner"]
            for meal_name, food in zip(meal_names, meals):
                name, cal, sugar, carbs, fat, sodium, nutrients = food

                plan.append(
                    f"  - {meal_name}: {name} ({cal} cal) "
                    f"| Carbs: {carbs}g | Fat: {fat}g | Sugar: {sugar}g | Sodium: {sodium}mg\n"
                    f"    Nutrients: {nutrients}"
                )

                # Add to daily totals
                daily_cals += cal
                daily_carbs += carbs
                daily_fat += fat
                daily_sugar += sugar
                daily_sodium += sodium

            # Summary line for the day
            plan.append(
                f"  🔹 Daily Total: {daily_cals} cal | Carbs: {daily_carbs}g | "
                f"Fat: {daily_fat}g | Sugar: {daily_sugar}g | Sodium: {daily_sodium}mg\n"
            )

        return "\n".join(plan)

    def _generate_general_diet_plan(self):
        """Fallback healthy diet plan"""
        return """🍽️ General Healthy Diet Plan:
- Breakfast: Oatmeal with berries (300 cal)
- Lunch: Grilled chicken with vegetables (400 cal)
- Dinner: Salmon with quinoa (450 cal)
- Snacks: Greek yogurt, almonds, fresh fruits

Stay hydrated with water and herbal teas!"""

    def _handle_exercise_flow(self, input_text):
        """Generate exercise plan immediately without confirmation"""
        if not self._has_symptoms():
            return ("⚠️ Please describe your symptoms first so I can recommend appropriate exercises.",
                    None)
        try:

            exercises_df = category_exercise(self.disease_handler.category)

            if exercises_df.empty:
                return (self._generate_general_exercise_plan(), None)

            exercise_items = []
            for _, row in exercises_df.iterrows():
                calories = str(row['Calories Burned (per 30 min)'])
                if '-' in calories:
                    avg_calories = sum(map(int, calories.split('-'))) // 2
                else:
                    try:
                        avg_calories = int(float(calories))
                    except:
                        avg_calories = 150  # fallback

                exercise_items.append((
                    row['Exercise'],
                    row['Intensity'],
                    avg_calories,
                    row.get('Key Benefits', 'Various benefits'),
                    row.get('Category', 'General')
                ))

            return (self._generate_7day_exercise_plan(exercise_items), None)

        except Exception as e:
            print(f"Exercise generation error: {str(e)}")
            return (self._generate_general_exercise_plan(), None)

    def _generate_7day_exercise_plan(self, exercise_items):
        """
        Generate a balanced 7-day exercise plan with variety.
        Each day gets a random exercise, ensuring mix of categories if possible.
        """
        days = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
        plan = ["🏋️ Your Personalized 7-Day Exercise Plan:\n"]

        # Shuffle for variety
        random.shuffle(exercise_items)

        tips = [
            "Stay hydrated during your workout 💧",
            "Warm up for 5-10 minutes before exercising 🔥",
            "Focus on correct posture to avoid injury ✅",
            "Cool down with stretching after workout 🧘",
            "Gradually increase intensity, don’t rush ⚡",
            "Mix cardio and strength for better results 💪",
            "Rest is part of training — listen to your body 💤"
        ]

        for i, day in enumerate(days):
            exercise = exercise_items[i % len(exercise_items)]
            name, intensity, calories, benefits, category = exercise

            # Estimate 30min and 60min burn
            cal_30 = calories
            cal_60 = calories * 2

            plan.append(
                f"📅 {day}:"
                f"\n  - Exercise: {name} ({category})"
                f"\n  - Intensity: {intensity}"
                f"\n  - Calories Burned: ~{cal_30} (30 min) | ~{cal_60} (1 hr)"
                f"\n  - Benefits: {benefits}"
                f"\n  💡 Tip: {tips[i]}"
            )
            plan.append("")  # blank line for readability

        return "\n".join(plan)

    def _generate_general_exercise_plan(self):
        """Fallback exercise plan"""
        return """🏋️ General Exercise Plan:
    - Monday: 30 min brisk walking
    - Tuesday: 30 min yoga
    - Wednesday: 30 min swimming
    - Thursday: Rest day
    - Friday: 30 min cycling
    - Saturday: 30 min strength training
    - Sunday: Recreational activity

    Aim for 150+ minutes of exercise weekly!"""

    def _handle_medicine_flow(self, input_text):
        """Provide medicine recommendations immediately if disease is identified"""
        if not self._has_symptoms():
            return ("⚠️ Please describe your symptoms first so I can recommend appropriate medicines.",
                    None)
        try:

            if hasattr(self.disease_handler, 'identified_disease'):

                meds = category_med(self.disease_handler.category)
                if meds.empty:
                    return ("❌ No specific medicine recommendations found for this condition. Please consult a doctor.",
                            None)

                if 'age_group' not in self.user_data:
                    age = extract_single_age(input_text)
                    if age and is_valid_number(age):
                        self.user_data.update({
                            'age': float(age),
                            'age_group': 'adult' if float(age) >= 18 else 'child'
                        })
                    else:
                        return ("🔍 For proper dosage information, please provide your age (e.g., 'I am 25').", self)

                response = ["💊 Recommended Medicines for your condition:"]

                for _, row in meds.iterrows():

                    dosage_col = 'Adults' if self.user_data['age_group'] == 'adult' else 'Children'
                    dosage = row.get(dosage_col, 'Consult doctor')

                    if pd.isna(dosage) or dosage in ['0', 0, 'N/A']:
                        continue

                    item = f"\n- {row['Medicine']}:"
                    item += f"\n  • Dosage: {dosage}"

                    if pd.notna(row.get('Notes')):
                        item += f"\n  • Notes: {row['Notes']}"

                    response.append(item)

                if len(response) == 1:
                    return ("❌ No suitable medicines found for your age group. Please consult a doctor.", None)

                response.extend([
                    "\n⚠️ Important:",
                    "• Always consult your doctor before taking any medication",
                    "• Report any side effects immediately",
                    "• Never share medications with others",
                    "• Follow dosage instructions carefully"
                ])

                return ('\n'.join(response), None)
            else:
                return ("Please describe your symptoms first so I can recommend appropriate medicines.", None)

        except Exception as e:
            print(f"Medicine error: {str(e)}", file=sys.stderr)
            return ("⚠️ Error retrieving medicine information. Please try again.", None)


def split_payload(payload):
    """
    Takes a variable that may contain:
      - (response_text, img_path)
      - response_text

    Returns:
        response_text (str)
        img_path (str or None)
    """
    if isinstance(payload, tuple) and len(payload) == 2:
        response_text, img_path = payload
        return response_text, img_path
    elif isinstance(payload, str):
        return payload, None
    else:
        raise ValueError(f"NotValidPayload: {payload}")


def get_bot_instance():
    """Returns initialized bot instance for server use"""
    return MyBot()


def process_text_input(text, bot_instance=None):
    """Process text input and return response for server"""
    if bot_instance is None:
        bot_instance = MyBot()

    try:
        doc, lang = detect_and_clean_languages(text)
        translated = translate_english(doc)
        dest_lang = get_language_code(lang)
        response, path = split_payload(bot_instance.process(translated))
        final_response = translate(response, dest_lang)

        return {
            'success': True,
            'response': final_response,
            'image_path': path,
            'language': lang
        }
    except Exception as e:
        return {
            'success': False,
            'response': f'Error: {str(e)}',
            'image_path': None,
            'language': 'english'
        }