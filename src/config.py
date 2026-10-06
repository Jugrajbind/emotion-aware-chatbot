"""Configuration settings, constants, and color palettes for Emotion-Aware Chatbot."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Supabase Credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

# NLP / Model Configurations
# Default lightweight transformer for emotion sequence classification
# Supports 7 classes: anger, disgust, fear, joy, neutral, sadness, surprise
DEFAULT_MODEL_NAME = os.getenv(
    "EMOTION_MODEL_NAME", "j-hartmann/emotion-english-distilroberta-base"
)

# 7 Standardized Emotions Required for the Project
EMOTIONS = [
    "Joy",
    "Sadness",
    "Anger",
    "Fear",
    "Surprise",
    "Disgust",
    "Neutral",
]

# Visual Palette for 7 Emotions (Modern HSL-tailored hex colors)
EMOTION_COLORS = {
    "Joy": "#F59E0B",       # Warm Amber / Golden
    "Sadness": "#3B82F6",   # Calming Royal Blue
    "Anger": "#EF4444",     # Softened Crimson
    "Fear": "#8B5CF6",      # Purple / Violet
    "Surprise": "#EC4899",  # Electric Pink / Rose
    "Disgust": "#10B981",   # Emerald Sage Green
    "Neutral": "#6B7280",   # Cool Slate Gray
}

# Emotion Emojis & Display Badges
EMOTION_ICONS = {
    "Joy": "✨",
    "Sadness": "🌧️",
    "Anger": "🔥",
    "Fear": "🛡️",
    "Surprise": "⚡",
    "Disgust": "🍃",
    "Neutral": "💬",
}

# Safety Intervention & Crisis Helplines
SAFETY_ENABLED = os.getenv("ENABLE_SAFETY_GUARD", "true").lower() == "true"

CRISIS_RESOURCES = {
    "India": [
        {"name": "Tele-MANAS (Govt of India)", "contact": "14416 or 1800-891-4416", "desc": "24/7 Toll-free Mental Health Helpline"},
        {"name": "KIRAN Helpline", "contact": "1800-599-0019", "desc": "24/7 National Mental Health Rehabilitation"},
        {"name": "Vandrevala Foundation", "contact": "+91 9999 666 555", "desc": "Free 24/7 Crisis Counseling & Mental Health Support"},
        {"name": "AASRA", "contact": "+91 98204 66726", "desc": "24/7 Suicide Prevention & Crisis Intervention"},
    ],
    "United States / Canada": [
        {"name": "Suicide & Crisis Lifeline", "contact": "Call or Text 988", "desc": "24/7 Free and Confidential Support"},
        {"name": "Crisis Text Line", "contact": "Text HOME to 741741", "desc": "24/7 Free Text-based Crisis Counseling"},
        {"name": "The Trevor Project", "contact": "1-866-488-7386", "desc": "24/7 Crisis Intervention for LGBTQ Youth"},
    ],
    "United Kingdom": [
        {"name": "Samaritans", "contact": "Call 116 123", "desc": "Free 24/7 Confidential Listening Service"},
        {"name": "SHOUT Crisis Line", "contact": "Text SHOUT to 85258", "desc": "Free 24/7 Mental Health Support via Text"},
    ],
    "International": [
        {"name": "Befrienders Worldwide", "contact": "befrienders.org", "desc": "Global Network of Emotional Support Lines"},
        {"name": "findahelpline.com", "contact": "https://findahelpline.com", "desc": "Free, Confidential Support in 130+ Countries"},
    ]
}

# Ethical Disclaimer Note
DISCLAIMER_TEXT = (
    "Disclaimer: This AI chatbot is designed for supportive listening and educational "
    "demonstration as part of an engineering mini project. It is NOT a clinical tool, "
    "licensed therapist, or a substitute for professional mental health diagnosis or emergency services."
)
