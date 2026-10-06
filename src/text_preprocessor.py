"""NLP Text Preprocessing Module for Emotion-Aware Chatbot.

Follows standard NLP preprocessing pipeline:
1. Whitespace and formatting cleanup
2. Contraction expansion (e.g., "I'm" -> "I am")
3. Emoji to sentiment token mapping
4. Redundant punctuation normalization while preserving emotional cues
5. Tokenization and linguistic feature extraction
"""

import re
from typing import Dict, List, Any

# Common English contractions dictionary
CONTRACTION_MAP: Dict[str, str] = {
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "we've": "we have",
    "they're": "they are",
    "they've": "they have",
    "can't": "cannot",
    "couldn't": "could not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "who's": "who is",
}

# Common emotion-rich emoji to text mappings
EMOJI_SENTIMENT_MAP: Dict[str, str] = {
    "😀": " joy ", "😃": " joy ", "😄": " joy ", "😁": " joy ", "😆": " joy ",
    "😊": " happy ", "🥰": " loved ", "😍": " love ", "🤩": " excited ",
    "🥳": " celebrating ", "😢": " sad ", "😭": " crying ", "😞": " depressed ",
    "😔": " disappointed ", "😟": " worried ", "😕": " confused ", "🙁": " sad ",
    "☹️": " unhappy ", "😣": " suffering ", "😖": " distressed ", "😫": " tired ",
    "😩": " weary ", "🥺": " pleading ", "😠": " angry ", "😡": " furious ",
    "🤬": " outraged ", "😤": " frustrated ", "😨": " scared ", "😰": " anxious ",
    "😥": " anxious ", "😱": " terrified ", "😳": " shocked ", "😲": " astonished ",
    "😮": " surprised ", "😯": " stunned ", "🤢": " disgusted ", "🤮": " revolted ",
    "😐": " neutral ", "😑": " indifferent ", "😶": " silent ", "💔": " heartbroken ",
    "❤️": " love ", "💙": " care ", "🫂": " hug ", "🙏": " hopeful "
}


class TextPreprocessor:
    """Preprocesses conversational text for transformer-based emotion classification."""

    def __init__(self):
        # Precompile regex for contraction lookup
        self.contraction_pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")\b",
            re.IGNORECASE,
        )

    def expand_contractions(self, text: str) -> str:
        """Expands informal contractions into full grammatical words."""
        def replace(match: re.Match) -> str:
            word = match.group(0).lower()
            return CONTRACTION_MAP.get(word, match.group(0))
        return self.contraction_pattern.sub(replace, text)

    def replace_emojis(self, text: str) -> str:
        """Translates emojis into descriptive sentiment words to assist NLP embeddings."""
        for emoji_char, replacement in EMOJI_SENTIMENT_MAP.items():
            if emoji_char in text:
                text = text.replace(emoji_char, replacement)
        return text

    def clean_special_characters(self, text: str) -> str:
        """Strips HTML tags, URLs, and excessive repeated punctuation."""
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)
        # Condense repeated punctuation (e.g., '!!!!' -> '!', '?????' -> '?')
        text = re.sub(r"([!?.,])\1+", r"\1", text)
        # Condense excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess(self, text: str) -> Dict[str, Any]:
        """Runs full preprocessing pipeline and extracts text statistics.

        Returns:
            Dict containing:
                - 'raw_text': Original unmodified input
                - 'clean_text': Normalized text optimized for BERT attention
                - 'tokens': Simple whitespace-separated token list
                - 'word_count': Number of words
                - 'char_count': Character length
        """
        if not text or not isinstance(text, str):
            return {
                "raw_text": "",
                "clean_text": "",
                "tokens": [],
                "word_count": 0,
                "char_count": 0,
            }

        raw = text.strip()
        expanded = self.expand_contractions(raw)
        with_emojis = self.replace_emojis(expanded)
        cleaned = self.clean_special_characters(with_emojis)

        # Fallback to raw if cleaning completely emptied the text (e.g. only symbols)
        if not cleaned:
            cleaned = raw

        tokens = re.findall(r"\b\w+\b", cleaned.lower())

        return {
            "raw_text": raw,
            "clean_text": cleaned,
            "tokens": tokens,
            "word_count": len(tokens),
            "char_count": len(cleaned),
        }


# Singleton preprocessor instance for global usage
preprocessor = TextPreprocessor()
