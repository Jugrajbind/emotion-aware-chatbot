"""Unit & Integration Test Suite for Emotion-Aware Chatbot Pipeline.

Validates:
1. Text Preprocessing & Tokenization
2. Safety Guard & Crisis Intervention Detection
3. Emotion Classifier (7 classes and probability sum)
4. Empathetic Response Generation
5. Database Storage & Analytics Dataframe Extraction
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress Hugging Face internal module warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMOTIONS
from src.text_preprocessor import preprocessor
from src.safety_guard import safety_guard
from src.emotion_classifier import emotion_classifier
from src.response_generator import response_generator
from src.database import db_service
from src.analytics import analytics_engine


def test_preprocessing():
    print("Testing Text Preprocessor...")
    sample = "I'm so exhausted and don't know what to do 😢!!!!"
    result = preprocessor.preprocess(sample)
    assert "i am" in result["clean_text"].lower(), "Contraction 'I'm' not expanded"
    assert "sad" in result["clean_text"].lower(), "Emoji '😢' not translated"
    assert result["word_count"] > 0, "Word count should be positive"
    print("[PASS] Preprocessing test passed.")


def test_safety_guard():
    print("Testing Safety Guard...")
    safe_text = "I had a busy day at college."
    res_safe = safety_guard.analyze(safe_text)
    assert res_safe["is_safe"] is True, "Safe text was incorrectly flagged"

    crisis_text = "I feel like ending my life, cannot go on anymore."
    res_crisis = safety_guard.analyze(crisis_text)
    assert res_crisis["is_safe"] is False, "Crisis text was not flagged"
    assert res_crisis["severity"] == "CRITICAL", "Crisis severity should be CRITICAL"
    assert "14416" in res_crisis["helpline_message"] or "988" in res_crisis["helpline_message"], "Helpline contact missing"
    print("[PASS] Safety Guard test passed.")


def test_emotion_classifier():
    print("Testing Emotion Classifier (7 Emotions)...")
    sample_joy = "I am so happy and thrilled that I passed my exam!"
    pred = emotion_classifier.predict(sample_joy)

    assert pred["predicted_emotion"] in EMOTIONS, f"Predicted emotion '{pred['predicted_emotion']}' not in 7 classes"
    assert 0.0 <= pred["confidence"] <= 1.0, "Confidence score out of bounds [0, 1]"
    assert all(e in pred["distribution"] for e in EMOTIONS), "Missing emotion in distribution dictionary"
    
    dist_sum = sum(pred["distribution"].values())
    assert abs(dist_sum - 1.0) < 0.05, f"Distribution sum {dist_sum} does not equal 1.0"
    print(f"[PASS] Emotion Classifier test passed. (Predicted: {pred['predicted_emotion']}, Model: {pred['model_used']})")


def test_response_generator():
    print("Testing Empathetic Response Generation...")
    for emo in EMOTIONS:
        resp = response_generator.generate(
            emotion=emo,
            confidence=0.85,
            user_message="Test prompt",
            conversation_history=[]
        )
        assert len(resp["response_text"]) > 20, f"Response too short for emotion {emo}"
    print("[PASS] Response Generator test passed for all 7 emotions.")


def test_database_and_analytics():
    print("Testing Database and Analytics Engine...")
    session_id = db_service.create_session("test_student")
    assert session_id, "Session ID should not be empty"

    db_service.log_message(
        session_id=session_id,
        role="user",
        message="I feel overwhelmed.",
        detected_emotion="Fear",
        confidence_score=0.85,
        emotion_distribution={"Fear": 0.85, "Neutral": 0.15},
    )

    db_service.log_message(
        session_id=session_id,
        role="assistant",
        message="Take a deep breath with me.",
        detected_emotion="Fear",
        confidence_score=0.85,
    )

    msgs = db_service.get_messages(session_id)
    assert len(msgs) >= 2, "Messages not retrieved from database"

    df = analytics_engine.extract_dataframe(msgs)
    assert not df.empty, "Analytics DataFrame is empty"
    metrics = analytics_engine.get_summary_metrics(df)
    assert metrics["total_messages"] >= 1, "Metrics total message mismatch"
    print("[PASS] Database and Analytics test passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING NLP EMOTION CHATBOT PIPELINE INTEGRATION TESTS")
    print("=" * 60)
    test_preprocessing()
    test_safety_guard()
    test_emotion_classifier()
    test_response_generator()
    test_database_and_analytics()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (100% HEALTHY)")
    print("=" * 60)
