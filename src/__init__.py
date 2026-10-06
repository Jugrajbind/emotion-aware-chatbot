"""Emotion-Aware AI Chatbot for Emotional Support Using NLP.
Core package initialization & Streamlit Cloud entry-point proxy.
"""

from pathlib import Path

__version__ = "1.0.0"

# Fallback: If Streamlit Cloud runs src/__init__.py as the main file, proxy execution to app.py
_app_file = Path(__file__).resolve().parent.parent / "app.py"
if _app_file.exists():
    with open(_app_file, "r", encoding="utf-8") as _f:
        _code = compile(_f.read(), str(_app_file), "exec")
        exec(_code, globals())
