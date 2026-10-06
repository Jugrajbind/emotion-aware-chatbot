"""=============================================================================
Emotion-Aware AI Chatbot for Emotional Support Using NLP
BTech Mini Project - Web Application Entry Point (Streamlit)
Architecture: User Input -> Text Preprocessing -> Emotion Detection -> Safety Check -> Response Generation -> Chat UI -> Database
=============================================================================
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime

# Suppress Hugging Face internal module deprecation and path warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd

# Load internal project modules
from src.config import (
    BASE_DIR,
    EMOTIONS,
    EMOTION_COLORS,
    EMOTION_ICONS,
    CRISIS_RESOURCES,
    DISCLAIMER_TEXT,
    DEFAULT_MODEL_NAME,
)
from src.text_preprocessor import preprocessor
from src.emotion_classifier import emotion_classifier
from src.safety_guard import safety_guard
from src.response_generator import response_generator
from src.database import db_service
from src.analytics import analytics_engine

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING INJECTION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion-Aware AI Chatbot | NLP Mini Project",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS stylesheet
css_path = BASE_DIR / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = db_service.create_session("student_user")

if "messages" not in st.session_state:
    # Hydrate messages from database (Supabase or local SQLite)
    loaded_msgs = db_service.get_messages(st.session_state.session_id)
    if loaded_msgs:
        st.session_state.messages = loaded_msgs
    else:
        # Initial welcoming greeting
        welcome_text = (
            "Hello! I am your Emotion-Aware AI companion. I am here to listen with empathy, "
            "help you process your thoughts, and navigate your feelings in a safe, non-judgmental space.\n\n"
            "How are you feeling right now? Feel free to share whatever is on your mind."
        )
        st.session_state.messages = [
            {
                "id": "welcome-msg",
                "role": "assistant",
                "message": welcome_text,
                "detected_emotion": "Neutral",
                "confidence_score": 1.0,
                "is_safety_triggered": False,
                "created_at": datetime.now().isoformat(),
            }
        ]


# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_emotion_badge_html(emotion: str, confidence: float) -> str:
    """Renders a styled HTML badge pill for the detected emotion."""
    css_class = f"badge-{emotion.lower()}" if emotion else "badge-neutral"
    icon = EMOTION_ICONS.get(emotion, "💬")
    pct = int(confidence * 100) if confidence else 100
    return (
        f'<span class="emotion-badge {css_class}">'
        f'{icon} {emotion} &bull; {pct}%'
        f'</span>'
    )


def handle_user_submission(user_text: str):
    """Executes the full pipeline:

    User Input -> Text Preprocessing -> Emotion Detection -> Safety Check -> Response Generation -> Chat UI -> Database
    """
    if not user_text or not user_text.strip():
        return

    # STEP 1: Text Preprocessing
    prep_res = preprocessor.preprocess(user_text)
    clean_text = prep_res["clean_text"]

    # STEP 2: Safety Check Layer
    safety_res = safety_guard.analyze(clean_text)
    is_crisis = not safety_res["is_safe"]

    if is_crisis:
        # High-risk crisis intervention flow
        detected_emotion = "Fear"
        confidence = 0.99
        distribution = {"Fear": 0.99}
        bot_response = safety_res["helpline_message"]
        safety_cat = safety_res["category"]
    else:
        # STEP 3: Emotion Detection (Hugging Face Transformers + BERT)
        emotion_res = emotion_classifier.predict(clean_text)
        detected_emotion = emotion_res["predicted_emotion"]
        confidence = emotion_res["confidence"]
        distribution = emotion_res["distribution"]
        safety_cat = None

        # STEP 4: Empathetic Response Generation
        gen_res = response_generator.generate(
            emotion=detected_emotion,
            confidence=confidence,
            user_message=clean_text,
            conversation_history=st.session_state.messages,
        )
        bot_response = gen_res["response_text"]

    # STEP 5: Store turns in Database (Supabase with Local Fallback)
    user_record = db_service.log_message(
        session_id=st.session_state.session_id,
        role="user",
        message=user_text,
        detected_emotion=detected_emotion,
        confidence_score=confidence,
        emotion_distribution=distribution,
        is_safety_triggered=is_crisis,
        safety_category=safety_cat,
    )

    assistant_record = db_service.log_message(
        session_id=st.session_state.session_id,
        role="assistant",
        message=bot_response,
        detected_emotion=detected_emotion,
        confidence_score=confidence,
        is_safety_triggered=is_crisis,
        safety_category=safety_cat,
    )

    # STEP 6: Update UI Session State
    st.session_state.messages.append(user_record)
    st.session_state.messages.append(assistant_record)


# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & SYSTEM METRICS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 System Dashboard")

    # Supabase Connection Status
    if db_service.is_connected_to_supabase():
        st.markdown(
            '<div class="status-pill"><span class="dot-green"></span> <b>Supabase Cloud</b>: Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill"><span class="dot-amber"></span> <b>Storage</b>: Local SQLite Demo</div>',
            unsafe_allow_html=True,
        )

    st.caption("Add `SUPABASE_URL` and `SUPABASE_KEY` in `.env` for cloud sync.")
    st.divider()

    # Active Session Info
    st.markdown("#### 📌 Active Session")
    st.caption(f"**ID:** `{st.session_state.session_id[:8]}...`")
    
    # Live Emotion Summary in Sidebar
    df_history = analytics_engine.extract_dataframe(st.session_state.messages)
    metrics = analytics_engine.get_summary_metrics(df_history)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Total Turns", metrics["total_messages"])
    with col_s2:
        st.metric("Top Emotion", f"{metrics['dominant_icon']} {metrics['dominant_emotion']}")

    if metrics["safety_triggers"] > 0:
        st.warning(f"⚠️ {metrics['safety_triggers']} Safety Intervention(s)")

    st.divider()

    # Quick Conversation Reset Button
    if st.button("🔄 Start New Conversation", use_container_width=True, type="secondary"):
        new_sid = db_service.create_session("student_user")
        st.session_state.session_id = new_sid
        st.session_state.messages = []
        st.rerun()

    # Quick Viva Prompts (Demonstrating all 7 emotions + crisis safety trigger)
    st.markdown("#### 🧪 Quick Viva Demo Prompts")
    st.caption("Click to test any emotion classification:")
    
    demo_prompts = [
        ("✨ Joy", "I passed my final semester exams with distinction and got my dream job!"),
        ("🌧️ Sadness", "I feel so lonely and completely drained lately, like nothing is working out."),
        ("🔥 Anger", "It is so infuriating when teammates do zero work and take all the credit!"),
        ("🛡️ Fear", "I am having severe anxiety and panic about the presentation tomorrow."),
        ("⚡ Surprise", "I was completely astonished when my professor called my name for the award!"),
        ("🍃 Disgust", "The way they treated that homeless animal was utterly sickening and repulsive."),
        ("💬 Neutral", "The team meeting is scheduled for 3 PM tomorrow in the conference hall."),
        ("⚠️ Safety Crisis", "I feel like giving up and I just want to end my life."),
    ]

    for label, text in demo_prompts:
        if st.button(label, key=f"btn_{label}", use_container_width=True):
            handle_user_submission(text)
            st.rerun()

    st.divider()
    st.markdown("#### 🎓 Project Metadata")
    st.caption("**Project:** BTech Mini Project")
    st.caption("**Domain:** NLP / Affective Computing")
    st.caption(f"**NLP Engine:** Transformers / BERT")
    st.caption(f"**Classes (7):** Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral")


# -----------------------------------------------------------------------------
# 5. MAIN HEADER & ETHICAL DISCLAIMER BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">🧠 Emotion-Aware AI Chatbot for Emotional Support</div>
        <p class="hero-subtitle">
            Natural Language Processing (BERT / Transformers) &bull; Empathetic Response Generation &bull; Crisis Safety Guard &bull; Supabase Storage
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="disclaimer-card">
        🛡️ <b>Notice:</b> {DISCLAIMER_TEXT}
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 6. TABBED INTERFACE (Chat, Analytics Dashboard, Crisis Helplines, Architecture)
# -----------------------------------------------------------------------------
tab_chat, tab_analytics, tab_safety, tab_arch = st.tabs([
    "💬 Supportive Chat",
    "📊 Emotion Analytics",
    "🛡️ Crisis & Helplines",
    "🔬 System Architecture",
])

# =============================================================================
# TAB 1: SUPPORTIVE CHAT INTERFACE
# =============================================================================
with tab_chat:
    # Render Conversation History
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role = msg.get("role", "user")
            content = msg.get("message", "")
            emotion = msg.get("detected_emotion")
            confidence = msg.get("confidence_score")
            is_crisis = msg.get("is_safety_triggered", False)

            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)
                    if emotion:
                        badge_html = get_emotion_badge_html(emotion, confidence)
                        st.markdown(badge_html, unsafe_allow_html=True)

            elif role == "assistant":
                avatar_icon = "⚠️" if is_crisis else "🧠"
                with st.chat_message("assistant", avatar=avatar_icon):
                    if is_crisis:
                        st.markdown(
                            f'<div class="crisis-alert-box">{content}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(content)

    # Chat Input Box
    if user_prompt := st.chat_input("Type your thoughts or feelings here... (e.g. 'I feel anxious about my project')"):
        handle_user_submission(user_prompt)
        st.rerun()


# =============================================================================
# TAB 2: EMOTION ANALYTICS DASHBOARD
# =============================================================================
with tab_analytics:
    st.markdown("### 📊 Conversation Emotion Analytics")
    st.caption("Real-time NLP sentiment & affective classification breakdown across the session.")

    df_analytics = analytics_engine.extract_dataframe(st.session_state.messages)

    if df_analytics.empty:
        st.info("No conversation data yet. Send a message in the chat tab or test with sample prompts to view live charts!")
    else:
        summary = analytics_engine.get_summary_metrics(df_analytics)

        # 4 Metric Cards
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Total Messages</div>
                    <div class="metric-number">{summary['total_messages']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mcol2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Dominant Emotion</div>
                    <div class="metric-number">{summary['dominant_icon']} {summary['dominant_emotion']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mcol3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Avg BERT Confidence</div>
                    <div class="metric-number">{summary['avg_confidence']}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mcol4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Emotional Flow</div>
                    <div class="metric-number">{summary['emotional_stability']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Charts (Donut & Timeline)
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            fig_dist = analytics_engine.create_distribution_chart(df_analytics)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)

        with chart_col2:
            fig_timeline = analytics_engine.create_timeline_chart(df_analytics)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)

        # Tabular Data & Export Section
        st.markdown("#### 📋 Detailed Turn-by-Turn NLP Log")
        display_df = df_analytics.copy()
        display_df["confidence"] = display_df["confidence"].apply(lambda c: f"{int(c*100)}%")
        st.dataframe(
            display_df[["turn", "emotion", "confidence", "message", "is_safety_triggered"]],
            use_container_width=True,
            hide_index=True,
        )

        # Export Buttons
        exp_col1, exp_col2, _ = st.columns([1, 1, 2])
        with exp_col1:
            csv_data = df_analytics.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv_data,
                file_name=f"emotion_session_{st.session_state.session_id[:8]}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_col2:
            json_data = json.dumps(st.session_state.messages, indent=2).encode("utf-8")
            st.download_button(
                label="📥 Export Full Chat (JSON)",
                data=json_data,
                file_name=f"chat_logs_{st.session_state.session_id[:8]}.json",
                mime="application/json",
                use_container_width=True,
            )


# =============================================================================
# TAB 3: SAFETY GUARD & MENTAL HEALTH CRISIS DIRECTORY
# =============================================================================
with tab_safety:
    st.markdown("### 🛡️ Safety Layer & Professional Support Directory")
    st.info(
        "**Safety Protocol:** When the chatbot detects crisis triggers (such as suicidal ideation, "
        "self-harm phrases, or severe abuse), the conversation flow is intercepted immediately. "
        "Normal algorithmic replies are replaced with de-escalation guidance and official helpline contacts."
    )

    for region, resources in CRISIS_RESOURCES.items():
        st.markdown(f"#### 📍 {region}")
        r_cols = st.columns(len(resources))
        for idx, res in enumerate(resources):
            with r_cols[idx % len(r_cols)]:
                st.markdown(
                    f"""
                    <div class="metric-box" style="text-align: left; height: 160px;">
                        <b style="color: #60a5fa;">{res['name']}</b><br>
                        <span style="font-size: 1.1rem; color: #f8fafc; font-weight: 600;">{res['contact']}</span><br>
                        <span style="font-size: 0.8rem; color: #94a3b8;">{res['desc']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# TAB 4: SYSTEM ARCHITECTURE & NLP PIPELINE (FOR BTECH VIVA)
# =============================================================================
with tab_arch:
    st.markdown("### 🔬 System Architecture & NLP Technical Details")
    st.markdown(
        """
        The architecture follows a strict, safety-first sequential pipeline designed for affective computing:
        """
    )

    st.code(
        """
┌─────────────────┐       ┌────────────────────┐       ┌────────────────────────┐
│   User Input    │ ───>  │ Text Preprocessing │ ───>  │ Safety Risk Classifier │
└─────────────────┘       └────────────────────┘       └────────────────────────┘
                                                                    │
                                                ┌───────────────────┴───────────────────┐
                                                │                                       │
                                         [Risk Detected]                         [Safe Dialogue]
                                                │                                       │
                                                ▼                                       ▼
                                  ┌───────────────────────────┐           ┌───────────────────────────┐
                                  │ Emergency Crisis Response │           │ BERT Emotion Classifier   │
                                  │ (Helplines & Resources)   │           │ (7 Affective Classes)     │
                                  └───────────────────────────┘           └───────────────────────────┘
                                                │                                       │
                                                │                                       ▼
                                                │                         ┌───────────────────────────┐
                                                │                         │ Empathetic Response Engine│
                                                │                         └───────────────────────────┘
                                                │                                       │
                                                └───────────────────┬───────────────────┘
                                                                    │
                                                                    ▼
                                                      ┌───────────────────────────┐
                                                      │  Streamlit UI & Supabase  │
                                                      │  PostgreSQL Persistence   │
                                                      └───────────────────────────┘
        """,
        language="text",
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### 1. NLP Preprocessing & Tokenization")
        st.markdown(
            """
            - **Contraction Expansion:** Normalizes informal text (`can't` -> `cannot`, `I'm` -> `I am`).
            - **Affective Emoji Translation:** Translates graphical symbols (e.g., `😢` -> `sad`, `😊` -> `joy`) into sentiment tokens.
            - **Syntactic Normalization:** Compresses repeated punctuation while preserving emotional cues.
            """
        )

        st.markdown("#### 2. BERT Emotion Classification")
        st.markdown(
            """
            - **Model Family:** Bidirectional Encoder Representations from Transformers (BERT / DistilRoBERTa).
            - **Classes (7):** `Joy`, `Sadness`, `Anger`, `Fear`, `Surprise`, `Disgust`, `Neutral`.
            - **Softmax Head:** Generates full probability distribution across all 7 classes.
            """
        )

    with col_t2:
        st.markdown("#### 3. Empathetic Dialogue Generation")
        st.markdown(
            """
            - **Affective Mirroring:** Validates the user's emotion without judgment.
            - **Reflective Inquiry:** Encourages gentle elaboration.
            - **Grounding Interventions:** Delivers exercises (e.g. 4-4-4-4 Box Breathing for Fear/Anger, 5-4-3-2-1 Sensory Grounding for Sadness).
            """
        )

        st.markdown("#### 4. Database & Cloud Architecture")
        st.markdown(
            """
            - **Supabase Cloud:** PostgreSQL database storing session UUIDs, chat logs, JSONB emotion distributions, and safety incidents.
            - **Row Level Security (RLS):** Granular access control policies.
            - **Local Fallback:** Seamless SQLite storage if running offline or in demo sandbox.
            """
        )
