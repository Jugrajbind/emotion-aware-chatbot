# Emotion-Aware AI Chatbot for Emotional Support Using NLP

An affective computing web application engineered for supportive dialogue, emotion classification across 7 target states, empathetic response generation, crisis safety interception, real-time analytics, and persistent cloud storage.

> **Academic Context:** Developed as a BTech Mini Project in Artificial Intelligence / Natural Language Processing.
> 
> **Important Ethical Disclaimer:** This system is an AI conversational companion created for educational, exploratory, and supportive listening purposes. It is **NOT** a licensed psychotherapist, clinical diagnostic tool, or replacement for professional mental healthcare. If you or someone you know is in crisis, please consult professional helplines immediately.

---

## 🌟 Key Features

1. **7-Class Affective NLP Classifier:**
   - Detects **Joy**, **Sadness**, **Anger**, **Fear**, **Surprise**, **Disgust**, and **Neutral**.
   - Outputs dominant class and full softmax probability distribution.
   - Built on PyTorch and Hugging Face Transformers (`DistilRoBERTa` / `BERT`).
2. **Empathetic Dialogue Engine:**
   - Multi-tiered response architecture (Validation &rarr; Reflection &rarr; Affirmation &rarr; Grounding).
   - Incorporates psychological grounding exercises (4-4-4-4 Box Breathing for Fear/Anger, 5-4-3-2-1 Sensory Grounding for Sadness).
   - Contextual tracking of emotional trajectory across dialogue turns.
3. **Safety Layer & Crisis Intervention:**
   - Real-time screening for suicidal ideation, self-harm, and severe abuse language.
   - Overrides standard responses during crisis events with immediate, de-escalating compassionate messaging and verified 24/7 national/international helplines (Tele-MANAS, KIRAN, 988 Lifeline, etc.).
4. **Interactive Emotion Analytics Dashboard:**
   - Donut chart of conversation emotion breakdown.
   - Turn-by-turn emotional trajectory timeline.
   - Key session metrics: total turns, dominant emotion, average BERT confidence, and stability index.
   - Data export in both CSV and JSON formats.
5. **Secure Cloud Database (Supabase) with Local Fallback:**
   - Full PostgreSQL integration using Supabase with Row Level Security (RLS).
   - Automatic local SQLite fallback when running offline or in demo mode without credentials.
6. **Modern, Responsive Web UI:**
   - Streamlit interface styled with custom CSS (glassmorphism, badge pills, dark aesthetic, responsive cards).
   - Quick viva demo prompt buttons for instant examiner demonstration.

---

## 🏗️ System Architecture

```text
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
```

---

## 📂 Project Structure

```
emotion_aware_chatbot/
│
├── .env                       # Local environment variables (Supabase keys, model options)
├── .env.example               # Environment template for repository
├── .gitignore                 # Git ignore rules for caches and secrets
├── requirements.txt           # Python dependency specifications
├── schema.sql                 # Supabase PostgreSQL schema, tables, and RLS policies
├── app.py                     # Streamlit frontend & application orchestration
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Central configuration, emotion palettes, helplines
│   ├── text_preprocessor.py   # NLP cleaning, contraction expansion, emoji mapping
│   ├── emotion_classifier.py  # PyTorch BERT architecture & Transformers inference pipeline
│   ├── safety_guard.py        # Crisis detection patterns, risk scoring, helpline generation
│   ├── response_generator.py  # Empathetic dialogue generation & grounding exercises
│   ├── database.py            # Supabase cloud client with seamless local SQLite fallback
│   └── analytics.py           # Pandas metrics calculation & Plotly chart builders
│
├── assets/
│   └── style.css              # Custom dark-mode glassmorphic styling
│
├── tests/
│   └── test_pipeline.py       # Integration & unit test suite
│
└── docs/
    └── PROJECT_REPORT.md      # Comprehensive BTech academic project documentation
```

---

## ⚡ Quick Start: Running Locally

### 1. Prerequisites
- Python 3.10 or higher installed
- Git installed

### 2. Clone or Navigate to the Project
```bash
cd emotion_aware_chatbot
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables (Optional for Local Demo)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
> **Note:** The app works right out of the box even without Supabase credentials by utilizing the automatic local SQLite fallback mode.

### 6. Run the Test Suite
```bash
python tests/test_pipeline.py
```

### 7. Launch the Streamlit Application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 🗄️ Supabase Cloud Database Setup

To enable cloud persistence across devices:

1. Sign up or log into [Supabase](https://supabase.com/).
2. Create a new project (e.g., `emotion-chatbot-db`).
3. In the left sidebar, navigate to the **SQL Editor**.
4. Click **New Query**, open [`schema.sql`](file:///C:/Users/jugra/.gemini/antigravity-ide/scratch/emotion_aware_chatbot/schema.sql) from this repository, copy its contents, paste them into the editor, and click **Run**.
5. Navigate to **Project Settings &rarr; API**.
6. Copy your **Project URL** and **Project API Key (`anon` / `public`)**.
7. Update your `.env` file:
   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```
8. Restart the Streamlit app. The sidebar badge will now indicate: `🟢 Supabase Cloud: Connected`.

---

## 🚀 Deployment Instructions

### Deploying to Streamlit Community Cloud (Recommended)

1. Push this project to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Emotion-Aware AI Chatbot"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **New app**.
4. Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Under **Advanced settings &rarr; Secrets**, add your Supabase credentials:
   ```toml
   SUPABASE_URL = "https://your-project-id.supabase.co"
   SUPABASE_KEY = "your-anon-key"
   ENABLE_SAFETY_GUARD = "true"
   ```
6. Click **Deploy!** Your application will be live with a public URL for your project review!

---

## 🧪 Viva Demonstration Guide

During your project viva or presentation, follow this sequence:
1. **Show the UI & Architecture Tab:** Explain the sequential 7-step pipeline.
2. **Demonstrate Sample Emotions:** Click the quick viva prompt buttons in the sidebar:
   - Joy: Shows positive reinforcement.
   - Fear/Sadness: Shows empathetic validation and triggers interactive grounding techniques.
   - Anger: Demonstrates de-escalation breathing exercises.
3. **Demonstrate Safety Layer Interception:** Click the `⚠️ Safety Crisis` prompt to show how high-risk language immediately suppresses normal output and surfaces verified crisis hotlines.
4. **Show Analytics Dashboard:** Switch to the `📊 Emotion Analytics` tab to show real-time donut charts, timeline progression, and export capabilities.
5. **Demonstrate Database Records:** Show the `local_chat_history.db` or Supabase Table Editor showing encrypted session records with emotion distributions.

---

## 📜 License
Developed under the MIT License for educational and academic submission purposes.
