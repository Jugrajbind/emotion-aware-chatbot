-- =============================================================================
-- Supabase Schema for "Emotion-Aware AI Chatbot for Emotional Support Using NLP"
-- Run this script in your Supabase SQL Editor: Dashboard -> SQL Editor -> New Query
-- =============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- 1. SESSIONS TABLE
-- Stores individual chat sessions and client identity metadata
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_title TEXT DEFAULT 'Emotional Support Session',
    user_identifier TEXT DEFAULT 'anonymous_user',
    created_at TIMESTAMPTZ WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    last_active_at TIMESTAMPTZ WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- -----------------------------------------------------------------------------
-- 2. CHAT LOGS TABLE
-- Stores conversation turns, detected emotions, confidence scores, and safety flags
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.chat_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    message TEXT NOT NULL,
    detected_emotion TEXT CHECK (
        detected_emotion IN ('Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Neutral') OR detected_emotion IS NULL
    ),
    confidence_score NUMERIC(5, 4),
    emotion_distribution JSONB DEFAULT '{}'::jsonb,
    is_safety_triggered BOOLEAN DEFAULT FALSE NOT NULL,
    safety_category TEXT,
    created_at TIMESTAMPTZ WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- -----------------------------------------------------------------------------
-- 3. INDEXES FOR PERFORMANCE
-- Fast lookups by session and timestamps for dashboard generation
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id ON public.chat_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_logs_created_at ON public.chat_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_logs_detected_emotion ON public.chat_logs(detected_emotion);
CREATE INDEX IF NOT EXISTS idx_chat_logs_safety ON public.chat_logs(is_safety_triggered);

-- -----------------------------------------------------------------------------
-- 4. ANALYTICS VIEW
-- Real-time aggregation of emotion distribution per session
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.session_emotion_summary AS
SELECT 
    session_id,
    detected_emotion,
    COUNT(*) AS message_count,
    ROUND(AVG(confidence_score)::numeric, 4) AS avg_confidence,
    SUM(CASE WHEN is_safety_triggered THEN 1 ELSE 0 END) AS safety_incidents
FROM public.chat_logs
WHERE role = 'user' AND detected_emotion IS NOT NULL
GROUP BY session_id, detected_emotion;

-- -----------------------------------------------------------------------------
-- 5. ROW LEVEL SECURITY (RLS) POLICIES
-- For mini-project development with anon key access
-- -----------------------------------------------------------------------------
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_logs ENABLE ROW LEVEL SECURITY;

-- Allow anonymous inserts and selects for demo purposes
CREATE POLICY "Allow public read access to sessions"
ON public.sessions FOR SELECT
TO anon, authenticated
USING (true);

CREATE POLICY "Allow public insert access to sessions"
ON public.sessions FOR INSERT
TO anon, authenticated
WITH CHECK (true);

CREATE POLICY "Allow public update access to sessions"
ON public.sessions FOR UPDATE
TO anon, authenticated
USING (true);

CREATE POLICY "Allow public read access to chat_logs"
ON public.chat_logs FOR SELECT
TO anon, authenticated
USING (true);

CREATE POLICY "Allow public insert access to chat_logs"
ON public.chat_logs FOR INSERT
TO anon, authenticated
WITH CHECK (true);

CREATE POLICY "Allow public delete access to chat_logs"
ON public.chat_logs FOR DELETE
TO anon, authenticated
USING (true);

-- -----------------------------------------------------------------------------
-- 6. SAMPLE TEST DATA (Optional Seed)
-- -----------------------------------------------------------------------------
INSERT INTO public.sessions (id, session_title, user_identifier)
VALUES ('00000000-0000-0000-0000-000000000001', 'Sample Demo Session', 'demo_student')
ON CONFLICT (id) DO NOTHING;

INSERT INTO public.chat_logs (session_id, role, message, detected_emotion, confidence_score, emotion_distribution, is_safety_triggered)
VALUES 
('00000000-0000-0000-0000-000000000001', 'user', 'I am feeling overwhelmed with my upcoming college exams and assignments.', 'Fear', 0.8845, '{"Fear": 0.8845, "Sadness": 0.0712, "Neutral": 0.0443}', FALSE),
('00000000-0000-0000-0000-000000000001', 'assistant', 'It is completely valid to feel overwhelmed when deadlines pile up. Take a slow, deep breath with me right now. Let us break this down together step by step.', 'Fear', 0.8845, '{}', FALSE),
('00000000-0000-0000-0000-000000000001', 'user', 'Finally submitted my project report today, feel so relieved!', 'Joy', 0.9412, '{"Joy": 0.9412, "Neutral": 0.0381, "Surprise": 0.0207}', FALSE),
('00000000-0000-0000-0000-000000000001', 'assistant', 'That is wonderful news! You worked hard for it, and that sense of relief is well-deserved. Celebrate this win!', 'Joy', 0.9412, '{}', FALSE)
ON CONFLICT (id) DO NOTHING;
