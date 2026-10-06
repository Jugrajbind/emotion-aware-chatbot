"""Database Management Layer for Supabase with Local SQLite Fallback.

Provides persistent storage for:
- Chat sessions
- User inputs & AI responses
- Detected emotions, confidence scores, and safety flags
- Automatic fallback to local SQLite when Supabase credentials are not configured.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from src.config import SUPABASE_URL, SUPABASE_KEY, BASE_DIR

logger = logging.getLogger(__name__)

LOCAL_DB_PATH = BASE_DIR / "local_chat_history.db"


class DatabaseService:
    """Manages chat persistence with Supabase primary client and SQLite fallback."""

    def __init__(self):
        self.supabase_client = None
        self.is_cloud = False
        self._init_connection()
        self._init_local_db()

    def _init_connection(self):
        """Attempts to initialize Supabase client if valid credentials are present."""
        if SUPABASE_URL and SUPABASE_KEY and "supabase.co" in SUPABASE_URL:
            try:
                from supabase import create_client, Client
                self.supabase_client: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY)
                # Quick ping test
                self.is_cloud = True
                logger.info("Successfully initialized Supabase cloud client.")
            except Exception as e:
                logger.warning(f"Failed to connect to Supabase: {e}. Falling back to local storage.")
                self.supabase_client = None
                self.is_cloud = False
        else:
            logger.info("Supabase credentials not configured. Using local SQLite mode.")
            self.is_cloud = False

    def _init_local_db(self):
        """Initializes local SQLite schema mirroring Supabase for offline/demo operation."""
        try:
            with sqlite3.connect(LOCAL_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        session_title TEXT,
                        user_identifier TEXT,
                        created_at TEXT,
                        last_active_at TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_logs (
                        id TEXT PRIMARY KEY,
                        session_id TEXT,
                        role TEXT,
                        message TEXT,
                        detected_emotion TEXT,
                        confidence_score REAL,
                        emotion_distribution TEXT,
                        is_safety_triggered INTEGER,
                        safety_category TEXT,
                        created_at TEXT,
                        FOREIGN KEY(session_id) REFERENCES sessions(id)
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing local SQLite database: {e}")

    def is_connected_to_supabase(self) -> bool:
        """Returns True if connected to Supabase cloud."""
        return self.is_cloud and self.supabase_client is not None

    def create_session(self, user_identifier: str = "anonymous_student") -> str:
        """Creates a new session record and returns its UUID."""
        session_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        if self.is_connected_to_supabase():
            try:
                res = self.supabase_client.table("sessions").insert({
                    "id": session_id,
                    "session_title": f"Session - {datetime.now().strftime('%b %d, %H:%M')}",
                    "user_identifier": user_identifier,
                    "created_at": now_iso,
                    "last_active_at": now_iso,
                }).execute()
                if res.data:
                    return session_id
            except Exception as e:
                logger.warning(f"Supabase create_session error: {e}. Storing locally.")

        # Local SQLite storage fallback
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (id, session_title, user_identifier, created_at, last_active_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, f"Session - {datetime.now().strftime('%b %d, %H:%M')}", user_identifier, now_iso, now_iso),
            )
            conn.commit()

        return session_id

    def log_message(
        self,
        session_id: str,
        role: str,
        message: str,
        detected_emotion: Optional[str] = None,
        confidence_score: Optional[float] = None,
        emotion_distribution: Optional[Dict[str, float]] = None,
        is_safety_triggered: bool = False,
        safety_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Logs a conversation turn to the database."""
        msg_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        distribution_json = emotion_distribution or {}

        record = {
            "id": msg_id,
            "session_id": session_id,
            "role": role,
            "message": message,
            "detected_emotion": detected_emotion,
            "confidence_score": round(confidence_score, 4) if confidence_score else None,
            "emotion_distribution": distribution_json,
            "is_safety_triggered": is_safety_triggered,
            "safety_category": safety_category,
            "created_at": now_iso,
        }

        # Attempt Supabase cloud insertion
        if self.is_connected_to_supabase():
            try:
                self.supabase_client.table("chat_logs").insert(record).execute()
                # Update session timestamp
                self.supabase_client.table("sessions").update({"last_active_at": now_iso}).eq("id", session_id).execute()
                return record
            except Exception as e:
                logger.warning(f"Supabase log_message error: {e}. Fallback to local.")

        # Local SQLite insertion
        try:
            with sqlite3.connect(LOCAL_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO chat_logs 
                    (id, session_id, role, message, detected_emotion, confidence_score, emotion_distribution, is_safety_triggered, safety_category, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        msg_id,
                        session_id,
                        role,
                        message,
                        detected_emotion,
                        confidence_score,
                        json.dumps(distribution_json),
                        1 if is_safety_triggered else 0,
                        safety_category,
                        now_iso,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Local database insertion failed: {e}")

        return record

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all chat messages for a given session."""
        if self.is_connected_to_supabase():
            try:
                res = (
                    self.supabase_client.table("chat_logs")
                    .select("*")
                    .eq("session_id", session_id)
                    .order("created_at", desc=False)
                    .execute()
                )
                if res.data:
                    return res.data
            except Exception as e:
                logger.warning(f"Supabase get_messages error: {e}. Using local storage.")

        # Local SQLite fallback
        results = []
        try:
            with sqlite3.connect(LOCAL_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM chat_logs WHERE session_id = ? ORDER BY created_at ASC",
                    (session_id,),
                )
                rows = cursor.fetchall()
                for row in rows:
                    item = dict(row)
                    item["is_safety_triggered"] = bool(item.get("is_safety_triggered", 0))
                    try:
                        item["emotion_distribution"] = json.loads(item.get("emotion_distribution") or "{}")
                    except Exception:
                        item["emotion_distribution"] = {}
                    results.append(item)
        except Exception as e:
            logger.error(f"Local query error: {e}")

        return results

    def clear_session(self, session_id: str) -> bool:
        """Deletes chat logs for a specific session."""
        if self.is_connected_to_supabase():
            try:
                self.supabase_client.table("chat_logs").delete().eq("session_id", session_id).execute()
            except Exception as e:
                logger.warning(f"Supabase delete error: {e}")

        try:
            with sqlite3.connect(LOCAL_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_logs WHERE session_id = ?", (session_id,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Local delete error: {e}")
            return False


# Global database service instance
db_service = DatabaseService()
