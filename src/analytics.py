"""Conversation Emotion Analytics and Dashboard Visualizations.

Computes metrics and creates Plotly visualizations using Pandas and NumPy:
- Emotion frequency and percentage distribution
- Turn-by-turn emotion progression trajectory
- Confidence score analytics
- Summary statistics for the dashboard
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import EMOTIONS, EMOTION_COLORS, EMOTION_ICONS


class EmotionAnalytics:
    """Analyzes emotion data stored in conversation history and produces visual charts."""

    @staticmethod
    def extract_dataframe(messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts raw message records into a structured Pandas DataFrame."""
        user_msgs = [m for m in messages if m.get("role") == "user" and m.get("detected_emotion")]
        
        if not user_msgs:
            return pd.DataFrame(columns=[
                "turn", "message", "emotion", "confidence", "is_safety_triggered", "timestamp"
            ])

        records = []
        for idx, msg in enumerate(user_msgs, start=1):
            records.append({
                "turn": idx,
                "message": msg.get("message", "")[:40] + "..." if len(msg.get("message", "")) > 40 else msg.get("message", ""),
                "emotion": msg.get("detected_emotion", "Neutral"),
                "confidence": float(msg.get("confidence_score", 0.0) or 0.0),
                "is_safety_triggered": bool(msg.get("is_safety_triggered", False)),
                "timestamp": msg.get("created_at", ""),
            })

        return pd.DataFrame(records)

    @staticmethod
    def get_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates key statistical metrics across the session."""
        if df.empty:
            return {
                "total_messages": 0,
                "dominant_emotion": "None",
                "dominant_icon": "💬",
                "avg_confidence": 0.0,
                "safety_triggers": 0,
                "emotional_stability": "N/A",
            }

        total_msgs = len(df)
        counts = df["emotion"].value_counts()
        dominant_emotion = counts.index[0]
        dominant_icon = EMOTION_ICONS.get(dominant_emotion, "💬")
        avg_confidence = round(float(df["confidence"].mean()) * 100, 1)
        safety_triggers = int(df["is_safety_triggered"].sum())

        # Emotion diversity index (number of unique emotions experienced)
        unique_emotions = df["emotion"].nunique()
        stability = "Steady" if unique_emotions <= 2 else "Dynamic" if unique_emotions <= 4 else "Fluctuating"

        return {
            "total_messages": total_msgs,
            "dominant_emotion": dominant_emotion,
            "dominant_icon": dominant_icon,
            "avg_confidence": avg_confidence,
            "safety_triggers": safety_triggers,
            "emotional_stability": stability,
        }

    @staticmethod
    def create_distribution_chart(df: pd.DataFrame) -> Optional[go.Figure]:
        """Builds a modern donut chart showing the percentage breakdown of emotions."""
        if df.empty:
            return None

        counts = df["emotion"].value_counts().reset_index()
        counts.columns = ["emotion", "count"]

        # Ensure consistent color mapping
        colors = [EMOTION_COLORS.get(e, "#6B7280") for e in counts["emotion"]]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=counts["emotion"],
                    values=counts["count"],
                    hole=0.55,
                    marker=dict(colors=colors, line=dict(color="#1e293b", width=2)),
                    textinfo="label+percent",
                    hoverinfo="label+value+percent",
                    textfont=dict(size=13, family="Inter, sans-serif"),
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text="<b>Emotion Distribution</b>",
                font=dict(size=16, color="#f8fafc"),
                x=0.05,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=10, r=10),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(color="#94a3b8"),
            ),
        )
        return fig

    @staticmethod
    def create_timeline_chart(df: pd.DataFrame) -> Optional[go.Figure]:
        """Plots turn-by-turn emotion progression with confidence levels."""
        if df.empty:
            return None

        fig = go.Figure()

        # Add confidence bar trace
        fig.add_trace(
            go.Bar(
                x=[f"Turn {t}" for t in df["turn"]],
                y=df["confidence"],
                text=[f"{e} ({int(c*100)}%)" for e, c in zip(df["emotion"], df["confidence"])],
                textposition="auto",
                marker=dict(
                    color=[EMOTION_COLORS.get(e, "#6B7280") for e in df["emotion"]],
                    line=dict(color="#ffffff", width=1),
                    opacity=0.88,
                ),
                name="Confidence Level",
            )
        )

        fig.update_layout(
            title=dict(
                text="<b>Emotional Trajectory Across Conversation</b>",
                font=dict(size=16, color="#f8fafc"),
                x=0.05,
            ),
            yaxis=dict(
                title="Classification Confidence",
                range=[0, 1.05],
                tickformat=".0%",
                gridcolor="#334155",
                color="#94a3b8",
            ),
            xaxis=dict(
                title="Dialogue Turn",
                gridcolor="#334155",
                color="#94a3b8",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=30, l=40, r=20),
            showlegend=False,
        )
        return fig


analytics_engine = EmotionAnalytics()
