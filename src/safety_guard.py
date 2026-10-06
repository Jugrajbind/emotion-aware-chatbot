"""Safety and Crisis Intervention Guard for Emotion-Aware Chatbot.

Detects high-risk, self-harm, suicidal ideation, or crisis-related statements.
When triggered, it bypasses regular chatbot output and displays compassionate,
de-escalating guidance along with official professional mental healthcare helplines.
"""

import re
from typing import Dict, Any, List, Optional
from src.config import SAFETY_ENABLED, CRISIS_RESOURCES, DISCLAIMER_TEXT


class SafetyGuard:
    """Multi-tiered regex and keyword safety analyzer for user input."""

    # High-risk crisis phrases and patterns (Suicidal ideation, self-harm, severe harm)
    CRITICAL_PATTERNS: List[re.Pattern] = [
        re.compile(r"\b(kill|hang|shoot|drown|poison|end|take|ending)\s+(my\s*self|me|my\s*life)\b", re.IGNORECASE),
        re.compile(r"\b(want|wish|going)\s+to\s+(die|end\s+it\s+all|kill\s+myself|end\s+my\s*life)\b", re.IGNORECASE),
        re.compile(r"\b(suicide|suicidal|suiciding)\b", re.IGNORECASE),
        re.compile(r"\b(better\s+off\s+dead|no\s+reason\s+to\s+live)\b", re.IGNORECASE),
        re.compile(r"\b(do\s*not\s+want\s+to\s+live|cannot\s+live\s+anymore|cant\s+live\s+anymore|cannot\s+go\s+on|cant\s+go\s+on)\b", re.IGNORECASE),
        re.compile(r"\b(cut|cutting|burn|harm|hurting)\s+my\s*self\b", re.IGNORECASE),
        re.compile(r"\b(take\s+an\s+overdose|swallow\s+all\s+my\s+pills)\b", re.IGNORECASE),
        re.compile(r"\b(goodbye\s+cruel\s+world|final\s+goodbye|leaving\s+this\s+world)\b", re.IGNORECASE),
        re.compile(r"\b(nobody\s+would\s+care\s+if\s+i\s+died)\b", re.IGNORECASE),
    ]

    # Moderate distress phrases requiring empathetic caution
    HIGH_DISTRESS_PATTERNS: List[re.Pattern] = [
        re.compile(r"\b(feeling\s+hopeless|lost\s+all\s+hope|giving\s+up\s+on\s+life)\b", re.IGNORECASE),
        re.compile(r"\b(cannot\s+take\s+this\s+pain|unbearable\s+pain)\b", re.IGNORECASE),
        re.compile(r"\b(abused|being\s+hit|domestic\s+violence)\b", re.IGNORECASE),
    ]

    def __init__(self, enabled: bool = SAFETY_ENABLED):
        self.enabled = enabled

    def analyze(self, text: str) -> Dict[str, Any]:
        """Scans the text for crisis indicators.

        Returns:
            Dict containing:
                - 'is_safe': bool (False if crisis detected)
                - 'severity': 'CRITICAL' | 'HIGH' | 'SAFE'
                - 'category': detected category or None
                - 'helpline_message': formatted emergency message with contacts
        """
        if not self.enabled or not text:
            return {
                "is_safe": True,
                "severity": "SAFE",
                "category": None,
                "helpline_message": None,
            }

        cleaned = text.strip()

        # Check for Critical Suicide / Self-Harm Patterns
        for pattern in self.CRITICAL_PATTERNS:
            if pattern.search(cleaned):
                return {
                    "is_safe": False,
                    "severity": "CRITICAL",
                    "category": "Self-Harm or Suicidal Ideation",
                    "helpline_message": self.generate_crisis_response(severity="CRITICAL"),
                }

        # Check for High Distress Patterns
        for pattern in self.HIGH_DISTRESS_PATTERNS:
            if pattern.search(cleaned):
                return {
                    "is_safe": False,
                    "severity": "HIGH",
                    "category": "Severe Distress / Abuse",
                    "helpline_message": self.generate_crisis_response(severity="HIGH"),
                }

        return {
            "is_safe": True,
            "severity": "SAFE",
            "category": None,
            "helpline_message": None,
        }

    def generate_crisis_response(self, severity: str = "CRITICAL") -> str:
        """Constructs an immediate de-escalation response with verified crisis helplines."""
        header = (
            "⚠️ **I hear how much pain you are in, and your life and safety matter deeply.**\n\n"
            "While I am an AI conversational companion, **I cannot provide crisis care or replace human medical support**. "
            "Please connect immediately with someone who can support you through this exact moment. You do not have to carry this alone."
        )

        helpline_sections = ["\n### 📞 Free, Confidential 24/7 Crisis Helplines:"]
        
        for region, resources in CRISIS_RESOURCES.items():
            helpline_sections.append(f"\n**{region}:**")
            for res in resources:
                helpline_sections.append(f"- **{res['name']}**: `{res['contact']}` — *{res['desc']}*")

        helpline_sections.append(
            "\n💡 *If you are in immediate physical danger, please contact your local emergency services (like 112, 911, or 999) "
            "or proceed to the nearest hospital emergency room.*"
        )

        return f"{header}\n" + "\n".join(helpline_sections)


# Global safety instance
safety_guard = SafetyGuard()
