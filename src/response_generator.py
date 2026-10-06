"""Empathetic Response Generation Engine for Emotion-Aware Chatbot.

Generates psychologically grounded, supportive responses tailored to:
1. Detected emotion (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral)
2. Confidence score
3. Contextual conversational trajectory (history of emotional transitions)
4. Active coping mechanisms (Grounding, Box Breathing, Savoring, Reframing)
"""

import random
from typing import Dict, List, Optional, Any


class ResponseGenerator:
    """Produces empathetic, non-clinical supportive dialogues aligned with emotional state."""

    # Rich empathetic response templates categorized by emotion
    EMOTION_RESPONSES: Dict[str, Dict[str, List[str]]] = {
        "Joy": {
            "validation": [
                "That brings such a genuine smile to hear! Your energy and happiness are truly radiant.",
                "I am so thrilled to hear that with you! Moments like this are so valuable to cherish.",
                "What fantastic news! It sounds like you are experiencing a wonderful burst of positivity.",
            ],
            "reflection": [
                "What was the highlight that made this moment feel so meaningful?",
                "How are you planning to celebrate or savor this positive milestone?",
                "Who is the first person you want to share this good news with?",
            ],
            "affirmation": [
                "You earned this joy. Allow yourself to soak in every bit of this good feeling!",
                "Take a mental snapshot of this moment so you can revisit this warmth whenever you need it.",
            ]
        },
        "Sadness": {
            "validation": [
                "I hear how heavy your heart feels right now. It is completely okay to feel sad and let those feelings exist.",
                "Thank you for sharing that with me. Please know you do not have to put on a brave face here.",
                "I am sitting with you in this difficult feeling. Your pain is valid, and it matters.",
            ],
            "reflection": [
                "Would you like to tell me more about what is weighing on you most heavily today?",
                "Is there someone close to you who brings you comfort when things feel this gray?",
                "Do you feel like resting and having quiet space, or would it help to talk through the feelings?",
            ],
            "affirmation": [
                "Please be exceedingly gentle with yourself today. Even getting through a hard day is an act of resilience.",
                "Remember: emotions are like weather patterns. As heavy as this storm feels, it will not last forever.",
            ]
        },
        "Anger": {
            "validation": [
                "I can sense the deep frustration and heat in what you are experiencing. It sounds genuinely unfair.",
                "Your frustration makes total sense given what happened. It is completely healthy to acknowledge anger.",
                "I hear you loud and clear. When boundaries get crossed or expectations shatter, feeling angry is natural.",
            ],
            "reflection": [
                "What part of this situation felt the most unfair or out of your control?",
                "Would it help to vent everything out uncensored, or would you like to explore ways to discharge the tension?",
            ],
            "affirmation": [
                "Let us take a slow breath together. Unclench your jaw, drop your shoulders, and give yourself permission to pause.",
                "Anger often protects what we deeply care about. Your boundaries and self-respect are important.",
            ]
        },
        "Fear": {
            "validation": [
                "I can feel how tense and unsettled you are feeling. Anxiety and fear can feel so overwhelming inside your body.",
                "It is completely understandable to feel frightened or anxious right now. Take comfort: you are safe here in this moment.",
                "Uncertainty can make our thoughts race fast. I am right here with you, and we will take things one second at a time.",
            ],
            "reflection": [
                "What is the scariest thought that is looping in your mind right now?",
                "Can you feel your feet grounded on the floor beneath you? Let us slow down our tempo together.",
            ],
            "affirmation": [
                "Try the 5-4-3-2-1 technique: notice 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 slow breath.",
                "You do not need to figure out the whole future right now. Just focus on the very next ten minutes.",
            ]
        },
        "Surprise": {
            "validation": [
                "Whoa, that sounds completely unexpected! A sudden turn of events can really catch us off balance.",
                "That must have caught you completely by surprise! Life certainly throws curveballs when we least expect them.",
                "I can hear the shock in your words. When the unexpected happens, our minds need a moment to catch up.",
            ],
            "reflection": [
                "Did this surprise feel like a pleasant excitement or more of an unsettling shock?",
                "How are you processing this sudden change of plans?",
            ],
            "affirmation": [
                "Take a moment to let the dust settle before deciding what to do next.",
                "Give your mind a few quiet minutes to integrate this new information.",
            ]
        },
        "Disgust": {
            "validation": [
                "I can understand why that left such a bad taste. Feeling repulsed or deeply uncomfortable is a strong, valid reaction.",
                "That sounds really unpleasant and off-putting. It is completely natural to want to distance yourself from that.",
                "Your instinct to reject that situation is totally understandable. Your comfort zone deserves to be respected.",
            ],
            "reflection": [
                "What was the most unsettling aspect of that experience?",
                "What would help you cleanse your space or regain a feeling of peace right now?",
            ],
            "affirmation": [
                "Stepping away and protecting your peace is always a wise choice.",
                "Focus on surroundings and activities that restore your sense of calm and cleanliness.",
            ]
        },
        "Neutral": {
            "validation": [
                "I am here and listening attentively. It sounds like you are reflecting on things evenly.",
                "Thank you for sharing your thoughts with me. I appreciate you taking the time to check in.",
                "I am glad we are connected today. Tell me more about what is on your mind.",
            ],
            "reflection": [
                "How has the rest of your day been treating you overall?",
                "Is there any specific topic or thought you would like to explore together?",
            ],
            "affirmation": [
                "Sometimes a steady, neutral day is the most restorative kind of day to have.",
                "Take your time — whenever you want to chat about anything big or small, I am here.",
            ]
        }
    }

    # Coping grounding exercises for emotional distress
    EXERCISES = {
        "box_breathing": (
            "🧘 **Quick Grounding Exercise (Box Breathing):**\n"
            "1. Breathe in slowly through your nose for **4 seconds**.\n"
            "2. Hold your breath gently for **4 seconds**.\n"
            "3. Exhale completely through your mouth for **4 seconds**.\n"
            "4. Rest for **4 seconds** before repeating twice."
        ),
        "grounding_54321": (
            "🌿 **The 5-4-3-2-1 Sensory Grounding Technique:**\n"
            "- Name **5** things you can see around your room.\n"
            "- Touch **4** different textures near your hands.\n"
            "- Listen for **3** distinct background sounds.\n"
            "- Identify **2** things you can smell or enjoy smelling.\n"
            "- Notice **1** good thing about your resilience right now."
        )
    }

    def generate(
        self,
        emotion: str,
        confidence: float,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generates contextual empathetic reply.

        Returns:
            Dict containing:
                - 'response_text': empathetic chatbot reply
                - 'coping_suggestion': optional exercise tip
                - 'transition_note': insight on emotional change if any
        """
        # Validate emotion key
        safe_emotion = emotion if emotion in self.EMOTION_RESPONSES else "Neutral"
        pool = self.EMOTION_RESPONSES[safe_emotion]

        # Select responses
        val = random.choice(pool["validation"])
        ref = random.choice(pool["reflection"])
        aff = random.choice(pool["affirmation"])

        # Detect transition from previous turns
        transition_note = None
        if conversation_history and len(conversation_history) >= 2:
            prev_turns = [t for t in conversation_history if t.get("role") == "user"]
            if len(prev_turns) >= 2:
                prev_emotion = prev_turns[-2].get("detected_emotion")
                if prev_emotion and prev_emotion != safe_emotion:
                    if prev_emotion in ["Sadness", "Fear", "Anger"] and safe_emotion == "Joy":
                        transition_note = "It is really uplifting to see your spirits lift compared to earlier in our conversation."
                    elif prev_emotion == "Joy" and safe_emotion in ["Sadness", "Fear"]:
                        transition_note = "I notice things took a heavier turn since earlier. I am right here with you."

        # Build response paragraphs
        parts = []
        if transition_note:
            parts.append(transition_note)
        parts.append(val)
        parts.append(ref)
        parts.append(aff)

        # Append coping exercise for high fear/anger/sadness
        coping_exercise = None
        if safe_emotion in ["Fear", "Anger"] and confidence > 0.60:
            coping_exercise = self.EXERCISES["box_breathing"]
        elif safe_emotion == "Sadness" and confidence > 0.70:
            coping_exercise = self.EXERCISES["grounding_54321"]

        if coping_exercise:
            parts.append("\n" + coping_exercise)

        full_text = "\n\n".join(parts)

        return {
            "response_text": full_text,
            "coping_suggestion": coping_exercise,
            "transition_note": transition_note,
        }


# Global response generator instance
response_generator = ResponseGenerator()
