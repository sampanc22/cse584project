from typing import Dict, List, Any, Tuple
import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in .env")

client = genai.Client(api_key=_api_key)

# Simple in-memory cache so repeated signature extraction is cheap
_signature_cache: Dict[Tuple[str, ...], Dict[str, object]] = {}


def _normalize_signature(raw: Dict[str, Any]) -> Dict[str, object]:
    """
    Normalize model output into a stable signature shape.
    """
    domains = raw.get("domains", [])
    if not isinstance(domains, list):
        domains = []

    intents = raw.get("intents", [])
    if not isinstance(intents, list):
        intents = []

    constraints = raw.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}

    requested_info = raw.get("requested_info", [])
    if not isinstance(requested_info, list):
        requested_info = []

    normalized_domains = sorted(
        {str(d).strip().lower() for d in domains if str(d).strip()}
    )
    normalized_intents = sorted(
        {str(i).strip().lower() for i in intents if str(i).strip()}
    )
    normalized_constraints = {
        str(k).strip().lower(): str(v).strip().lower()
        for k, v in constraints.items()
        if str(k).strip() and str(v).strip()
    }
    normalized_requested_info = sorted(
        {str(x).strip().lower() for x in requested_info if str(x).strip()}
    )

    return {
        "domains": normalized_domains,
        "intents": normalized_intents,
        "constraints": dict(sorted(normalized_constraints.items())),
        "requested_info": normalized_requested_info,
    }


def _extract_signature_with_gemini(user_utterance: str, history: List[str]) -> Dict[str, object]:
    recent_history = history[-5:]
    conversation = "\n".join([f"History: {turn}" for turn in recent_history] + [f"User: {user_utterance}"])

    prompt = f"""
You are extracting structured dialogue state for a task-oriented assistant.

Given the recent conversation, return ONLY valid JSON with exactly this schema:
{{
  "domains": ["..."],
  "intents": ["..."],
  "constraints": {{
    "slot_name": "slot_value"
  }},
  "requested_info": ["..."]
}}

Rules:
- "domains" should contain high-level task areas like hotel, restaurant, train, taxi, attraction, flight, shopping, weather, travel.
- "intents" should capture what the user is trying to do, like search, book, compare, ask_price, ask_location, ask_availability.
- "constraints" should include explicit user constraints only, such as city, area, price, date, time, destination, departure, stars, food, people, stay, etc.
- "requested_info" should include information the user is asking for, such as address, phone, wifi, parking, price, availability.
- Omit unknown fields rather than guessing.
- Return JSON only. No markdown. No explanation.

Conversation:
{conversation}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text = response.text.strip()

    # Try direct JSON parse first
    try:
        parsed = json.loads(text)
        return _normalize_signature(parsed)
    except Exception:
        pass

    # Fallback: extract the first JSON object from the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return _normalize_signature(parsed)
        except Exception:
            pass

    # Final fallback
    return {
        "domains": [],
        "intents": [],
        "constraints": {},
        "requested_info": [],
    }


def make_dialogue_signature(user_utterance: str, history: List[str]) -> Dict[str, object]:
    """
    Build a dialogue signature dynamically from recent dialogue.

    The signature is cached locally by the last few turns + current utterance
    so repeated calls do not repeatedly hit Gemini.
    """
    cache_key = tuple(history[-5:] + [user_utterance])

    if cache_key in _signature_cache:
        return _signature_cache[cache_key]

    signature = _extract_signature_with_gemini(user_utterance, history)
    _signature_cache[cache_key] = signature
    return signature