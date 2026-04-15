from typing import Dict, List, Any, Tuple
import os
import json
import time
import httpx
from dotenv import load_dotenv
from google import genai

load_dotenv()

_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in .env")

client = genai.Client(api_key=_api_key)

_signature_cache: Dict[Tuple[str, ...], Dict[str, object]] = {}


def _normalize_signature(raw: Dict[str, Any]) -> Dict[str, object]:
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


def _empty_signature() -> Dict[str, object]:
    return {
        "domains": [],
        "intents": [],
        "constraints": {},
        "requested_info": [],
    }

def make_oracle_dialogue_signature(gold_state: Dict[str, Any]) -> Dict[str, object]:
    return _normalize_signature(gold_state)

def _extract_signature_with_gemini(
    latest_user_turn: str,
    history: List[str],
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Dict[str, object]:
    recent_history = history[-6:]
    conversation = "\n".join(recent_history + [f"USER: {latest_user_turn}"])

    prompt = f"""
You are extracting the CURRENT structured dialogue state for a task-oriented assistant.

Return ONLY valid JSON with exactly this schema:
{{
  "domains": ["..."],
  "intents": ["..."],
  "constraints": {{
    "slot_name": "slot_value"
  }},
  "requested_info": ["..."]
}}

Interpretation rules:
- Extract the user's CURRENT active task, not every historical topic mentioned earlier.
- Focus on the most recent user goal that is still active.
- "domains" should usually contain only the currently relevant domain, such as restaurant, hotel, train, taxi, attraction, flight, shopping, weather, or travel.
- "intents" should use a small normalized vocabulary when possible, such as search, book, request_info, compare, change_constraint, confirm, or cancel.
- "constraints" should include currently active user constraints that are explicitly stated or clearly carried over from recent turns.
- "requested_info" should include the specific pieces of information the user is currently asking for, such as address, postcode, phone, price, availability, or reference number.
- Prefer concise normalized values.
- Do not include stale constraints or stale requested_info from earlier completed tasks if the user has shifted to a new task.
- If the active task is partially clear, return the best grounded partial state instead of guessing.
- Return JSON only. No markdown. No explanation.

Recent conversation:
{conversation}
"""

    last_err = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            text = response.text.strip()

            try:
                parsed = json.loads(text)
                return _normalize_signature(parsed)
            except Exception:
                pass

            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end + 1])
                    return _normalize_signature(parsed)
                except Exception:
                    pass

            print("[SIGNATURE WARNING] Could not parse model output as JSON.")
            print("Raw output:", text[:300])
            return _empty_signature()

        except Exception as e:
            last_err = e
            msg = f"{type(e).__name__}: {e}"

            is_retryable = (
                isinstance(e, httpx.ConnectTimeout)
                or "ConnectTimeout" in msg
                or "timed out" in msg.lower()
                or "503" in msg
                or "UNAVAILABLE" in msg
            )

            if is_retryable and attempt < max_retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                print(
                    f"[SIGNATURE RETRY] attempt={attempt + 1}/{max_retries} "
                    f"sleeping {sleep_time:.1f}s after error: {msg}"
                )
                time.sleep(sleep_time)
                continue

            print(f"[SIGNATURE ERROR] {msg}")
            return _empty_signature()

    print(f"[SIGNATURE ERROR] {type(last_err).__name__}: {last_err}")
    return _empty_signature()


def make_gemini_dialogue_signature(
    latest_user_turn: str,
    history: List[str],
) -> Dict[str, object]:
    cache_key = tuple(history[-6:] + [latest_user_turn])

    if cache_key in _signature_cache:
        return _signature_cache[cache_key]

    signature = _extract_signature_with_gemini(latest_user_turn, history)
    _signature_cache[cache_key] = signature
    return signature

def make_dialogue_signature(
    latest_user_turn: str,
    history: List[str],
    gold_state: Dict[str, Any] | None = None,
    mode: str = "gemini",
) -> Dict[str, object]:
    if mode == "oracle":
        if gold_state is None:
            raise ValueError("gold_state is required for oracle dialogue signatures")
        return make_oracle_dialogue_signature(gold_state)

    if mode == "gemini":
        return make_gemini_dialogue_signature(latest_user_turn, history)

    raise ValueError(f"Unknown signature mode: {mode}")