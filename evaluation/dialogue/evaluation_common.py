import json
import random
import re
import time
from pathlib import Path
from math import ceil
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from core.cache import SimpleSemanticCache
from core.adk_runtime import generate_fresh_with_agent
from core.signatures import make_dialogue_signature
from core.validity import dialogue_valid


# ----------------------------
# Shared config
# ----------------------------

DIALOGUE_DATASET_PATH = "evaluation/dialogue/data.json"
DIALOGUE_SAMPLED_DIALOGUES_PATH = "evaluation/dialogue/dialogue_sampled_dialogues.json"

RANDOM_SEED = 42
TOP_K = 5
SIMILARITY_THRESHOLD = 0.92
REQUEST_SLEEP_SECONDS = 8.0

MAX_DIALOGUES = 30
BATCH_SIZE = 10
NUM_BATCHES = ceil(MAX_DIALOGUES / BATCH_SIZE)

DIALOGUE_QUESTION = (
    'Return the user\'s current dialogue state as JSON with exactly this schema: '
    '{"domains": ["..."], "intents": ["..."], "constraints": {"slot_name": "slot_value"}, '
    '"requested_info": ["..."]}. '
    "Use only information explicitly supported by the dialogue. "
    "Return JSON only."
)

DIALOGUE_STRICTNESS = "strict"


# ----------------------------
# Utilities
# ----------------------------

def sleep_between_requests() -> None:
    if REQUEST_SLEEP_SECONDS > 0:
        time.sleep(REQUEST_SLEEP_SECONDS)


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"Could not parse JSON from {path}")


def truncate_text(text: str, max_len: int = 240) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def is_model_failure(response: str) -> bool:
    return response.startswith("GEMINI_CALL_FAILED:")


def get_batch(
    items: List[Any],
    batch_idx: int,
    batch_size: int,
) -> List[Any]:
    if batch_idx < 0:
        raise ValueError("batch_idx must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    start = batch_idx * batch_size
    end = start + batch_size

    if start >= len(items):
        raise ValueError(
            f"batch_idx={batch_idx} out of range for {len(items)} items "
            f"with batch_size={batch_size}"
        )

    return items[start:end]


# ----------------------------
# Dataset structures
# ----------------------------

@dataclass
class DialogueExample:
    dialogue_id: str
    turn_id: str
    question: str
    history: List[str]
    latest_user_turn: str
    gold_answers: List[str]
    gold_state: Dict[str, Any]
    domain: str
    intent: str
    constraints: Dict[str, str]
    requested_info: List[str]
    context: str = ""


@dataclass
class DialogueTrajectory:
    dialogue_id: str
    examples: List[DialogueExample]


# ----------------------------
# Gold state construction
# ----------------------------

def _strip_domain_prefix(slot_name: str) -> str:
    if "-" in slot_name:
        return slot_name.split("-", 1)[1].strip().lower()
    return slot_name.strip().lower()


def _extract_active_frame(turn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    frames = turn.get("frames", [])

    active_frames = []
    for frame in frames:
        state = frame.get("state", {})
        active_intent = state.get("active_intent", "NONE")
        if active_intent and active_intent != "NONE":
            active_frames.append(frame)

    if len(active_frames) != 1:
        return None

    return active_frames[0]


def _extract_constraints(frame: Dict[str, Any]) -> Dict[str, str]:
    state = frame.get("state", {})
    slot_values = state.get("slot_values", {})

    constraints: Dict[str, str] = {}
    for slot_name, values in slot_values.items():
        if not values:
            continue
        value = values[0]
        if value is None:
            continue
        constraints[_strip_domain_prefix(slot_name)] = str(value).strip().lower()

    return dict(sorted(constraints.items()))


def _extract_requested_info(frame: Dict[str, Any]) -> List[str]:
    state = frame.get("state", {})
    requested_slots = state.get("requested_slots", [])

    requested = []
    for slot_name in requested_slots:
        requested.append(_strip_domain_prefix(slot_name))

    return sorted(set(requested))


def build_gold_state_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    service = frame.get("service", "").strip().lower()
    state = frame.get("state", {})
    active_intent = state.get("active_intent", "").strip().lower()

    constraints = _extract_constraints(frame)
    requested_info = _extract_requested_info(frame)

    return {
        "domains": [service] if service else [],
        "intents": [active_intent] if active_intent else [],
        "constraints": dict(sorted(constraints.items())),
        "requested_info": sorted(requested_info),
    }


# ----------------------------
# Clean normalization for accuracy
# ----------------------------

def _normalize_slot_name(slot: str, domain_hint: Optional[str] = None) -> str:
    slot = normalize_text(slot).replace(" ", "").replace("_", "").replace("-", "")

    slot_aliases = {
        "leaveafter": "leaveat",
        "leaveby": "leaveat",
        "leavetime": "leaveat",
        "leaveat": "leaveat",
        "departtime": "leaveat",
        "departuretime": "leaveat",
        "departureafter": "leaveat",
        "leaveaftertime": "leaveat",
        "departafter": "leaveat",

        "arrivaltime": "arriveby",
        "arrivaltime": "arriveby",
        "arrivalby": "arriveby",
        "arrivebefore": "arriveby",
        "arriveby": "arriveby",

        "numberofpeople": "bookpeople",
        "numpeople": "bookpeople",
        "numppl": "bookpeople",
        "people": "bookpeople",
        "bookpeople": "bookpeople",

        "stay": "bookstay",
        "bookstay": "bookstay",

        "time": "booktime" if domain_hint in {"restaurant", "hotel"} else "time",
        "booktime": "booktime",
        "bookday": "bookday",
        "bookdate": "bookday",

        "wifi": "internet",
        "internet": "internet",

        "price": "pricerange",
        "pricerange": "pricerange",
        "pricerang": "pricerange",
        "pricerangee": "pricerange",
        "price range": "pricerange",

        "postcode": "postcode",
        "postalcode": "postcode",
        "post code": "postcode",

        "address": "address",
        "phone": "phone",
        "entrancefee": "entrancefee",
        "entrance fee": "entrancefee",
        "fee": "entrancefee",

        "type": "type",
        "area": "area",
        "stars": "stars",
        "parking": "parking",
        "destination": "destination",
        "departfrom": "departure",
        "departure": "departure",
        "name": "name",
        "food": "food",
    }

    return slot_aliases.get(slot, slot)


def _normalize_value(value: Any) -> str:
    value = normalize_text(str(value))

    value_aliases = {
        "free": "yes",
        "free parking": "yes",
        "free wifi": "yes",
        "wifi": "yes",
        "internet": "yes",
        "doesntcare": "dontcare",
        "doesn'tcare": "dontcare",
        "doesnt matter": "dontcare",
        "doesn't matter": "dontcare",
        "any": "dontcare",
        "center": "centre",
        "city center": "centre",
        "city centre": "centre",
        "guest house": "guesthouse",
    }

    return value_aliases.get(value, value)


def _normalize_domain(domain: str) -> str:
    domain = normalize_text(domain)

    domain_aliases = {
        "food": "restaurant",
        "hotel": "hotel",
        "train": "train",
        "restaurant": "restaurant",
        "attraction": "attraction",
        "taxi": "taxi",
    }

    return domain_aliases.get(domain, domain)


def _normalize_intents(intents: List[str], domains: List[str]) -> List[str]:
    normalized = []
    domain_hint = domains[0] if domains else None

    for intent in intents:
        intent = normalize_text(intent).replace(" ", "_")

        if intent == "find":
            if domain_hint:
                intent = f"find_{domain_hint}"
        elif intent == "book":
            if domain_hint:
                intent = f"book_{domain_hint}"
        elif intent in {"request", "request_info", "request_information"}:
            intent = "request_info"
        elif intent in {"inform", "confirm"}:
            intent = intent

        normalized.append(intent)

    return sorted(set(normalized))


def normalize_dialogue_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    domains = raw.get("domains", [])
    if not isinstance(domains, list):
        domains = []

    constraints = raw.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}

    requested_info = raw.get("requested_info", [])
    if not isinstance(requested_info, list):
        requested_info = []

    norm_domains = sorted(
        {_normalize_domain(str(x)) for x in domains if str(x).strip()}
    )

    intents = raw.get("intents", [])
    if not isinstance(intents, list):
        intents = []
    norm_intents = _normalize_intents(
        [str(x) for x in intents if str(x).strip()],
        norm_domains,
    )

    norm_constraints = {}
    domain_hint = norm_domains[0] if norm_domains else None
    for k, v in constraints.items():
        k_norm = _normalize_slot_name(str(k), domain_hint)
        v_norm = _normalize_value(v)
        if k_norm and v_norm:
            norm_constraints[k_norm] = v_norm

    norm_requested = sorted(
        {
            _normalize_slot_name(str(x), domain_hint)
            for x in requested_info
            if str(x).strip()
        }
    )

    return {
        "domains": norm_domains,
        "intents": norm_intents,
        "constraints": dict(sorted(norm_constraints.items())),
        "requested_info": norm_requested,
    }


def parse_dialogue_state_json(text: str) -> Dict[str, Any]:
    empty = {
        "domains": [],
        "intents": [],
        "constraints": {},
        "requested_info": [],
    }

    text = text.strip()
    if not text:
        return empty

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return normalize_dialogue_state(parsed)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                return normalize_dialogue_state(parsed)
        except Exception:
            pass

    return empty


def build_gold_answer_from_frame(frame: Dict[str, Any]) -> str:
    gold_state = build_gold_state_from_frame(frame)
    return json.dumps(gold_state, sort_keys=True, ensure_ascii=False)


def gold_states_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return normalize_dialogue_state(a) == normalize_dialogue_state(b)


def dialogue_answer_accuracy_match(pred: str, gold_answers: List[str]) -> bool:
    pred_state = parse_dialogue_state_json(pred)

    for gold in gold_answers:
        gold_state = parse_dialogue_state_json(gold)
        if pred_state == gold_state:
            return True

    return False


# ----------------------------
# MultiWOZ loading
# ----------------------------

def load_multiwoz_trajectories(
    path: str,
    max_dialogues: Optional[int] = None,
    question: str = DIALOGUE_QUESTION,
) -> List[DialogueTrajectory]:
    data = safe_load_json(path)

    if not isinstance(data, list):
        raise ValueError("Expected top-level list for MultiWOZ-style JSON")

    trajectories: List[DialogueTrajectory] = []

    for dialogue in data:
        dialogue_id = dialogue.get("dialogue_id", "unknown_dialogue")
        turns = dialogue.get("turns", [])

        history: List[str] = []
        examples: List[DialogueExample] = []

        for turn in turns:
            speaker = turn.get("speaker", "").strip().upper()
            utterance = turn.get("utterance", "").strip()
            turn_id = str(turn.get("turn_id", ""))

            if not utterance:
                continue

            if speaker == "USER":
                active_frame = _extract_active_frame(turn)

                if active_frame is not None:
                    domain = active_frame.get("service", "").strip().lower()
                    intent = (
                        active_frame.get("state", {})
                        .get("active_intent", "")
                        .strip()
                        .lower()
                    )
                    constraints = _extract_constraints(active_frame)
                    requested_info = _extract_requested_info(active_frame)

                    if domain and intent and (constraints or requested_info):
                        gold_state = build_gold_state_from_frame(active_frame)
                        gold_answer = json.dumps(gold_state, sort_keys=True, ensure_ascii=False)

                        examples.append(
                            DialogueExample(
                                dialogue_id=dialogue_id,
                                turn_id=turn_id,
                                question=question,
                                history=list(history),
                                latest_user_turn=utterance,
                                gold_answers=[gold_answer],
                                gold_state=gold_state,
                                domain=domain,
                                intent=intent,
                                constraints=constraints,
                                requested_info=requested_info,
                                context="",
                            )
                        )

                history.append(f"USER: {utterance}")

            elif speaker == "SYSTEM":
                history.append(f"SYSTEM: {utterance}")

        if len(examples) >= 2:
            trajectories.append(
                DialogueTrajectory(
                    dialogue_id=dialogue_id,
                    examples=examples,
                )
            )

    random.shuffle(trajectories)

    if max_dialogues is not None:
        return trajectories[:max_dialogues]
    return trajectories


# ----------------------------
# Persistent sampled trajectories
# ----------------------------

def save_sampled_dialogue_trajectories(
    trajectories: List[DialogueTrajectory],
    path: str = DIALOGUE_SAMPLED_DIALOGUES_PATH,
) -> None:
    payload = [asdict(traj) for traj in trajectories]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_sampled_dialogue_trajectories(
    path: str = DIALOGUE_SAMPLED_DIALOGUES_PATH,
) -> List[DialogueTrajectory]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    trajectories: List[DialogueTrajectory] = []
    for item in payload:
        examples = [DialogueExample(**ex) for ex in item["examples"]]
        trajectories.append(
            DialogueTrajectory(
                dialogue_id=item["dialogue_id"],
                examples=examples,
            )
        )
    return trajectories


def get_or_create_dialogue_sampled_trajectories(
    dataset_path: str,
    max_dialogues: int,
    sampled_path: str = DIALOGUE_SAMPLED_DIALOGUES_PATH,
    force_resample: bool = False,
) -> List[DialogueTrajectory]:
    sampled_file = Path(sampled_path)

    if sampled_file.exists() and not force_resample:
        trajectories = load_sampled_dialogue_trajectories(sampled_path)
        if len(trajectories) != max_dialogues:
            raise ValueError(
                f"{sampled_path} contains {len(trajectories)} trajectories, "
                f"but this run expects {max_dialogues}. "
                f"Delete the file or call with force_resample=True."
            )
        return trajectories

    random.seed(RANDOM_SEED)
    trajectories = load_multiwoz_trajectories(dataset_path, max_dialogues=max_dialogues)
    save_sampled_dialogue_trajectories(trajectories, sampled_path)
    return trajectories


# ----------------------------
# Stream construction
# ----------------------------

def _classify_transition(a: DialogueExample, b: DialogueExample) -> str:
    if a.domain != b.domain:
        return "domain_shift"
    if a.intent != b.intent:
        return "intent_shift"
    return "constraint_shift"


def build_dialogue_cache_stream(
    trajectories: List[DialogueTrajectory],
) -> List[Dict[str, Any]]:
    stream: List[Dict[str, Any]] = []

    for traj in trajectories:
        examples = sorted(
            traj.examples,
            key=lambda x: int(x.turn_id) if str(x.turn_id).isdigit() else str(x.turn_id)
        )

        if not examples:
            continue

        for i, ex in enumerate(examples):
            stream.append(
                {
                    "dialogue_id": ex.dialogue_id,
                    "turn_id": ex.turn_id,
                    "question": ex.question,
                    "history": ex.history,
                    "latest_user_turn": ex.latest_user_turn,
                    "context": ex.context,
                    "gold_answers": ex.gold_answers,
                    "gold_state": ex.gold_state,
                    "transition_type": "none",
                    "tag": "initial",
                    "domain": ex.domain,
                    "intent": ex.intent,
                }
            )

            stream.append(
                {
                    "dialogue_id": ex.dialogue_id,
                    "turn_id": ex.turn_id,
                    "question": ex.question,
                    "history": ex.history,
                    "latest_user_turn": ex.latest_user_turn,
                    "context": ex.context,
                    "gold_answers": ex.gold_answers,
                    "gold_state": ex.gold_state,
                    "transition_type": "none",
                    "tag": "repeat_same_state",
                    "domain": ex.domain,
                    "intent": ex.intent,
                }
            )

            if i + 1 < len(examples):
                nxt = examples[i + 1]
                if not gold_states_equal(ex.gold_state, nxt.gold_state):
                    transition_type = _classify_transition(ex, nxt)
                    stream.append(
                        {
                            "dialogue_id": nxt.dialogue_id,
                            "turn_id": nxt.turn_id,
                            "question": nxt.question,
                            "history": nxt.history,
                            "latest_user_turn": nxt.latest_user_turn,
                            "context": nxt.context,
                            "gold_answers": nxt.gold_answers,
                            "gold_state": nxt.gold_state,
                            "transition_type": transition_type,
                            "tag": transition_type,
                            "domain": nxt.domain,
                            "intent": nxt.intent,
                        }
                    )

    return stream


# ----------------------------
# Eval runner
# ----------------------------

class DialogueEvalRunner:
    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_entries: int = 1000,
        eviction_policy: str = "lru",
        top_k: int = TOP_K,
    ):
        self.cache = SimpleSemanticCache(
            similarity_threshold=similarity_threshold,
            max_entries=max_entries,
            eviction_policy=eviction_policy,
        )
        self.top_k = top_k
        Path("tmp").mkdir(parents=True, exist_ok=True)

    def answer_no_cache(
        self,
        question: str,
        history: List[str],
        latest_user_turn: str,
        context: str = "",
    ) -> str:
        full_history = history + [f"USER: {latest_user_turn}"]
        fresh = generate_fresh_with_agent(
            prompt=question,
            long_context=context,
            history=full_history,
            mode="dialogue",
        )
        return fresh

    def answer_semantic_only(
        self,
        question: str,
        history: List[str],
        latest_user_turn: str,
        gold_state: Dict[str, Any],
        context: str = "",
    ) -> Tuple[str, bool, Optional[bool]]:
        query_embedding = self.cache.embed_prompt(question)
        candidates = self.cache.lookup_top_k_with_embedding(query_embedding, k=self.top_k)

        for entry, _sim in candidates:
            self.cache.record_hit(entry)
            cached_gold_state = entry.metadata.get("gold_state", {})
            hit_gold_valid = gold_states_equal(cached_gold_state, gold_state)
            return entry.response, True, hit_gold_valid

        full_history = history + [f"USER: {latest_user_turn}"]
        fresh = generate_fresh_with_agent(
            prompt=question,
            long_context=context,
            history=full_history,
            mode="dialogue",
        )

        if not is_model_failure(fresh):
            self.cache.insert_with_embedding(
                prompt=question,
                response=fresh,
                embedding=query_embedding,
                metadata={
                    "gold_state": normalize_dialogue_state(gold_state),
                    "mode": "dialogue",
                },
            )

        return fresh, False, None

    def answer_semantic_plus_dialogue_validity(
        self,
        question: str,
        history: List[str],
        latest_user_turn: str,
        gold_state: Dict[str, Any],
        context: str = "",
        strictness: str = DIALOGUE_STRICTNESS,
    ) -> Tuple[str, bool, Optional[bool]]:
        query_embedding = self.cache.embed_prompt(question)
        candidates = self.cache.lookup_top_k_with_embedding(query_embedding, k=self.top_k)
        current_sig = make_dialogue_signature(
                latest_user_turn=latest_user_turn,
                history=history,
                gold_state=gold_state,
                mode="oracle",
            )

        for entry, _sim in candidates:
            cached_sig = entry.metadata.get("dialogue_signature", {})
            is_valid = dialogue_valid(current_sig, cached_sig, strictness=strictness)

            if is_valid:
                self.cache.record_hit(entry)
                cached_gold_state = entry.metadata.get("gold_state", {})
                hit_gold_valid = gold_states_equal(cached_gold_state, gold_state)
                return entry.response, True, hit_gold_valid

        full_history = history + [f"USER: {latest_user_turn}"]
        fresh = generate_fresh_with_agent(
            prompt=question,
            long_context=context,
            history=full_history,
            mode="dialogue",
        )

        if not is_model_failure(fresh):
            self.cache.insert_with_embedding(
                prompt=question,
                response=fresh,
                embedding=query_embedding,
                metadata={
                    "dialogue_signature": current_sig,
                    "gold_state": normalize_dialogue_state(gold_state),
                    "mode": "dialogue",
                },
            )

        return fresh, False, None


# ----------------------------
# Metrics
# ----------------------------

def get_dialogue_condition_bucket(transition_type: str) -> str:
    if transition_type in {"domain_shift", "intent_shift", "constraint_shift"}:
        return "state_changing"
    return "state_preserving"


def _fresh_dialogue_metric_bucket() -> Dict[str, Any]:
    return {
        "total": 0,
        "hits": 0,
        "misses": 0,
        "true_hits": 0,
        "false_hits": 0,
        "correct_answers": 0,
        "incorrect_answers": 0,
        "model_failures": 0,
        "latency_ms_sum": 0.0,
        "transition_counts": {
            "none": 0,
            "domain_shift": 0,
            "intent_shift": 0,
            "constraint_shift": 0,
        },
    }


def fresh_dialogue_metrics() -> Dict[str, Any]:
    return {
        "overall": _fresh_dialogue_metric_bucket(),
        "state_preserving": _fresh_dialogue_metric_bucket(),
        "state_changing": _fresh_dialogue_metric_bucket(),
    }


def update_dialogue_cache_metrics(
    metrics: Dict[str, Any],
    *,
    cache_hit: bool,
    hit_gold_valid: Optional[bool],
    answer_correct: bool,
    transition_type: str,
    latency_ms: float,
    model_failed: bool = False,
) -> None:
    bucket_names = ["overall", get_dialogue_condition_bucket(transition_type)]

    for bucket_name in bucket_names:
        bucket = metrics[bucket_name]

        if model_failed:
            bucket["model_failures"] += 1
            continue

        bucket["total"] += 1
        bucket["latency_ms_sum"] += latency_ms

        if transition_type not in bucket["transition_counts"]:
            bucket["transition_counts"][transition_type] = 0
        bucket["transition_counts"][transition_type] += 1

        if answer_correct:
            bucket["correct_answers"] += 1
        else:
            bucket["incorrect_answers"] += 1

        if cache_hit:
            bucket["hits"] += 1
            if hit_gold_valid is True:
                bucket["true_hits"] += 1
            elif hit_gold_valid is False:
                bucket["false_hits"] += 1
        else:
            bucket["misses"] += 1