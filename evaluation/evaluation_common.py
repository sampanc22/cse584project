import json
import random
import re
import time
from pathlib import Path
from math import ceil
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from core.cache import SimpleSemanticCache
from core.document_registry import DocumentRegistry
from core.adk_runtime import generate_fresh_with_agent

# ----------------------------
# Shared config
# ----------------------------

DATASET_PATH = "evaluation/squad.json"
SAMPLED_EXAMPLES_PATH = "evaluation/sampled_examples.json"
RANDOM_SEED = 42
TOP_K = 5
SIMILARITY_THRESHOLD = 0.92
REQUEST_SLEEP_SECONDS = 10.0
MAX_EXAMPLES = 100
BATCH_SIZE = 20
NUM_BATCHES = ceil(MAX_EXAMPLES / BATCH_SIZE)
NUM_ANSWER_CHANGING_VARIANTS = 1
INCLUDE_IRRELEVANT_EDIT = True

# ----------------------------
# Utilities
# ----------------------------

def sleep_between_requests() -> None:
    if REQUEST_SLEEP_SECONDS > 0:
        time.sleep(REQUEST_SLEEP_SECONDS)


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def choose_edit_answer_span(
    context: str,
    answers: List[Dict[str, Any]],
) -> Tuple[str, int]:
    candidates = []

    for ans in answers:
        text = ans.get("text", "").strip()
        start = ans.get("answer_start", -1)

        if not text:
            continue

        exact = False
        if start >= 0 and start + len(text) <= len(context):
            exact = context[start:start + len(text)] == text

        token_count = len(text.split())

        candidates.append({
            "text": text,
            "start": start,
            "exact": exact,
            "length": len(text),
            "token_count": token_count,
        })

    if not candidates:
        return "", -1

    exact_candidates = [c for c in candidates if c["exact"]]
    pool = exact_candidates if exact_candidates else candidates

    # Prefer "answer-like" phrase spans:
    # 1) 2-6 tokens (e.g. "the north", "Levi's Stadium", "San Jose")
    phrase_like = [c for c in pool if 2 <= c["token_count"] <= 6]
    if phrase_like:
        phrase_like.sort(key=lambda c: (c["length"], c["start"]))
        best = phrase_like[0]
        return best["text"], best["start"]

    # 2) Single-token answers if no good phrase-like span exists
    single_token = [c for c in pool if c["token_count"] == 1]
    if single_token:
        single_token.sort(key=lambda c: (c["length"], c["start"]))
        best = single_token[0]
        return best["text"], best["start"]

    # 3) Fallback: shortest exact candidate
    pool.sort(key=lambda c: (c["length"], c["start"]))
    best = pool[0]
    return best["text"], best["start"]


def replace_at_answer_start(
    context: str,
    old_answer: str,
    new_answer: str,
    answer_start: int,
) -> Tuple[str, bool]:
    if answer_start < 0:
        return context, False

    end = answer_start + len(old_answer)

    if end > len(context):
        return context, False

    if context[answer_start:end] != old_answer:
        return context, False

    updated = context[:answer_start] + new_answer + context[end:]
    return updated, True


def answer_matches(pred: str, gold_answers: List[str]) -> bool:
    pred_n = normalize_text(pred)

    for gold in gold_answers:
        gold_n = normalize_text(gold)

        if pred_n == gold_n:
            return True

        if gold_n and gold_n in pred_n:
            return True

        if pred_n and pred_n in gold_n:
            return True

    return False


def safe_load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError(f"Could not parse JSON from {path}")


def truncate_text(text: str, max_len: int = 240) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def is_model_failure(response: str) -> bool:
    return response.startswith("GEMINI_CALL_FAILED:")


# ----------------------------
# Dataset structures
# ----------------------------

@dataclass
class QAExample:
    doc_title: str
    qa_id: str
    question: str
    context: str
    gold_answers: List[str]

    # dedicated span for synthetic edits
    edit_answer: str
    edit_answer_start: int


def get_batch(
    examples: List[QAExample],
    batch_idx: int,
    batch_size: int,
) -> List[QAExample]:
    if batch_idx < 0:
        raise ValueError("batch_idx must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    start = batch_idx * batch_size
    end = start + batch_size

    if start >= len(examples):
        raise ValueError(
            f"batch_idx={batch_idx} out of range for {len(examples)} examples "
            f"with batch_size={batch_size}"
        )

    return examples[start:end]


def load_squad_examples(path: str, max_examples: Optional[int] = None) -> List[QAExample]:
    data = safe_load_json(path)

    if "data" not in data:
        raise ValueError("Expected top-level 'data' key in SQuAD-style JSON")

    examples: List[QAExample] = []

    for article in data["data"]:
        doc_title = article.get("title", "untitled_doc")

        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")

            for qa in paragraph.get("qas", []):
                question = qa.get("question", "").strip()
                qa_id = qa.get("id", "")
                answers = qa.get("answers", [])

                gold_answers = []
                for ans in answers:
                    text = ans.get("text", "").strip()
                    if text:
                        gold_answers.append(text)

                if not question or not context or not gold_answers:
                    continue

                seen = set()
                unique_gold = []
                for g in gold_answers:
                    if g not in seen:
                        unique_gold.append(g)
                        seen.add(g)

                edit_answer, edit_answer_start = choose_edit_answer_span(
                    context=context,
                    answers=answers,
                )

                if not edit_answer:
                    continue

                examples.append(
                    QAExample(
                        doc_title=doc_title,
                        qa_id=qa_id,
                        question=question,
                        context=context,
                        gold_answers=unique_gold,
                        edit_answer=edit_answer,
                        edit_answer_start=edit_answer_start,
                    )
                )

    random.shuffle(examples)

    if max_examples is not None:
        return examples[:max_examples]
    return examples


# ----------------------------
# Persistent sampled examples
# ----------------------------

def save_sampled_examples(
    examples: List[QAExample],
    path: str = SAMPLED_EXAMPLES_PATH,
) -> None:
    payload = [asdict(ex) for ex in examples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_sampled_examples(
    path: str = SAMPLED_EXAMPLES_PATH,
) -> List[QAExample]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [QAExample(**item) for item in payload]


def get_or_create_sampled_examples(
    dataset_path: str,
    max_examples: int,
    sampled_path: str = SAMPLED_EXAMPLES_PATH,
    force_resample: bool = False,
) -> List[QAExample]:
    sampled_file = Path(sampled_path)

    if sampled_file.exists() and not force_resample:
        examples = load_sampled_examples(sampled_path)
        if len(examples) != max_examples:
            raise ValueError(
                f"{sampled_path} contains {len(examples)} examples, "
                f"but this run expects {max_examples}. "
                f"Delete the file or call with force_resample=True."
            )
        return examples

    random.seed(RANDOM_SEED)
    examples = load_squad_examples(dataset_path, max_examples=max_examples)
    save_sampled_examples(examples, sampled_path)
    return examples


# ----------------------------
# Simulated document updates
# ----------------------------

def update_gold_answers_after_edit(
    gold_answers: List[str],
    original_answer: str,
    new_answer: str,
) -> List[str]:
    """
    Update only the gold answers that are affected by the synthetic edit.

    Rules:
    - if a gold answer exactly equals the edited answer, replace it
    - if the edited answer appears as a substring inside a longer gold answer,
      replace that substring
    - otherwise keep the gold answer unchanged
    - deduplicate results while preserving order
    """
    updated = []

    for ans in gold_answers:
        new_gold = ans

        if ans == original_answer:
            new_gold = new_answer
        elif original_answer in ans:
            new_gold = ans.replace(original_answer, new_answer)
        else:
            new_gold = ans

        updated.append(new_gold)

    # dedupe while preserving order
    deduped = []
    seen = set()
    for ans in updated:
        if ans not in seen:
            deduped.append(ans)
            seen.add(ans)

    return deduped


def replace_first_case_sensitive(text: str, old: str, new: str) -> Tuple[str, bool]:
    idx = text.find(old)
    if idx == -1:
        return text, False
    return text[:idx] + new + text[idx + len(old):], True


def mutate_answer_text(answer: str, variant_id: int = 0) -> str:
    answer = answer.strip()

    city_replacements = [
        "New York City",
        "Los Angeles",
        "Chicago",
        "London",
        "Paris",
        "Berlin",
        "Tokyo",
        "Toronto",
        "Houston",
        "Seattle",
        "Madrid",
        "Rome",
        "Beijing",
        "Sydney",
    ]

    org_replacements = [
        "ABC",
        "CBS",
        "FOX",
        "NBC",
        "ESPN",
        "NFL",
        "NFC",
        "AFC",
        "BBC",
        "FCC",
    ]

    single_word_entity_replacements = [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Omega",
        "Sigma",
        "Atlas",
        "Orion",
    ]

    location_phrase_replacements = [
    "the city centre",
    "the riverfront",
    "the northern edge",
    "the downtown area",
    "the waterfront",
    "the south",
    "the north",
    ]

    number_words = [
        "zero", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten", "eleven", "twelve"
    ]
    ordinals = [
        "first", "second", "third", "fourth", "fifth",
        "sixth", "seventh", "eighth", "ninth", "tenth"
    ]

    roman_numerals = [
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
        "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
        "XL", "XLI", "XLII", "XLIII", "XLIV", "XLV", "XLVI", "XLVII", "XLVIII",
        "XLIX", "L", "LI", "LII", "LIII", "LIV", "LV"
    ]

    score_deltas = [1, 2, 3, 4]
    number_deltas = [-3, -2, -1, 1, 2, 3]
    percent_deltas = [-10, -5, 5, 10]
    year_deltas = [-2, -1, 1, 2]

    # 1) Currency phrases like £76 million, $5 million, €1.2 billion
    money_phrase = re.fullmatch(
        r"([£$€])\s?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|thousand)?",
        answer,
        flags=re.IGNORECASE,
    )
    if money_phrase:
        currency = money_phrase.group(1)
        value = float(money_phrase.group(2).replace(",", ""))
        scale = money_phrase.group(3) or ""
        delta = [5, 10, 15, 20][variant_id % 4]

        if value >= 100:
            new_value = value + delta
        else:
            new_value = value + (delta / 10)

        if new_value.is_integer():
            new_num = str(int(new_value))
        else:
            new_num = f"{new_value:.1f}".rstrip("0").rstrip(".")

        return f"{currency}{new_num}" + (f" {scale}" if scale else "")

    # 2) Score / record patterns like 24–10, 15–1, 12-4
    dash_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", answer)
    if dash_match:
        a = int(dash_match.group(1))
        b = int(dash_match.group(2))
        delta = score_deltas[variant_id % len(score_deltas)]
        return f"{a + delta}–{max(0, b + (delta - 1))}"

    # 3) Mixed fraction style like 2½
    mixed_frac_match = re.fullmatch(r"(\d+)\s*½", answer)
    if mixed_frac_match:
        base = int(mixed_frac_match.group(1))
        delta = [1, 2][variant_id % 2]
        return f"{base + delta}½"

    # 4) Plain integer
    if re.fullmatch(r"\d+", answer):
        val = int(answer)
        delta = number_deltas[variant_id % len(number_deltas)]
        return str(max(1, val + delta))

    # 5) 4-digit year
    if re.fullmatch(r"\d{4}", answer):
        val = int(answer)
        delta = year_deltas[variant_id % len(year_deltas)]
        return str(val + delta)

    # 6) Decade-like years: 1950s
    decade_match = re.fullmatch(r"(\d{4})s", answer)
    if decade_match:
        val = int(decade_match.group(1))
        delta = year_deltas[variant_id % len(year_deltas)]
        return f"{val + delta}s"

    # 7) Percentages
    if re.fullmatch(r"\d+%", answer):
        val = int(answer[:-1])
        delta = percent_deltas[variant_id % len(percent_deltas)]
        val = max(0, min(100, val + delta))
        return f"{val}%"

    # 8) Year embedded inside phrase
    if re.search(r"\d{4}", answer):
        delta = 1 if variant_id % 2 == 0 else -1
        mutated = re.sub(
            r"\d{4}",
            lambda m: str(int(m.group()) + delta),
            answer,
            count=1,
        )
        if mutated != answer:
            return mutated

    # 9) Month/date phrases
    months = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    )
    present_months = [m for m in months if m in answer]
    if present_months:
        current_month = present_months[0]
        other_months = [m for m in months if m != current_month]
        replacement = other_months[variant_id % len(other_months)]
        mutated = answer.replace(current_month, replacement, 1)
        if mutated != answer:
            return mutated

    # 10) Number words
    lower_answer = answer.lower()
    if lower_answer in number_words:
        candidates = [w for w in number_words if w != lower_answer]
        return candidates[variant_id % len(candidates)]

    # 11) Ordinals
    if lower_answer in ordinals:
        candidates = [w for w in ordinals if w != lower_answer]
        return candidates[variant_id % len(candidates)]

    # 12) Roman numerals / Super Bowl numerals
    roman_core = answer.strip(".")
    if roman_core in roman_numerals:
        candidates = [r for r in roman_numerals if r != roman_core]
        replacement = candidates[variant_id % len(candidates)]
        return replacement + ("." if answer.endswith(".") else "")

    # 13) Acronyms / all-caps abbreviations like CBS, AFC, NFL
    if re.fullmatch(r"[A-Z]{2,6}", answer):
        candidates = [x for x in org_replacements if x != answer]
        if candidates:
            return candidates[variant_id % len(candidates)]

    # 14) Location-style phrases
    if (
        lower_answer.startswith("in ")
        or lower_answer.startswith("on ")
        or lower_answer.startswith("at ")
        or "north" in lower_answer
        or "south" in lower_answer
        or "east" in lower_answer
        or "west" in lower_answer
        or "waterfront" in lower_answer
    ):
        candidates = [x for x in location_phrase_replacements if x.lower() != lower_answer]
        if candidates:
            return candidates[variant_id % len(candidates)]

    # 15) Multi-word named entities
    if len(answer.split()) >= 2:
        candidates = [x for x in city_replacements if x.lower() != answer.lower()]
        if candidates:
            return candidates[variant_id % len(candidates)]

    # 16) Single-word entity fallback
    candidates = [x for x in single_word_entity_replacements if x.lower() != answer.lower()]
    if candidates:
        return candidates[variant_id % len(candidates)]

    return answer + f"_ALT_{variant_id + 1}"

def same_answer_type(original: str, new: str) -> bool:
    patterns = {
        "money": r"[£$€]\s?\d",
        "dash_number": r"\d+\s*[-–]\s*\d+",
        "fraction_half": r"\d+\s*½",
        "year": r"\b\d{4}\b",
        "decade": r"\b\d{4}s\b",
        "percent": r"\b\d+%\b",
        "roman": r"^(?:[IVXLCDM]+)\.?$",
        "acronym": r"^[A-Z]{2,6}$",
    }

    for _, pattern in patterns.items():
        if re.search(pattern, original):
            return bool(re.search(pattern, new))

    if len(original.split()) >= 2:
        return len(new.split()) >= 2

    return True

def replace_all_exact_occurrences(
    context: str,
    old_answer: str,
    new_answer: str,
) -> Tuple[str, bool]:
    count = context.count(old_answer)
    if count == 0:
        return context, False
    return context.replace(old_answer, new_answer), True

def simulate_document_update(
    context: str,
    gold_answers: List[str],
    edit_answer: str,
    edit_answer_start: int,
    update_type: str,
    variant_id: int = 0,
) -> Tuple[str, List[str], bool, Dict[str, Any]]:

    canonical = edit_answer

    new_answer = mutate_answer_text(canonical, variant_id=variant_id)

    if not same_answer_type(canonical, new_answer):
        new_answer = canonical + f"_ALT_{variant_id + 1}"

    metadata = {
        "update_type": update_type,
        "original_answer": canonical,
        "new_answer": new_answer,
        "edit_description": "no change",
        "applied": False,
    }

    if update_type == "none":
        metadata["edit_description"] = "original document"
        return context, gold_answers, False, metadata

    if update_type == "irrelevant_edit":
        updated_context = context + " NOTE: This document was reviewed for clarity."
        metadata["applied"] = True
        metadata["edit_description"] = "irrelevant sentence appended"
        return updated_context, gold_answers, False, metadata

    if update_type == "answer_changing_edit":
        occurrence_count = context.count(canonical)

        # If the answer appears multiple times, replace all exact occurrences.
        # This is especially helpful for short repeated answers like "San Jose".
        if occurrence_count > 1:
            updated_context, did_replace = replace_all_exact_occurrences(
                context=context,
                old_answer=canonical,
                new_answer=new_answer,
            )

            if did_replace:
                metadata["applied"] = True
                metadata["edit_description"] = (
                    f"replaced all {occurrence_count} occurrences of '{canonical}' -> '{new_answer}'"
                )

                updated_gold_answers = update_gold_answers_after_edit(
                    gold_answers=gold_answers,
                    original_answer=canonical,
                    new_answer=new_answer,
                )

                return updated_context, updated_gold_answers, True, metadata

        # Otherwise use the single labeled span
        updated_context, did_replace = replace_at_answer_start(
            context=context,
            old_answer=canonical,
            new_answer=new_answer,
            answer_start=edit_answer_start,
        )

        if did_replace:
            metadata["applied"] = True
            metadata["edit_description"] = f"replaced answer span: '{canonical}' -> '{new_answer}'"

            updated_gold_answers = update_gold_answers_after_edit(
                gold_answers=gold_answers,
                original_answer=canonical,
                new_answer=new_answer,
            )

            return updated_context, updated_gold_answers, True, metadata

        metadata["applied"] = False
        metadata["edit_description"] = (
            f"answer span not found at edit_answer_start={edit_answer_start}; "
            f"no replacement applied for '{canonical}'"
        )
        return context, gold_answers, False, metadata

# ----------------------------
# Stream construction
# ----------------------------
def build_cache_stream(
    examples: List[QAExample],
    num_answer_changing_variants: int = 2,
    include_irrelevant_edit: bool = True,
) -> List[Dict[str, Any]]:
    stream: List[Dict[str, Any]] = []

    for ex in examples:
        canonical = ex.edit_answer
        canonical_start = ex.edit_answer_start

        stream.append({
            "doc_title": ex.doc_title,
            "qa_id": ex.qa_id,
            "question": ex.question,
            "context": ex.context,
            "gold_answers": ex.gold_answers,
            "update_type": "none",
            "changed_answer": False,
            "tag": "initial",
            "original_answer": canonical,
            "new_answer": canonical,
            "edit_description": "original document",
        })

        stream.append({
            "doc_title": ex.doc_title,
            "qa_id": ex.qa_id,
            "question": ex.question,
            "context": ex.context,
            "gold_answers": ex.gold_answers,
            "update_type": "none",
            "changed_answer": False,
            "tag": "repeat_same_doc",
            "original_answer": canonical,
            "new_answer": canonical,
            "edit_description": "original document",
        })

        if include_irrelevant_edit:
            context_irrelevant, gold_irrelevant, changed_irrelevant, irrelevant_meta = simulate_document_update(
                ex.context,
                ex.gold_answers,
                canonical,
                canonical_start,
                update_type="irrelevant_edit",
            )

            irrelevant_request = {
                "doc_title": ex.doc_title,
                "qa_id": ex.qa_id,
                "question": ex.question,
                "context": context_irrelevant,
                "gold_answers": gold_irrelevant,
                "update_type": "irrelevant_edit",
                "changed_answer": changed_irrelevant,
                "original_answer": irrelevant_meta["original_answer"],
                "new_answer": irrelevant_meta["new_answer"],
                "edit_description": irrelevant_meta["edit_description"],
            }

            stream.append({
                **irrelevant_request,
                "tag": "updated_doc_irrelevant",
            })

            stream.append({
                **irrelevant_request,
                "tag": "repeat_irrelevant",
            })

        for variant_id in range(num_answer_changing_variants):
            context_changed, gold_changed, changed_answer, change_meta = simulate_document_update(
                ex.context,
                ex.gold_answers,
                canonical,
                canonical_start,
                update_type="answer_changing_edit",
                variant_id=variant_id,
            )

            changed_request = {
                "doc_title": ex.doc_title,
                "qa_id": ex.qa_id,
                "question": ex.question,
                "context": context_changed,
                "gold_answers": gold_changed,
                "update_type": "answer_changing_edit",
                "changed_answer": changed_answer,
                "original_answer": change_meta["original_answer"],
                "new_answer": change_meta["new_answer"],
                "edit_description": change_meta["edit_description"],
            }

            stream.append({
                **changed_request,
                "tag": f"updated_doc_changing_{variant_id + 1}",
            })

            stream.append({
                **changed_request,
                "tag": f"repeat_changed_{variant_id + 1}",
            })

    return stream


# ----------------------------
# Eval runner
# ----------------------------

class EvalRunner:
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
        self.doc_registry = DocumentRegistry()
        self.top_k = top_k
        Path("tmp").mkdir(parents=True, exist_ok=True)

    def answer_no_cache(self, question: str, context: str) -> str:
        return generate_fresh_with_agent(
            prompt=question,
            long_context=context,
            history=[],
            mode="document",
        )

    def answer_semantic_only(
        self,
        question: str,
        context: str,
        doc_title: str,
    ) -> Tuple[str, bool]:
        query_embedding = self.cache.embed_prompt(question)
        candidates = self.cache.lookup_top_k_with_embedding(query_embedding, k=self.top_k)

        if candidates:
            entry, _sim = candidates[0]
            self.cache.record_hit(entry)
            return entry.response, True

        fresh = generate_fresh_with_agent(
            prompt=question,
            long_context=context,
            history=[],
            mode="document",
        )

        if not is_model_failure(fresh):
            current_versions = self.doc_registry.get_document_versions(doc_title, context)
            self.cache.insert_with_embedding(
                prompt=question,
                response=fresh,
                embedding=query_embedding,
                metadata={
                    "document_versions": current_versions,
                    "mode": "document",
                },
            )

        return fresh, False

    def answer_semantic_plus_doc_validity(
        self,
        question: str,
        context: str,
        doc_title: str,
    ) -> Tuple[str, bool]:
        query_embedding = self.cache.embed_prompt(question)
        candidates = self.cache.lookup_top_k_with_embedding(query_embedding, k=self.top_k)
        current_versions = self.doc_registry.get_document_versions(doc_title, context)

        for entry, _sim in candidates:
            cached_versions = entry.metadata.get("document_versions", {})
            is_valid = cached_versions == current_versions

            if is_valid:
                self.cache.record_hit(entry)
                return entry.response, True

        fresh = generate_fresh_with_agent(
            prompt=question,
            long_context=context,
            history=[],
            mode="document",
        )

        if not is_model_failure(fresh):
            self.cache.insert_with_embedding(
                prompt=question,
                response=fresh,
                embedding=query_embedding,
                metadata={
                    "document_versions": current_versions,
                    "mode": "document",
                },
            )

        return fresh, False


# ----------------------------
# Metrics
# ----------------------------

def get_condition_bucket(update_type: str) -> str:
    if update_type == "answer_changing_edit":
        return "answer_changing_edit"
    return "non_answer_changing_edit"

def _fresh_metric_bucket() -> Dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "hits": 0,
        "misses": 0,
        "true_hits": 0,
        "false_hits": 0,
        "model_failures": 0,
        "latency_ms_sum": 0.0,
        "update_counts": {
            "none": 0,
            "irrelevant_edit": 0,
            "answer_changing_edit": 0,
        },
    }


def fresh_metrics() -> Dict[str, Any]:
    return {
        "overall": _fresh_metric_bucket(),
        "answer_changing_edit": _fresh_metric_bucket(),
        "non_answer_changing_edit": _fresh_metric_bucket(),
    }


def update_metrics(
    metrics: Dict[str, Any],
    *,
    correct: bool,
    cache_hit: bool,
    update_type: str,
    latency_ms: float,
    model_failed: bool = False,
) -> None:
    bucket_names = ["overall", get_condition_bucket(update_type)]

    for bucket_name in bucket_names:
        bucket = metrics[bucket_name]

        if model_failed:
            bucket["model_failures"] += 1
            continue

        bucket["total"] += 1
        bucket["latency_ms_sum"] += latency_ms

        if update_type not in bucket["update_counts"]:
            bucket["update_counts"][update_type] = 0
        bucket["update_counts"][update_type] += 1

        if correct:
            bucket["correct"] += 1

        if cache_hit:
            bucket["hits"] += 1
            if correct:
                bucket["true_hits"] += 1
            else:
                bucket["false_hits"] += 1
        else:
            bucket["misses"] += 1