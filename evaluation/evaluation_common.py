import csv
import json
import random
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cache import SimpleSemanticCache
from document_registry import DocumentRegistry
from adk_runtime import generate_fresh_with_agent


# ----------------------------
# Shared config
# ----------------------------

DATASET_PATH = "squad.json"
RANDOM_SEED = 42

TOP_K = 5
SIMILARITY_THRESHOLD = 0.92
REQUEST_SLEEP_SECONDS = 3.0
MAX_EXAMPLES = 5

# New: persistent sampled subset so all runs use the same examples
SAMPLED_EXAMPLES_PATH = "sampled_examples.json"


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


def exact_match(pred: str, gold_answers: List[str]) -> bool:
    pred_n = normalize_text(pred)
    return any(pred_n == normalize_text(g) for g in gold_answers)


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

                examples.append(
                    QAExample(
                        doc_title=doc_title,
                        qa_id=qa_id,
                        question=question,
                        context=context,
                        gold_answers=unique_gold,
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

def choose_canonical_answer(gold_answers: List[str]) -> str:
    return sorted(gold_answers, key=lambda x: (len(x), x))[0]


def replace_first_case_sensitive(text: str, old: str, new: str) -> Tuple[str, bool]:
    idx = text.find(old)
    if idx == -1:
        return text, False
    return text[:idx] + new + text[idx + len(old):], True


def append_irrelevant_sentence(context: str) -> str:
    additions = [
        " This event received significant media attention across the country.",
        " Organizers also coordinated transportation and security for attendees.",
        " Analysts later discussed the long-term impact of the event.",
        " Additional background information was published in follow-up reports.",
    ]
    return context + random.choice(additions)


def mutate_answer_text(answer: str, variant_id: int = 0) -> str:
    answer = answer.strip()

    entity_replacements = [
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

    single_word_replacements = [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Omega",
        "Sigma",
        "Atlas",
        "Orion",
    ]

    number_deltas = [-3, -2, -1, 1, 2, 3]
    percent_deltas = [-10, -5, 5, 10]
    money_deltas = [-1000, -500, -100, 100, 500, 1000]
    year_deltas = [-2, -1, 1, 2]

    if re.fullmatch(r"\d+", answer):
        val = int(answer)
        delta = number_deltas[variant_id % len(number_deltas)]
        return str(max(1, val + delta))

    if re.fullmatch(r"\d{4}", answer):
        val = int(answer)
        delta = year_deltas[variant_id % len(year_deltas)]
        return str(val + delta)

    if re.fullmatch(r"\d+%", answer):
        val = int(answer[:-1])
        delta = percent_deltas[variant_id % len(percent_deltas)]
        val = max(0, min(100, val + delta))
        return f"{val}%"

    if re.fullmatch(r"\$?\d+(,\d{3})*(\.\d+)?", answer):
        digits = re.sub(r"[^\d]", "", answer)
        if digits:
            val = int(digits)
            delta = money_deltas[variant_id % len(money_deltas)]
            val = max(1, val + delta)
            return f"${val:,}"

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

    if len(answer.split()) >= 2:
        candidates = [x for x in entity_replacements if x.lower() != answer.lower()]
        if candidates:
            return candidates[variant_id % len(candidates)]

    candidates = [x for x in single_word_replacements if x.lower() != answer.lower()]
    if candidates:
        return candidates[variant_id % len(candidates)]

    return answer + f"_ALT_{variant_id + 1}"


def simulate_document_update(
    context: str,
    gold_answers: List[str],
    update_type: str,
    variant_id: int = 0,
) -> Tuple[str, List[str], bool]:
    canonical = choose_canonical_answer(gold_answers)

    if update_type == "none":
        return context, gold_answers, False

    if update_type == "irrelevant_edit":
        return append_irrelevant_sentence(context), gold_answers, False

    if update_type == "answer_changing_edit":
        new_answer = mutate_answer_text(canonical, variant_id=variant_id)

        updated_context, did_replace = replace_first_case_sensitive(
            context,
            canonical,
            new_answer,
        )

        if not did_replace:
            updated_context = (
                context
                + f" However, updated records clearly state that the correct answer is {new_answer}."
            )

        return updated_context, [new_answer], True

    raise ValueError(f"Unknown update_type: {update_type}")


# ----------------------------
# Stream construction
# ----------------------------

def build_no_cache_stream(examples: List[QAExample]) -> List[Dict[str, Any]]:
    stream: List[Dict[str, Any]] = []

    for ex in examples:
        stream.append({
            "doc_title": ex.doc_title,
            "qa_id": ex.qa_id,
            "question": ex.question,
            "context": ex.context,
            "gold_answers": ex.gold_answers,
            "update_type": "none",
            "changed_answer": False,
            "tag": "initial",
        })

    return stream


def build_cache_stream(
    examples: List[QAExample],
    num_answer_changing_variants: int = 2,
    include_irrelevant_edit: bool = True,
) -> List[Dict[str, Any]]:
    stream: List[Dict[str, Any]] = []

    for ex in examples:
        stream.append({
            "doc_title": ex.doc_title,
            "qa_id": ex.qa_id,
            "question": ex.question,
            "context": ex.context,
            "gold_answers": ex.gold_answers,
            "update_type": "none",
            "changed_answer": False,
            "tag": "initial",
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
        })

        if include_irrelevant_edit:
            context_irrelevant, gold_irrelevant, changed_irrelevant = simulate_document_update(
                ex.context,
                ex.gold_answers,
                update_type="irrelevant_edit",
            )
            stream.append({
                "doc_title": ex.doc_title,
                "qa_id": ex.qa_id,
                "question": ex.question,
                "context": context_irrelevant,
                "gold_answers": gold_irrelevant,
                "update_type": "irrelevant_edit",
                "changed_answer": changed_irrelevant,
                "tag": "updated_doc_irrelevant",
            })

        for variant_id in range(num_answer_changing_variants):
            context_changed, gold_changed, changed_answer = simulate_document_update(
                ex.context,
                ex.gold_answers,
                update_type="answer_changing_edit",
                variant_id=variant_id,
            )
            stream.append({
                "doc_title": ex.doc_title,
                "qa_id": ex.qa_id,
                "question": ex.question,
                "context": context_changed,
                "gold_answers": gold_changed,
                "update_type": "answer_changing_edit",
                "changed_answer": changed_answer,
                "tag": f"updated_doc_changing_{variant_id + 1}",
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
    ) -> Tuple[str, bool, bool]:
        query_embedding = self.cache.embed_prompt(question)
        candidates = self.cache.lookup_top_k_with_embedding(query_embedding, k=self.top_k)
        current_versions = self.doc_registry.get_document_versions(doc_title, context)

        stale_candidate_rejected = False

        for entry, _sim in candidates:
            cached_versions = entry.metadata.get("document_versions", {})
            is_valid = cached_versions == current_versions

            if is_valid:
                self.cache.record_hit(entry)
                return entry.response, True, stale_candidate_rejected

            stale_candidate_rejected = True

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

        return fresh, False, stale_candidate_rejected


# ----------------------------
# Metrics
# ----------------------------

def fresh_metrics() -> Dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "hits": 0,
        "misses": 0,
        "true_hits": 0,
        "false_hits": 0,
        "stale_avoided": 0,
        "answer_changed_cases": 0,
        "model_failures": 0,
        "update_counts": {
            "none": 0,
            "irrelevant_edit": 0,
            "answer_changing_edit": 0,
        },
    }


def update_metrics(
    metrics: Dict[str, Any],
    *,
    correct: bool,
    cache_hit: bool,
    stale_avoided: bool = False,
    changed_answer: bool = False,
    update_type: str,
    model_failed: bool = False,
) -> None:
    if model_failed:
        metrics["model_failures"] += 1
        return

    metrics["total"] += 1

    if update_type not in metrics["update_counts"]:
        metrics["update_counts"][update_type] = 0
    metrics["update_counts"][update_type] += 1

    if changed_answer:
        metrics["answer_changed_cases"] += 1

    if correct:
        metrics["correct"] += 1

    if cache_hit:
        metrics["hits"] += 1
        if correct:
            metrics["true_hits"] += 1
        else:
            metrics["false_hits"] += 1
    else:
        metrics["misses"] += 1

    if stale_avoided:
        metrics["stale_avoided"] += 1


def compute_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    total = max(metrics["total"], 1)
    hits = metrics["hits"]

    return {
        "total": metrics["total"],
        "accuracy": metrics["correct"] / total,
        "hits": metrics["hits"],
        "misses": metrics["misses"],
        "hit_rate": metrics["hits"] / total,
        "true_hits": metrics["true_hits"],
        "false_hits": metrics["false_hits"],
        "false_hit_rate_overall": metrics["false_hits"] / total,
        "false_hit_rate_given_hit": (metrics["false_hits"] / hits) if hits > 0 else 0.0,
        "stale_avoided": metrics["stale_avoided"],
        "answer_changed_cases": metrics["answer_changed_cases"],
        "model_failures": metrics["model_failures"],
        "update_counts": metrics["update_counts"],
    }


def summarize(name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    summary = compute_summary(metrics)

    print(f"\n===== {name} =====")
    print(f"total: {summary['total']}")
    print(f"accuracy: {summary['accuracy']:.3f}")
    print(f"hits: {summary['hits']}")
    print(f"misses: {summary['misses']}")
    print(f"hit_rate: {summary['hit_rate']:.3f}")
    print(f"true_hits: {summary['true_hits']}")
    print(f"false_hits: {summary['false_hits']}")
    print(f"false_hit_rate_overall: {summary['false_hit_rate_overall']:.3f}")
    print(f"false_hit_rate_given_hit: {summary['false_hit_rate_given_hit']:.3f}")
    print(f"stale_avoided: {summary['stale_avoided']}")
    print(f"answer_changed_cases: {summary['answer_changed_cases']}")
    print(f"model_failures: {summary['model_failures']}")
    print(f"update_counts: {summary['update_counts']}")

    return summary


# ----------------------------
# Logging
# ----------------------------

CSV_FIELDNAMES = [
    "timestamp",
    "request_index",
    "tag",
    "baseline",
    "qa_id",
    "doc_title",
    "question",
    "gold_answers",
    "prediction",
    "correct",
    "cache_hit",
    "true_hit",
    "false_hit",
    "stale_avoided",
    "model_failed",
    "update_type",
    "changed_answer",
    "context_preview",
]


def init_csv_log(csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()


def append_csv_row(csv_path: str, row: Dict[str, Any]) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)


def log_result(
    csv_path: str,
    *,
    request_index: int,
    tag: str,
    baseline: str,
    qa_id: str,
    doc_title: str,
    question: str,
    gold_answers: List[str],
    prediction: str,
    correct: bool,
    cache_hit: bool,
    stale_avoided: bool,
    update_type: str,
    changed_answer: bool,
    context: str,
    model_failed: bool,
) -> None:
    true_hit = cache_hit and correct and (not model_failed)
    false_hit = cache_hit and (not correct) and (not model_failed)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_index": request_index,
        "tag": tag,
        "baseline": baseline,
        "qa_id": qa_id,
        "doc_title": doc_title,
        "question": question,
        "gold_answers": json.dumps(gold_answers, ensure_ascii=False),
        "prediction": prediction,
        "correct": correct,
        "cache_hit": cache_hit,
        "true_hit": true_hit,
        "false_hit": false_hit,
        "stale_avoided": stale_avoided,
        "model_failed": model_failed,
        "update_type": update_type,
        "changed_answer": changed_answer,
        "context_preview": truncate_text(context),
    }
    append_csv_row(csv_path, row)


def write_summary_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)