import json
from pathlib import Path

from evaluation.document.evaluation_common import NUM_BATCHES as NUM_DOCUMENT_BATCHES
from evaluation.dialogue.evaluation_common import NUM_BATCHES as NUM_DIALOGUE_BATCHES


def make_document_batch_files(prefix: str) -> list[str]:
    return [f"tmp/document_{prefix}_batch_{i}.json" for i in range(NUM_DOCUMENT_BATCHES)]


def make_dialogue_batch_files(prefix: str) -> list[str]:
    return [f"tmp/dialogue_{prefix}_batch_{i}.json" for i in range(NUM_DIALOGUE_BATCHES)]


EXPERIMENTS = {
    "document_no_cache": {
        "kind": "document",
        "input_files": make_document_batch_files("no_cache"),
        "output_file": "results/document_no_cache_summary.json",
        "display_name": "NO CACHE (DOCUMENT)",
    },
    "document_semantic_only": {
        "kind": "document",
        "input_files": make_document_batch_files("semantic_only"),
        "output_file": "results/document_semantic_only_summary.json",
        "display_name": "SEMANTIC ONLY (DOCUMENT)",
    },
    "document_semantic_plus_doc_validity": {
        "kind": "document",
        "input_files": make_document_batch_files("semantic_plus_doc_validity"),
        "output_file": "results/document_semantic_plus_doc_validity_summary.json",
        "display_name": "SEMANTIC + DOC VALIDITY (DOCUMENT)",
    },
    "dialogue_no_cache": {
        "kind": "dialogue",
        "input_files": make_dialogue_batch_files("no_cache"),
        "output_file": "results/dialogue_no_cache_summary.json",
        "display_name": "NO CACHE (DIALOGUE)",
    },
    "dialogue_semantic_only": {
        "kind": "dialogue",
        "input_files": make_dialogue_batch_files("semantic_only"),
        "output_file": "results/dialogue_semantic_only_summary.json",
        "display_name": "SEMANTIC ONLY (DIALOGUE)",
    },
    "dialogue_semantic_plus_validity_strict": {
        "kind": "dialogue",
        "input_files": make_dialogue_batch_files("semantic_plus_validity_strict"),
        "output_file": "results/dialogue_semantic_plus_validity_strict_summary.json",
        "display_name": "SEMANTIC + STRICT VALIDITY (DIALOGUE)",
    },
    "dialogue_semantic_plus_validity_slot_relaxed": {
        "kind": "dialogue",
        "input_files": make_dialogue_batch_files("semantic_plus_validity_slot_relaxed"),
        "output_file": "results/dialogue_semantic_plus_validity_slot_relaxed_summary.json",
        "display_name": "SEMANTIC + SLOT-RELAXED VALIDITY (DIALOGUE)",
    },
    "dialogue_semantic_plus_validity_intent_domain": {
        "kind": "dialogue",
        "input_files": make_dialogue_batch_files("semantic_plus_validity_intent_domain"),
        "output_file": "results/dialogue_semantic_plus_validity_intent_domain_summary.json",
        "display_name": "SEMANTIC + INTENT-DOMAIN VALIDITY (DIALOGUE)",
    },
}

Path("results").mkdir(parents=True, exist_ok=True)


# ----------------------------
# Document metrics
# ----------------------------

def fresh_document_bucket():
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


def fresh_document_metrics():
    return {
        "overall": fresh_document_bucket(),
        "answer_changing_edit": fresh_document_bucket(),
        "non_answer_changing_edit": fresh_document_bucket(),
    }


def merge_document_bucket(dst, src):
    for key in [
        "total",
        "correct",
        "hits",
        "misses",
        "true_hits",
        "false_hits",
        "model_failures",
    ]:
        dst[key] += src.get(key, 0)

    dst["latency_ms_sum"] += src.get("latency_ms_sum", 0.0)

    for key, value in src.get("update_counts", {}).items():
        if key not in dst["update_counts"]:
            dst["update_counts"][key] = 0
        dst["update_counts"][key] += value


def merge_document_metrics(all_metrics):
    merged = fresh_document_metrics()
    for metrics in all_metrics:
        for bucket_name in merged.keys():
            merge_document_bucket(merged[bucket_name], metrics[bucket_name])
    return merged


def summarize_document_bucket(bucket):
    total = max(bucket["total"], 1)

    return {
        "total": bucket["total"],
        "accuracy": bucket["correct"] / total,
        "hits": bucket["hits"],
        "misses": bucket["misses"],
        "hit_rate": bucket["hits"] / total,
        "true_hits": bucket["true_hits"],
        "false_hits": bucket["false_hits"],
        "false_hit_rate_overall": bucket["false_hits"] / total,
        "avg_latency_ms": bucket["latency_ms_sum"] / total,
        "model_failures": bucket["model_failures"],
        "update_counts": bucket["update_counts"],
    }


def summarize_document_metrics(merged_metrics):
    return {
        "overall": summarize_document_bucket(merged_metrics["overall"]),
        "answer_changing_edit": summarize_document_bucket(merged_metrics["answer_changing_edit"]),
        "non_answer_changing_edit": summarize_document_bucket(merged_metrics["non_answer_changing_edit"]),
    }


# ----------------------------
# Dialogue metrics
# ----------------------------

def fresh_dialogue_bucket():
    return {
        "total": 0,
        "correct_answers": 0,
        "incorrect_answers": 0,
        "hits": 0,
        "misses": 0,
        "true_hits": 0,
        "false_hits": 0,
        "model_failures": 0,
        "latency_ms_sum": 0.0,
        "transition_counts": {
            "none": 0,
            "domain_shift": 0,
            "intent_shift": 0,
            "constraint_shift": 0,
        },
    }


def fresh_dialogue_metrics():
    return {
        "overall": fresh_dialogue_bucket(),
        "state_preserving": fresh_dialogue_bucket(),
        "state_changing": fresh_dialogue_bucket(),
    }


def merge_dialogue_bucket(dst, src):
    for key in [
        "total",
        "correct_answers",
        "incorrect_answers",
        "hits",
        "misses",
        "true_hits",
        "false_hits",
        "model_failures",
    ]:
        dst[key] += src.get(key, 0)

    dst["latency_ms_sum"] += src.get("latency_ms_sum", 0.0)

    for key, value in src.get("transition_counts", {}).items():
        if key not in dst["transition_counts"]:
            dst["transition_counts"][key] = 0
        dst["transition_counts"][key] += value


def merge_dialogue_metrics(all_metrics):
    merged = fresh_dialogue_metrics()
    for metrics in all_metrics:
        for bucket_name in merged.keys():
            merge_dialogue_bucket(merged[bucket_name], metrics[bucket_name])
    return merged


def summarize_dialogue_bucket(bucket):
    total = max(bucket["total"], 1)

    return {
        "total": bucket["total"],
        "accuracy": bucket["correct_answers"] / total,
        "hits": bucket["hits"],
        "misses": bucket["misses"],
        "hit_rate": bucket["hits"] / total,
        "true_hits": bucket["true_hits"],
        "false_hits": bucket["false_hits"],
        "false_hit_rate_overall": bucket["false_hits"] / total,
        "avg_latency_ms": bucket["latency_ms_sum"] / total,
        "model_failures": bucket["model_failures"],
        "transition_counts": bucket["transition_counts"],
    }


def summarize_dialogue_metrics(merged_metrics):
    return {
        "overall": summarize_dialogue_bucket(merged_metrics["overall"]),
        "state_preserving": summarize_dialogue_bucket(merged_metrics["state_preserving"]),
        "state_changing": summarize_dialogue_bucket(merged_metrics["state_changing"]),
    }


# ----------------------------
# Shared helpers
# ----------------------------

def print_summary(name, summary):
    for bucket_name, stats in summary.items():
        print(f"\n===== {name} | {bucket_name.upper()} =====")
        print(f"total: {stats['total']}")
        print(f"accuracy: {stats['accuracy']:.3f}")
        print(f"hits: {stats['hits']}")
        print(f"misses: {stats['misses']}")
        print(f"hit_rate: {stats['hit_rate']:.3f}")
        print(f"true_hits: {stats['true_hits']}")
        print(f"false_hits: {stats['false_hits']}")
        print(f"false_hit_rate_overall: {stats['false_hit_rate_overall']:.3f}")
        print(f"avg_latency_ms: {stats['avg_latency_ms']:.2f}")
        print(f"model_failures: {stats['model_failures']}")

        if "update_counts" in stats:
            print(f"update_counts: {stats['update_counts']}")
        if "transition_counts" in stats:
            print(f"transition_counts: {stats['transition_counts']}")


def aggregate_experiment(input_files, output_file, display_name, kind):
    all_metrics = []

    for file_path in input_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_metrics.append(data["metrics"])

    if kind == "document":
        merged = merge_document_metrics(all_metrics)
        summary = summarize_document_metrics(merged)
    elif kind == "dialogue":
        merged = merge_dialogue_metrics(all_metrics)
        summary = summarize_dialogue_metrics(merged)
    else:
        raise ValueError(f"Unknown experiment kind: {kind}")

    print_summary(display_name, summary)

    output_payload = {
        "kind": kind,
        "input_files": input_files,
        "summary": summary,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nWrote merged summary to {output_file}")


def main():
    for experiment_name, config in EXPERIMENTS.items():
        print(f"\n\n########## AGGREGATING {experiment_name} ##########")
        aggregate_experiment(
            input_files=config["input_files"],
            output_file=config["output_file"],
            display_name=config["display_name"],
            kind=config["kind"],
        )


if __name__ == "__main__":
    main()