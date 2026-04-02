import json
from pathlib import Path

EXPERIMENTS = {
    "no_cache": {
        "input_files": [
            "tmp/no_cache_batch_0.json",
            "tmp/no_cache_batch_1.json",
            "tmp/no_cache_batch_2.json",
            "tmp/no_cache_batch_3.json",
            "tmp/no_cache_batch_4.json",
        ],
        "output_file": "results/no_cache_summary.json",
        "display_name": "NO CACHE (MERGED)",
    },
    "semantic_only": {
        "input_files": [
            "tmp/semantic_only_batch_0.json",
            "tmp/semantic_only_batch_1.json",
            "tmp/semantic_only_batch_2.json",
            "tmp/semantic_only_batch_3.json",
            "tmp/semantic_only_batch_4.json",
        ],
        "output_file": "results/semantic_only_summary.json",
        "display_name": "SEMANTIC ONLY (MERGED)",
    },
    "semantic_plus_doc_validity": {
        "input_files": [
            "tmp/semantic_plus_doc_validity_batch_0.json",
            "tmp/semantic_plus_doc_validity_batch_1.json",
            "tmp/semantic_plus_doc_validity_batch_2.json",
            "tmp/semantic_plus_doc_validity_batch_3.json",
            "tmp/semantic_plus_doc_validity_batch_4.json",
        ],
        "output_file": "results/semantic_plus_doc_validity_summary.json",
        "display_name": "SEMANTIC + DOC VALIDITY (MERGED)",
    },
}

Path("results").mkdir(parents=True, exist_ok=True)

def fresh_bucket():
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


def fresh_metrics():
    return {
        "overall": fresh_bucket(),
        "answer_changing_edit": fresh_bucket(),
        "non_answer_changing_edit": fresh_bucket(),
    }


def merge_bucket(dst, src):
    for k in ["total", "correct", "hits", "misses", "true_hits", "false_hits", "model_failures"]:
        dst[k] += src.get(k, 0)

    dst["latency_ms_sum"] += src.get("latency_ms_sum", 0.0)

    for k, v in src.get("update_counts", {}).items():
        if k not in dst["update_counts"]:
            dst["update_counts"][k] = 0
        dst["update_counts"][k] += v


def merge_metrics(all_metrics):
    merged = fresh_metrics()

    for m in all_metrics:
        for bucket in merged.keys():
            merge_bucket(merged[bucket], m[bucket])

    return merged


def summarize(bucket):
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


def summarize_all(merged_metrics):
    return {
        "overall": summarize(merged_metrics["overall"]),
        "answer_changing_edit": summarize(merged_metrics["answer_changing_edit"]),
        "non_answer_changing_edit": summarize(merged_metrics["non_answer_changing_edit"]),
    }


def print_summary(name, summary):
    for bucket_name, s in summary.items():
        print(f"\n===== {name} | {bucket_name.upper()} =====")
        print(f"total: {s['total']}")
        print(f"accuracy: {s['accuracy']:.3f}")
        print(f"hits: {s['hits']}")
        print(f"misses: {s['misses']}")
        print(f"hit_rate: {s['hit_rate']:.3f}")
        print(f"true_hits: {s['true_hits']}")
        print(f"false_hits: {s['false_hits']}")
        print(f"false_hit_rate_overall: {s['false_hit_rate_overall']:.3f}")
        print(f"avg_latency_ms: {s['avg_latency_ms']:.2f}")
        print(f"model_failures: {s['model_failures']}")
        print(f"update_counts: {s['update_counts']}")


def aggregate_experiment(input_files, output_file, display_name):
    all_metrics = []

    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_metrics.append(data["metrics"])

    merged = merge_metrics(all_metrics)
    summary = summarize_all(merged)

    print_summary(display_name, summary)

    output_payload = {
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
        )


if __name__ == "__main__":
    main()