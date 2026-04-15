import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ---- Config ----
FILES = {
    "No Cache": "dialogue_no_cache_summary.json",
    "Semantic Only": "dialogue_semantic_only_summary.json",
    "Intent Domain": "dialogue_semantic_plus_validity_intent_domain_summary.json",
    "Slot Relaxed": "dialogue_semantic_plus_validity_slot_relaxed_summary.json",
    "Strict": "dialogue_semantic_plus_validity_strict_summary.json"
}

CATEGORIES = [
    "overall",
    "state_preserving",
    "state_changing"
]

OUTPUT_DIR = "plots_dialogue"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---- Load + flatten ----
def load_all_metrics(files):
    rows = []

    for method, path in files.items():
        with open(path, "r") as f:
            data = json.load(f)

        summary = data["summary"]

        for category in CATEGORIES:
            stats = summary[category]

            rows.append({
                "method": method,
                "category": category,
                "accuracy": stats.get("accuracy", np.nan),
                "hit_rate": stats.get("hit_rate", np.nan),
                "false_hit_rate": stats.get("false_hit_rate_overall", np.nan),
                "latency_ms": stats.get("avg_latency_ms", np.nan),
                "true_hits": stats.get("true_hits", 0),
                "false_hits": stats.get("false_hits", 0)
            })

    df = pd.DataFrame(rows)

    # Ensure consistent method ordering
    df["method"] = pd.Categorical(
        df["method"],
        categories=list(files.keys()),
        ordered=True
    )

    return df


# ---- Precision (derived metric, SAFE) ----
def add_precision(df):
    denom = df["true_hits"] + df["false_hits"]
    df["precision"] = np.where(denom > 0, df["true_hits"] / denom, 0.0)
    return df


# ---- Generic grouped bar plot ----
def plot_grouped(df, metric, title, ylabel, filename):
    pivot = df.pivot(index="method", columns="category", values=metric)

    # Ensure consistent ordering
    pivot = pivot[CATEGORIES]

    pivot.plot(kind="bar")

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.legend(title="Category")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")


# ---- Main ----
def main():
    df = load_all_metrics(FILES)
    df = add_precision(df)

    print("\n=== Data ===")
    print(df)

    # ---- Plots ----
    plot_grouped(df, "accuracy",
                 "Accuracy by Method and Edit Type",
                 "Accuracy",
                 "accuracy.png")

    plot_grouped(df, "false_hit_rate",
                 "False Hit Rate by Method and Edit Type",
                 "False Hit Rate",
                 "false_hit_rate.png")
    
    plot_grouped(df, "hit_rate",
                 "Hit Rate by Method and Edit Type",
                 "Hit Rate",
                 "hit_rate.png")

    plot_grouped(df, "latency_ms",
                 "Latency by Method and Edit Type",
                 "Milliseconds",
                 "latency.png")

    plot_grouped(df, "precision",
                 "Precision by Method and Edit Type",
                 "Precision",
                 "precision.png")


if __name__ == "__main__":
    main()