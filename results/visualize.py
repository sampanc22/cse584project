import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---- Config ----
FILES = {
    "No Cache": "no_cache_summary.json",
    "Semantic Only": "semantic_only_summary.json",
    "Semantic + Validity": "semantic_plus_doc_validity_summary.json"
}

CATEGORIES = [
    "overall",
    "answer_changing_edit",
    "non_answer_changing_edit"
]

OUTPUT_DIR = "plots"
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
                "accuracy": stats["accuracy"],
                "false_hit_rate": stats["false_hit_rate_overall"],
                "latency_ms": stats["avg_latency_ms"],
                "true_hits": stats["true_hits"],
                "false_hits": stats["false_hits"]
            })

    return pd.DataFrame(rows)


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


# ---- Precision (derived metric) ----
def add_precision(df):
    df["precision"] = df["true_hits"] / (df["true_hits"] + df["false_hits"])
    return df


# ---- Tradeoff scatter (still useful) ----
from matplotlib.lines import Line2D

def plot_tradeoff(df):
    plt.figure(figsize=(8, 6))

    # Use REAL colors (not numbers)
    method_colors = {
        "No Cache": "tab:blue",
        "Semantic Only": "tab:orange",
        "Semantic + Validity": "tab:green"
    }

    category_markers = {
        "overall": "o",
        "answer_changing_edit": "s",
        "non_answer_changing_edit": "^"
    }

    # Plot points
    for _, row in df.iterrows():
        plt.scatter(
            row["latency_ms"],
            row["accuracy"],
            color=method_colors[row["method"]],
            marker=category_markers[row["category"]],
            s=100
        )

    # ---- Legend for methods (colors) ----
    method_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color, markersize=8, label=method)
        for method, color in method_colors.items()
    ]

    # ---- Legend for categories (markers) ----
    category_handles = [
        Line2D([0], [0], marker=marker, color='gray',
               linestyle='None', markersize=8, label=category)
        for category, marker in category_markers.items()
    ]

    # Add both legends
    legend1 = plt.legend(handles=method_handles, title="Method", loc="lower right")
    plt.gca().add_artist(legend1)

    plt.legend(handles=category_handles, title="Edit Type", loc="lower center")

    # Labels
    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Latency Tradeoff")

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "tradeoff_detailed.png")
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
                 "accuracy_detailed.png")

    plot_grouped(df, "false_hit_rate",
                 "False Hit Rate by Method and Edit Type",
                 "False Hit Rate",
                 "false_hit_rate_detailed.png")

    plot_grouped(df, "latency_ms",
                 "Latency by Method and Edit Type",
                 "Milliseconds",
                 "latency_detailed.png")

    plot_grouped(df, "precision",
                 "Precision by Method and Edit Type",
                 "Precision",
                 "precision_detailed.png")

    plot_tradeoff(df)


if __name__ == "__main__":
    main()