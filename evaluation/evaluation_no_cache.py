import json
import random
import time
import argparse

from evaluation.evaluation_common import (
    DATASET_PATH,
    MAX_EXAMPLES,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_ANSWER_CHANGING_VARIANTS,
    INCLUDE_IRRELEVANT_EDIT,
    EvalRunner,
    answer_matches,
    build_cache_stream,
    fresh_metrics,
    get_or_create_sampled_examples,
    get_batch,
    is_model_failure,
    sleep_between_requests,
    update_metrics,
)

def run_no_cache_batch(
    batch_idx: int,
    dataset_path: str = DATASET_PATH,
    max_examples: int = 100,
) -> None:
    random.seed(RANDOM_SEED)

    examples = get_or_create_sampled_examples(
        dataset_path=dataset_path,
        max_examples=max_examples,
    )

    batch_examples = get_batch(
        examples=examples,
        batch_idx=batch_idx,
        batch_size=BATCH_SIZE,
    )

    print(f"Loaded {len(examples)} total sampled examples")
    print(f"Running batch {batch_idx} with {len(batch_examples)} examples")

    stream = build_cache_stream(
        batch_examples,
        num_answer_changing_variants=NUM_ANSWER_CHANGING_VARIANTS,
        include_irrelevant_edit=INCLUDE_IRRELEVANT_EDIT,
    )
    print(f"Built stream of {len(stream)} requests")

    runner = EvalRunner()
    metrics = fresh_metrics()

    for i, req in enumerate(stream, start=1):
        question = req["question"]
        context = req["context"]
        gold_answers = req["gold_answers"]
        update_type = req["update_type"]
        tag = req["tag"]
        original_answer = req.get("original_answer", "")
        new_answer = req.get("new_answer", "")
        edit_description = req.get("edit_description", "")

        print(f"\n[NO CACHE BATCH {batch_idx} {i}/{len(stream)}] {tag} | update={update_type}")
        print(f"Q: {question}")
        if update_type != "none":
            print(f"EDIT: {edit_description}")
        if update_type == "answer_changing_edit":
            print(f"ORIGINAL ANSWER: {original_answer}")
            print(f"NEW ANSWER: {new_answer}")

        start = time.perf_counter()
        pred = runner.answer_no_cache(
            question,
            context,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        failed = is_model_failure(pred)
        correct = (not failed) and answer_matches(pred, gold_answers)

        update_metrics(
            metrics,
            correct=correct,
            cache_hit=False,
            update_type=update_type,
            latency_ms=latency_ms,
            model_failed=failed,
        )

        print(
            f"  semantic only -> {pred!r} | "
            f"hit={False} | failed={failed} | correct={correct}"
        )

        sleep_between_requests()

    output_json_path = f"tmp/no_cache_batch_{batch_idx}.json"

    payload = {
        "config": {
            "batch_idx": batch_idx,
            "batch_size": BATCH_SIZE,
        },
        "metrics": metrics,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote batch JSON to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-idx", type=int, required=True)
    args = parser.parse_args()

    run_no_cache_batch(batch_idx=args.batch_idx, max_examples=MAX_EXAMPLES)