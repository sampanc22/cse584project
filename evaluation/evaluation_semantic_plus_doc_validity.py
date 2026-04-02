import argparse
import random
import time
import json

from evaluation_common import (
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

def run_semantic_plus_doc_validity_batch(
    batch_idx: int,
    dataset_path: str = DATASET_PATH,
    max_examples: int = 100,
) -> None:
    random.seed(RANDOM_SEED)

    max_examples = MAX_EXAMPLES

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
        doc_title = req["doc_title"]
        gold_answers = req["gold_answers"]
        update_type = req["update_type"]
        tag = req["tag"]
        original_answer = req.get("original_answer", "")
        new_answer = req.get("new_answer", "")
        edit_description = req.get("edit_description", "")

        print(f"\n[SEMANTIC+DOC VALIDITY BATCH {batch_idx} {i}/{len(stream)}] {tag} | update={update_type}")
        print(f"Q: {question}")
        if update_type != "none":
            print(f"EDIT: {edit_description}")
        if update_type == "answer_changing_edit":
            print(f"ORIGINAL ANSWER: {original_answer}")
            print(f"NEW ANSWER: {new_answer}")

        start = time.perf_counter()
        pred, hit = runner.answer_semantic_plus_doc_validity(
            question,
            context,
            doc_title,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        failed = is_model_failure(pred)
        correct = (not failed) and answer_matches(pred, gold_answers)

        update_metrics(
            metrics,
            correct=correct,
            cache_hit=hit,
            update_type=update_type,
            latency_ms=latency_ms,
            model_failed=failed,
        )

        print(
            f"  semantic+doc_validity -> {pred!r} | "
            f"hit={hit} | failed={failed} | correct={correct}"
        )

        sleep_between_requests()

    output_json_path = f"tmp/semantic_plus_doc_validity_batch_{batch_idx}.json"

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

    run_semantic_plus_doc_validity_batch(batch_idx=args.batch_idx, max_examples=100)