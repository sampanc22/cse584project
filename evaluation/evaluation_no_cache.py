from evaluation_common import (
    DATASET_PATH,
    MAX_EXAMPLES,
    RANDOM_SEED,
    REQUEST_SLEEP_SECONDS,
    EvalRunner,
    build_no_cache_stream,
    exact_match,
    fresh_metrics,
    init_csv_log,
    is_model_failure,
    get_or_create_sampled_examples,
    log_result,
    sleep_between_requests,
    summarize,
    update_metrics,
    write_summary_json,
)
import random

CSV_LOG_PATH = "no_cache_results.csv"
SUMMARY_JSON_PATH = "no_cache_summary.json"


def run_no_cache_baseline(
    dataset_path: str = DATASET_PATH,
    max_examples: int = MAX_EXAMPLES,
) -> None:
    random.seed(RANDOM_SEED)

    print(f"Loading dataset from: {dataset_path}")
    examples = get_or_create_sampled_examples(dataset_path=dataset_path, max_examples=max_examples)
    print(f"Loaded {len(examples)} QA examples for no-cache baseline")

    stream = build_no_cache_stream(examples)
    print(f"Built no-cache stream of {len(stream)} requests")

    init_csv_log(CSV_LOG_PATH)

    runner = EvalRunner()
    metrics = fresh_metrics()

    for i, req in enumerate(stream, start=1):
        question = req["question"]
        context = req["context"]
        doc_title = req["doc_title"]
        qa_id = req["qa_id"]
        gold_answers = req["gold_answers"]
        update_type = req["update_type"]
        changed_answer = req["changed_answer"]
        tag = req["tag"]

        print(f"\n[NO CACHE {i}/{len(stream)}] {tag}")
        print(f"Q: {question}")

        pred = runner.answer_no_cache(question, context)
        failed = is_model_failure(pred)
        correct = (not failed) and exact_match(pred, gold_answers)

        update_metrics(
            metrics,
            correct=correct,
            cache_hit=False,
            stale_avoided=False,
            changed_answer=changed_answer,
            update_type=update_type,
            model_failed=failed,
        )

        log_result(
            CSV_LOG_PATH,
            request_index=i,
            tag=tag,
            baseline="no_cache",
            qa_id=qa_id,
            doc_title=doc_title,
            question=question,
            gold_answers=gold_answers,
            prediction=pred,
            correct=correct,
            cache_hit=False,
            stale_avoided=False,
            update_type=update_type,
            changed_answer=changed_answer,
            context=context,
            model_failed=failed,
        )

        print(f"  no_cache -> {pred!r} | failed={failed} | correct={correct}")
        sleep_between_requests()

    summary_payload = {
        "config": {
            "dataset_path": dataset_path,
            "random_seed": RANDOM_SEED,
            "max_examples": max_examples,
            "request_sleep_seconds": REQUEST_SLEEP_SECONDS,
        },
        "results": {
            "no_cache": summarize("NO CACHE", metrics),
        },
    }

    write_summary_json(SUMMARY_JSON_PATH, summary_payload)

    print(f"\nWrote no-cache per-request log to {CSV_LOG_PATH}")
    print(f"Wrote no-cache summary to {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    run_no_cache_baseline()