from evaluation_common import (
    MAX_EXAMPLES,
    DATASET_PATH,
    RANDOM_SEED,
    REQUEST_SLEEP_SECONDS,
    EvalRunner,
    build_cache_stream,
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

CSV_LOG_PATH = "semantic_plus_doc_validity_results.csv"
SUMMARY_JSON_PATH = "semantic_plus_doc_validity_summary.json"

NUM_ANSWER_CHANGING_VARIANTS = 2
INCLUDE_IRRELEVANT_EDIT = True


def run_semantic_plus_doc_validity(
    dataset_path: str = DATASET_PATH,
    max_examples: int = MAX_EXAMPLES,
) -> None:
    random.seed(RANDOM_SEED)

    print(f"Loading dataset from: {dataset_path}")
    examples = get_or_create_sampled_examples(dataset_path=dataset_path, max_examples=max_examples)
    print(f"Loaded {len(examples)} QA examples for semantic+doc-validity evaluation")

    stream = build_cache_stream(
        examples,
        num_answer_changing_variants=NUM_ANSWER_CHANGING_VARIANTS,
        include_irrelevant_edit=INCLUDE_IRRELEVANT_EDIT,
    )
    print(f"Built semantic+doc-validity stream of {len(stream)} requests")

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

        print(f"\n[SEMANTIC+DOC VALIDITY {i}/{len(stream)}] {tag} | update={update_type}")
        print(f"Q: {question}")

        pred, hit, stale_avoided = runner.answer_semantic_plus_doc_validity(
            question,
            context,
            doc_title,
        )
        failed = is_model_failure(pred)
        correct = (not failed) and exact_match(pred, gold_answers)

        update_metrics(
            metrics,
            correct=correct,
            cache_hit=hit,
            stale_avoided=stale_avoided,
            changed_answer=changed_answer,
            update_type=update_type,
            model_failed=failed,
        )

        log_result(
            CSV_LOG_PATH,
            request_index=i,
            tag=tag,
            baseline="semantic_plus_doc_validity",
            qa_id=qa_id,
            doc_title=doc_title,
            question=question,
            gold_answers=gold_answers,
            prediction=pred,
            correct=correct,
            cache_hit=hit,
            stale_avoided=stale_avoided,
            update_type=update_type,
            changed_answer=changed_answer,
            context=context,
            model_failed=failed,
        )

        print(
            f"  semantic+doc_validity -> {pred!r} | hit={hit} | "
            f"failed={failed} | correct={correct} | stale_avoided={stale_avoided}"
        )
        sleep_between_requests()

    summary_payload = {
        "config": {
            "dataset_path": dataset_path,
            "random_seed": RANDOM_SEED,
            "max_examples": max_examples,
            "request_sleep_seconds": REQUEST_SLEEP_SECONDS,
            "num_answer_changing_variants": NUM_ANSWER_CHANGING_VARIANTS,
            "include_irrelevant_edit": INCLUDE_IRRELEVANT_EDIT,
        },
        "results": {
            "semantic_plus_doc_validity": summarize("SEMANTIC + DOC VALIDITY", metrics),
        },
    }

    write_summary_json(SUMMARY_JSON_PATH, summary_payload)

    print(f"\nWrote semantic+doc-validity per-request log to {CSV_LOG_PATH}")
    print(f"Wrote semantic+doc-validity summary to {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    run_semantic_plus_doc_validity()