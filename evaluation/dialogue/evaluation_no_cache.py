import argparse
import random
import time
import json

from .evaluation_common import (
    DIALOGUE_DATASET_PATH,
    MAX_DIALOGUES,
    RANDOM_SEED,
    BATCH_SIZE,
    DialogueEvalRunner,
    build_dialogue_cache_stream,
    fresh_dialogue_metrics,
    get_or_create_dialogue_sampled_trajectories,
    get_batch,
    is_model_failure,
    sleep_between_requests,
    update_dialogue_cache_metrics,
    dialogue_answer_accuracy_match,
)


def run_dialogue_no_cache_batch(
    batch_idx: int,
    dataset_path: str = DIALOGUE_DATASET_PATH,
    max_dialogues: int = MAX_DIALOGUES,
) -> None:
    random.seed(RANDOM_SEED)

    trajectories = get_or_create_dialogue_sampled_trajectories(
        dataset_path=dataset_path,
        max_dialogues=max_dialogues,
    )

    batch_trajectories = get_batch(
        items=trajectories,
        batch_idx=batch_idx,
        batch_size=BATCH_SIZE,
    )

    print(f"Loaded {len(trajectories)} total sampled dialogue trajectories")
    print(f"Running no-cache dialogue batch {batch_idx} with {len(batch_trajectories)} trajectories")

    stream = build_dialogue_cache_stream(batch_trajectories)
    print(f"Built dialogue stream of {len(stream)} requests")

    runner = DialogueEvalRunner()
    metrics = fresh_dialogue_metrics()

    for i, req in enumerate(stream, start=1):
        question = req["question"]
        history = req["history"]
        latest_user_turn = req["latest_user_turn"]
        context = req.get("context", "")
        gold_answers = req["gold_answers"]
        transition_type = req["transition_type"]
        tag = req["tag"]
        dialogue_id = req.get("dialogue_id", "")
        turn_id = req.get("turn_id", "")
        domain = req.get("domain", "")
        intent = req.get("intent", "")

        print(
            f"\n[DIALOGUE NO-CACHE BATCH {batch_idx} {i}/{len(stream)}] "
            f"{tag} | transition={transition_type}"
        )
        print(f"dialogue_id={dialogue_id} turn_id={turn_id}")
        print(f"domain={domain} intent={intent}")
        print(f"QUESTION: {question}")
        print(f"LATEST USER TURN: {latest_user_turn}")
        if history:
            print(f"HISTORY TAIL: {history[-2:]}")
        print(f"GOLD ANSWERS: {gold_answers}")

        start = time.perf_counter()
        pred = runner.answer_no_cache(
            question=question,
            history=history,
            latest_user_turn=latest_user_turn,
            context=context,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        failed = is_model_failure(pred)
        answer_correct = (not failed) and dialogue_answer_accuracy_match(pred, gold_answers)

        update_dialogue_cache_metrics(
            metrics,
            cache_hit=False,
            hit_gold_valid=None,
            answer_correct=answer_correct,
            transition_type=transition_type,
            latency_ms=latency_ms,
            model_failed=failed,
        )

        print(
            f"  no-cache -> {pred!r} | "
            f"answer_correct={answer_correct} | failed={failed}"
        )

        sleep_between_requests()

    output_json_path = f"tmp/dialogue_no_cache_batch_{batch_idx}.json"

    payload = {
        "config": {
            "batch_idx": batch_idx,
            "batch_size": BATCH_SIZE,
            "policy": "no_cache",
            "sample_unit": "dialogue_trajectory",
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

    run_dialogue_no_cache_batch(
        batch_idx=args.batch_idx,
        max_dialogues=MAX_DIALOGUES,
    )