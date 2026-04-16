from evaluation.document.evaluation_common import NUM_BATCHES as NUM_DOCUMENT_BATCHES
from evaluation.dialogue.evaluation_common import NUM_BATCHES as NUM_DIALOGUE_BATCHES
import subprocess
import sys
import time
import shutil

# no cache baseline experiment
for i in range(NUM_DOCUMENT_BATCHES):
    print(f"\n===== RUNNING BATCH {i} =====")

    result = subprocess.run(
        ["python", "-m", "evaluation.document.evaluation_no_cache", "--batch-idx", str(i)]
    )

    if result.returncode != 0:
        print(f"Batch {i} failed. Stopping.")
        sys.exit(1)

    time.sleep(5)

# semantic only baseline experiment
for i in range(NUM_DOCUMENT_BATCHES):
    print(f"\n===== RUNNING BATCH {i} =====")

    result = subprocess.run(
        ["python", "-m", "evaluation.document.evaluation_semantic_only", "--batch-idx", str(i)]
    )

    if result.returncode != 0:
        print(f"Batch {i} failed. Stopping.")
        sys.exit(1)

    time.sleep(5)

# our experiment = semantic + doc validity
for i in range(NUM_DOCUMENT_BATCHES):
    print(f"\n===== RUNNING BATCH {i} =====")

    result = subprocess.run(
        ["python", "-m", "evaluation.document.evaluation_semantic_plus_doc_validity", "--batch-idx", str(i)]
    )

    if result.returncode != 0:
        print(f"Batch {i} failed. Stopping.")
        sys.exit(1)

    time.sleep(5)

time.sleep(10)

# no cache baseline experiment
for i in range(NUM_DIALOGUE_BATCHES):
    print(f"\n===== RUNNING BATCH {i} =====")

    result = subprocess.run(
        ["python", "-m", "evaluation.dialogue.evaluation_no_cache", "--batch-idx", str(i)]
    )

    if result.returncode != 0:
        print(f"Batch {i} failed. Stopping.")
        sys.exit(1)

    time.sleep(5)

# semantic only baseline experiment
for i in range(NUM_DIALOGUE_BATCHES):
    # if i < 6: 
    #     continue
    print(f"\n===== RUNNING BATCH {i} =====")

    result = subprocess.run(
        ["python", "-m", "evaluation.dialogue.evaluation_semantic_only", "--batch-idx", str(i)]
    )

    if result.returncode != 0:
        print(f"Batch {i} failed. Stopping.")
        sys.exit(1)

    time.sleep(5)

# our experiment = semantic + dialogue validity
for i in range(NUM_DIALOGUE_BATCHES):
    print(f"\n===== RUNNING BATCH {i} =====")

    result = subprocess.run(
        ["python", "-m", "evaluation.dialogue.evaluation_semantic_plus_strict_dialogue_validity", "--batch-idx", str(i)]
    )

    if result.returncode != 0:
        print(f"Batch {i} failed. Stopping.")
        sys.exit(1)

    time.sleep(5)

time.sleep(10)

# our experiment = semantic + dialogue validity
for i in range(NUM_DIALOGUE_BATCHES):
    print(f"\n===== RUNNING BATCH {i} =====")

    result = subprocess.run(
        ["python", "-m", "evaluation.dialogue.evaluation_semantic_plus_slot_relaxed_dialogue_validity", "--batch-idx", str(i)]
    )

    if result.returncode != 0:
        print(f"Batch {i} failed. Stopping.")
        sys.exit(1)

    time.sleep(5)

time.sleep(10)

# our experiment = semantic + dialogue validity
for i in range(NUM_DIALOGUE_BATCHES):
    print(f"\n===== RUNNING BATCH {i} =====")

    result = subprocess.run(
        ["python", "-m", "evaluation.dialogue.evaluation_semantic_plus_intent_domain_dialogue_validity", "--batch-idx", str(i)]
    )

    if result.returncode != 0:
        print(f"Batch {i} failed. Stopping.")
        sys.exit(1)

    time.sleep(5)

time.sleep(10)

# aggregate results
print("\n===== MERGING RESULTS =====")
subprocess.run(["python", "-m", "evaluation.aggregate_batch_results"], check=True)

time.sleep(5)

subprocess.run(["python", "results/visualize.py"])

# clean up
print("\n===== DONE =====")
print("\nCleaning up individual batch summaries")
if shutil.os.path.exists("tmp"):
    shutil.rmtree("tmp")