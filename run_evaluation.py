from evaluation.evaluation_common import NUM_BATCHES
import subprocess
import sys
import time
import shutil

# no cache baseline experiment
for i in range(NUM_BATCHES):
    print(f"\n===== RUNNING BATCH {i} =====")

    result = subprocess.run(
        ["python", "-m", "evaluation.evaluation_no_cache", "--batch-idx", str(i)]
    )

    if result.returncode != 0:
        print(f"Batch {i} failed. Stopping.")
        sys.exit(1)

    time.sleep(5)

# semantic only baseline experiment
for i in range(NUM_BATCHES):
    print(f"\n===== RUNNING BATCH {i} =====")

    result = subprocess.run(
        ["python", "-m", "evaluation.evaluation_semantic_only", "--batch-idx", str(i)]
    )

    if result.returncode != 0:
        print(f"Batch {i} failed. Stopping.")
        sys.exit(1)

    time.sleep(5)

# our experiment = semantic + doc validity
for i in range(NUM_BATCHES):
    print(f"\n===== RUNNING BATCH {i} =====")

    result = subprocess.run(
        ["python", "-m", "evaluation.evaluation_semantic_plus_doc_validity", "--batch-idx", str(i)]
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

# clean up
print("\n===== DONE =====")
print("\nCleaning up individual batch summaries")
if shutil.os.path.exists("tmp"):
    shutil.rmtree("tmp")