Validity-Aware Prompt/Response Caching for Large Language Models

1. Usage Instructions
    This project implements a validity-aware semantic caching system for large language models and includes evaluation pipelines for both document-based and dialogue-based workloads.

    To run the system, first follow the setup instructions provided in INSTALL.txt.

    Once the environment is set up, the evaluation scripts can be executed directly from the command line.

    For document-based evaluation using the SQuAD dataset, run:
    python -m evaluation.document.evaluation_no_cache
    python -m evaluation.document.evaluation_semantic_only
    python -m evaluation.document.evaluation_semantic_plus_doc_validity

    For dialogue-based evaluation using the MultiWOZ dataset, run:
    python -m evaluation.dialogue.evaluation_semantic_only
    python -m evaluation.dialogue.evaluation_semantic_plus_strict_dialogue_validity
    python -m evaluation.dialogue.evaluation_semantic_plus_slot_relaxed_dialogue_validity
    python -m evaluation.dialogue.evaluation_semantic_plus_intent_domain_dialogue_validity

    All individual evaluation scripts require a --batch-idx argument. 
    Examples of individual run w/ batch index parameter:
        python -m evaluation.document.evaluation_semantic_only --batch-idx 0
        python -m evaluation.dialogue.evaluation_semantic_plus_slot_relaxed_dialogue_validity --batch-idx 1
    
    To run all batches sequentially + run the entire pipeline with multiple batches and aggregate results, use:
    python run_evaluation.py -> this automatically handles batch indexes for you!

    The system expects input datasets in JSON format, and they have been provided within the source code. 
    However, there are also instructions in the INSTALL.txt file on how to download and set up the datasets.
    * SQuAD dataset for document-based evaluation (evaluation.document.squad.json)
    * MultiWOZ dataset for dialogue-based evaluation (evaluation.dialogue.data.json)

    Small example datasets are included in the repository and can be found at these locations:
        - evaluation/document/squad.json
        - evaluation/dialogue/data.json

    These files contain reduced subsets of the full datasets and can be used to quickly test the system without downloading the full datasets.
        Evaluation outputs include metrics such as accuracy, hit rate, false hit rate, latency, and failure counts. 
        These results are saved as JSON files in the project directory.


2. Code Structure and Major Functions
    The source code is organized into 3 main modules: core, evaluation, and results.

    The core functionality of the system is implemented in the core/ directory:
        * cache.py
        Implements the semantic cache. It stores prompts, responses, embeddings, and metadata. It performs cosine similarity calculations, retrieves top-k candidates, and manages cache insertion and lookup.

        * validity.py
        Implements validity checks for cache reuse. It includes logic for document-based validation using version matching and dialogue-based validation using structured dialogue signatures. It supports strict, slot-relaxed, and intent-domain matching.

        * signatures.py
        Constructs structured dialogue signatures by extracting domains, intents, constraints, and requested information from dialogue inputs. It also normalizes values to ensure consistent comparison.

        * document_registry.py
        Tracks document versions by assigning version identifiers based on document content. This allows the system to detect changes in documents and invalidate outdated cache entries.

        * adk_runtime.py
        Handles interaction with the Gemini LLM. It generates fresh responses when cache reuse is not valid and includes retry logic for handling API failures.

    The evaluation pipeline is implemented in the evaluation/ directory:
        * aggregate_batch_results.py:
        Aggregates metrics from all per-batch evaluation output files and produces final summary results for each experiment configuration. 
        It merges batch-level statistics, computes overall metrics such as accuracy, hit rate, false hit rate, and average latency, and writes the summarized results.

        * evaluation_common.py
        Contains shared utilities for dataset loading, batching, request stream generation, and metric tracking.
        There are 2 versions for this file --> 1 for document workloads and 1 for dialogue workloads.

            These are some important utility functions within evaluation_common.py:
                * build_cache_stream(...)
                Generates sequences of evaluation requests. For document workloads, it simulates no-change, irrelevant-edit, and answer-changing scenarios. 
                For dialogue workloads, it simulates state-preserving and state-changing transitions.

                * update_metrics(...)
                Tracks evaluation metrics such as accuracy, hit rate, false hit rate, latency, and failures for each request.

            There are 2 subdirectories within evaluation to represent the 2 types of workloads: document and dialogue

            Within the document/ directory:

                * evaluation_no_cache.py
                Runs evaluation without caching (baseline).

                * evaluation_semantic_only.py
                Runs evaluation using only semantic similarity for cache reuse.

                * evaluation_semantic_plus_doc_validity.py
                Runs evaluation using semantic caching with document validity checks.

            Within the dialogue/ directory:
                * evaluation_no_cache.py
                Runs evaluation without caching (baseline).

                * evaluation_semantic_only.py
                Runs evaluation using only semantic similarity for cache reuse.

                * evaluation_semantic_plus_strict_dialogue_validity.py
                Runs evaluation using semantic caching with dialogue validity check (exactly strict strictness level)

                * evaluation_semantic_plus_slot_relaxed_dialogue_validity.py
                Runs evaluation using semantic caching with dialogue validity check (slot relaxed strictness level)

                * evaluation_semantic_plus_intent_domain_dialogue_validity.py
                Runs evaluation using semantic caching with dialogue validity check (intent domain strictness level)

    The results of our evaluation pipeline are stored in the results/ directory:
        * visualize_docs.py and visualize_dialogue.py
        Creates visualizations/graphs based on summarized metrics (which are the JSON files stored in results/ directory). 
        The graphs are found in the respective folder: plots_docs for documents and plots_dialogue for dialogue.
        Graphs include accuracy, false hit rate, hit rate, latency, and precision.


3. Execution flow of the system is as follows:

    1. Load dataset (SQuAD or MultiWOZ)
    2. Generate a stream of requests with simulated edits or transitions
    3. For each request:
        * Compute the query embedding
        * Retrieve the top-k semantic cache candidates
        * Apply validity checks (document or dialogue)
        * Reuse cached response if valid, otherwise make a fresh call to the LLM
    4. Store results in the cache and update evaluation metrics

    This design allows the system to improve efficiency through caching while maintaining correctness through explicit validity checks.
