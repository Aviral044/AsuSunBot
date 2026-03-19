# 📊 ASU Sun Bot: RAG Benchmarking & Evaluation Methodology

This document outlines the evaluation framework used to mathematically score, test, and optimize the ASU Sun Bot's Retrieval-Augmented Generation (RAG) pipeline. 

Rather than relying on human "vibe checks," this project uses an automated **LLM-as-a-Judge** methodology to rigorously grade the bot's accuracy, helpfulness, and retrieval capabilities against a curated golden dataset.

## 🎯 1. The Golden Dataset
At the core of our benchmarking is the `golden_dataset.csv`. This is a hand-curated exam containing 47 domain-specific questions about the Arizona State University Library, alongside perfect human-written "Ground Truth" answers. 

The dataset covers various difficulty tiers, including:
* Standard FAQs
* Specific Database queries
* Research Guide routing
* Complex timetable/hours logic

## ⚖️ 2. Evaluation Framework (Ragas)
To grade the bot at scale, we utilize the **Ragas** (Retrieval Augmented Generation Assessment) library. 

We employ a highly capable LLM (`llama-3.3-70b-versatile`) acting as an impartial judge. For every question in the dataset, the judge evaluates the bot's final answer and its retrieved database chunks across four strict mathematical metrics (scored 0.0 to 1.0):

#### Generation Metrics (Checking the Brain)
1. **Faithfulness (Hallucination Check):** Measures if the bot's final answer is entirely backed up by the retrieved context. Penalizes fabricated information.
2. **Answer Relevancy:** Measures how directly the bot answered the user's original question without being evasive or going on unhelpful tangents.

#### Retrieval Metrics (Checking the Search Engine)
3. **Context Precision:** Checks if the database successfully pulled the document containing the actual answer and ranked it at the very top of the pile.
4. **Context Recall:** Compares the retrieved documents against the human "Ground Truth" to ensure the retriever didn't miss any critical pieces of the puzzle.

## 🗺️ 3. The Testing Roadmap
We are executing a structured A/B testing roadmap to find the ultimate combination of speed, cost, and accuracy.

* **✅ Phase 1: The Baseline (Completed)**
  * **Setup:** Naive RAG (Basic Document Chunking).
  * **LLM:** `openai/gpt-oss-120b`
  * **Embeddings:** `BAAI/bge-small-en-v1.5`
  * **Goal:** Establish the minimum viable scores to beat.

* **🔄 Phase 2: Advanced Retrieval Tactics (In Progress)**
  * **Sentence Window Retrieval:** Embedding individual sentences but feeding the LLM a "window" of surrounding sentences for highly precise, context-rich retrieval.
  * **Hybrid Search (BM25 + Vectors):** Combining semantic meaning (ChromaDB vectors) with exact-keyword matching (BM25) to better handle specific noun searches (e.g., "JSTOR", "PubMed").

* **⏳ Phase 3: Model Ablation (Upcoming)**
  * Swapping out the massive 120B model for smaller, blazing-fast models (like `llama-3.1-8b-instant`) to see if we can maintain the same Ragas scores at a fraction of the compute cost.

* **⏳ Phase 4: Embedding Upgrades (Upcoming)**
  * Upgrading the vector map from `bge-small` (384 dimensions) to `bge-large` (1024 dimensions) to fix any missing documents highlighted by low Context Recall scores.

## 🛠️ 4. Technical Workarounds: Beating the Rate Limits
Evaluating RAG systems requires massive amounts of text generation, which easily triggers strict API rate limits (HTTP 429) and Daily Token quotas on free-tier platforms like Groq.

To automate the benchmark without crashing, our script implements a **Dynamic API Key Rotation System**:
1. A `.env` file holds a pool of n different API keys.
2. The script runs both the Generation and Evaluation phases inside custom `try/except` loops.
3. If the active key hits a timeout or token limit, the script intercepts the crash, seamlessly rotates to the next available key in the array, rebuilds the internal LlamaIndex components, and retries the exact question without losing any progress.

## 🚀 How to Run the Benchmarks

1. Ensure your `.env` is populated with your API keys:
   ```text
   n=3
   key1=gsk_your_first_key...
   key2=gsk_your_second_key...
   key3=gsk_your_third_key...
   ```
2. Build your target index (e.g., Baseline, Window, or Hybrid):
   ```bash
   python build_window_index.py
   ```
3. Run the evaluation suite:
   ```bash
   python run_benchmark.py
   ```
4. Check the generated `baseline_results.csv` for a question-by-question breakdown of where the bot succeeded or failed.

Naive RAG results:
=== BASELINE BENCHMARK RESULTS ===
faithfulness         0.538113
answer_relevancy     0.846330
context_precision    0.702128
context_recall       0.638298

=== SENTENCE WINDOW BENCHMARK RESULTS ===
faithfulness         0.487418
answer_relevancy     0.790733
context_precision    0.914894
context_recall       0.889016

=== HYBRID SEARCH BENCHMARK RESULTS ===
faithfulness         0.515784
answer_relevancy     0.829717
context_precision    0.946809
context_recall       0.968085

=== HYBRID SEARCH BENCHMARK RESULTS ===
faithfulness         0.659163
answer_relevancy     0.833863
context_precision    0.946809
context_recall       0.968085

Detailed results saved to 'hybrid_results.csv'

Grading with Ragas: 100%|██████████████████████████████████████████████████████████████████████████████████| 47/47 [21:10<00:00, 27.03s/question]

=== HYBRID SEARCH BENCHMARK RESULTS ===
faithfulness         0.788645
answer_relevancy     0.872636
context_precision    0.946809
context_recall       0.968085

Detailed results saved to 'hybrid_70b_results.csv'