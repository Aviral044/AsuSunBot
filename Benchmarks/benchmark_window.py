import os
import sys
import asyncio

# Fix for the noisy Windows "Event loop is closed" error
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import time
import pandas as pd
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.run_config import RunConfig

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# NEW IMPORT: This is the magic tool that replaces the single sentence with the full window
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# --- Load Environment Variables & API Keys ---
load_dotenv()

try:
    NUM_KEYS = int(os.getenv('n', 0))
except ValueError:
    raise ValueError("The variable 'n' in your .env file must be an integer.")

if NUM_KEYS == 0:
    raise ValueError("Variable 'n' not found in .env or set to 0!")

API_KEYS = []
for i in range(1, NUM_KEYS + 1):
    key = os.getenv(f'key{i}')
    if not key:
        raise ValueError(f"key{i} not found in .env!")
    API_KEYS.append(key)

current_key_idx = 0

# --- Configuration ---
DATASET_PATH = r"C:\Stuff\AsuSunBot\data\golden_dataset.csv"
PERSIST_DIR = r"C:\Stuff\AsuSunBot\llamachromadb"

# CHANGE 1: Point to the new Window database
COLLECTION_NAME = "asulib_window" 

TEST_LLM_NAME = "openai/gpt-oss-120b"
TEST_EMBED_NAME = "BAAI/bge-small-en-v1.5"
JUDGE_LLM_NAME = "llama-3.3-70b-versatile"

def setup_window_bot(api_key):
    """Recreates the LlamaIndex setup, but with Sentence Window Post-Processing."""
    embed_model = HuggingFaceEmbedding(model_name=TEST_EMBED_NAME, device="cpu")
    llm = Groq(model=TEST_LLM_NAME, api_key=api_key)

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME) 
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )

    system_prompt = (
        "You are ASU Sun bot, the AI assistant for the Arizona State University (ASU) Library. Respond supportively and professionally like a peer mentor. \n"
        "1. Keep responses detailed as possible\n"
        "2. Do not make assumptions or fabricate answers or URLs. \n"
        "3. Use ONLY the retrieved context. If the database is insufficient, say you don't know and refer users to Ask a Librarian.\n"
        "Context:\n{context}"
    )

    # CHANGE 2: Initialize the postprocessor to grab the surrounding context
    window_postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        system_prompt=system_prompt,
        verbose=False,
        # Inject the postprocessor into the retrieval pipeline
        node_postprocessors=[window_postprocessor]
    )
    return chat_engine

def run_evaluation():
    global current_key_idx
    
    print(f"\nLoading Golden Dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    questions = df['Question'].tolist()
    ground_truths = df['Ground_Truth'].tolist()
    
    answers = []
    contexts = []

    print(f"Initializing bot with Key {current_key_idx + 1}...")
    bot = setup_window_bot(API_KEYS[current_key_idx])

    print("\n--- Generating Answers from your Bot ---")
    
    for q in tqdm(questions, desc="Querying ASU Bot", unit="question"):
        time.sleep(3)  
        
        success = False
        while not success:
            try:
                response = bot.chat(q)
                answers.append(response.response)
                
                # Extract retrieved text chunks
                retrieved_texts = [node.node.text for node in response.source_nodes]
                contexts.append(retrieved_texts)
                
                bot.reset() 
                success = True 
                
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "timeout" in error_msg:
                    current_key_idx = (current_key_idx + 1) % NUM_KEYS
                    tqdm.write(f"\n⚠️ Generation rate limit hit! Switching to API Key {current_key_idx + 1}...")
                    bot = setup_window_bot(API_KEYS[current_key_idx])
                else:
                    raise e

    print("\n--- Evaluating with Ragas ---")
    
    judge_embeddings = HuggingFaceEmbeddings(
        model_name=TEST_EMBED_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    answer_relevancy.strictness = 1
    ragas_config = RunConfig(max_workers=1, max_retries=3)

    all_evaluation_results = []

    for i in tqdm(range(len(questions)), desc="Grading with Ragas", unit="question"):
        
        single_data = {
            "question": [questions[i]],
            "answer": [answers[i]],
            "contexts": [contexts[i]],
            "ground_truth": [ground_truths[i]]
        }
        single_dataset = Dataset.from_dict(single_data)
        
        success = False
        while not success:
            try:
                judge_llm = ChatGroq(
                    model_name=JUDGE_LLM_NAME, 
                    groq_api_key=API_KEYS[current_key_idx],
                    temperature=0.0
                )
                
                res = evaluate(
                    dataset=single_dataset, 
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                    llm=judge_llm,
                    embeddings=judge_embeddings,
                    run_config=ragas_config,
                    show_progress=False,
                    raise_exceptions=True
                )
                
                all_evaluation_results.append(res.to_pandas())
                success = True
                
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "timeout" in error_msg:
                    current_key_idx = (current_key_idx + 1) % NUM_KEYS
                    tqdm.write(f"\n⚠️ Eval rate limit hit! Switching to API Key {current_key_idx + 1}...")
                else:
                    raise e

    print("\n=== SENTENCE WINDOW BENCHMARK RESULTS ===")
    
    final_results_df = pd.concat(all_evaluation_results, ignore_index=True)
    avg_scores = final_results_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean()
    print(avg_scores.to_string())
    
    # CHANGE 3: Output to a new CSV file
    output_file = "window_results.csv"
    final_results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to '{output_file}'")

if __name__ == "__main__":
    run_evaluation()