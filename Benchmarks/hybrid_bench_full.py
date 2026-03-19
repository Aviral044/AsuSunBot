import os
import sys
import asyncio
import time
import datetime
import pandas as pd
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv

# Fix for the noisy Windows "Event loop is closed" error
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

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
PERSIST_DIR = r"C:\Stuff\AsuSunBot\llamachromadb_hybrid"
COLLECTION_NAME = "asulib_hybrid" 

TEST_LLM_NAME = "openai/gpt-oss-120b"
TEST_EMBED_NAME = "BAAI/bge-small-en-v1.5"
JUDGE_LLM_NAME = "llama-3.3-70b-versatile"

def setup_hybrid_bot(api_key):
    embed_model = HuggingFaceEmbedding(model_name=TEST_EMBED_NAME, device="cpu")
    llm = Groq(model=TEST_LLM_NAME, api_key=api_key)

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME) 
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR,
        vector_store=vector_store
    )
    
    index = load_index_from_storage(
        storage_context=storage_context,
        embed_model=embed_model
    )

    today = datetime.date.today().strftime('%A, %B %d, %Y')

    system_prompt = (
        f"You are ASU Sun bot, the AI assistant for the Arizona State University (ASU) Library. Respond supportively and professionally like a peer mentor. \n\n"
        f"Guidelines: \n"
        f"1. No creative content (stories, poems, tweets, code). \n"
        f"2. Simple jokes are allowed, but avoid jokes that could hurt any group. \n"
        f"3. Use up to two emojis when applicable. \n"
        f"4. Provide relevant search terms if asked. \n"
        f"5. Avoid providing information about celebrities, influential politicians, or state heads. \n"
        f"6. Answer fully using the provided context, but be concise if the context is short.\n"
        f"7. For unanswerable research questions, include the 'Ask A Librarian' URL: https://askalibrarian.asu.edu/ \n"
        f"8. Do not make assumptions or fabricate answers or URLs. \n"
        f"9. Use ONLY the retrieved context. If the database is insufficient, say you don't know and refer users to Ask a Librarian. \n"
        f"10. Do not provide specific book recommendations; instead, direct the user to search an ASU library database. \n"
        f"11. Please end your response with a reference URL from the source of the response content if available in the context. \n"
        f"12. CRITICAL: Today is {today}. If a user asks 'Is the library open today?' or asks about 'today's hours', figure out what day of the week it is based on {today}, and ONLY read the hours for that specific day of the week from the provided schedule context. \n"
        f"13. When users ask about broad research topics, recommend ASU Library OneSearch as a starting point. \n\n"
        "Context:\n"
        "{context}" 
    )

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        system_prompt=system_prompt,
        verbose=False,
        vector_store_query_mode="hybrid", 
        sparse_top_k=2 
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
    bot = setup_hybrid_bot(API_KEYS[current_key_idx])

    print("\n--- Generating Answers from your Bot ---")
    
    for q in tqdm(questions, desc="Querying ASU Bot", unit="question"):
        time.sleep(3)  
        
        success = False
        while not success:
            try:
                response = bot.chat(q)
                answers.append(response.response)
                
                retrieved_texts = [node.node.text for node in response.source_nodes]
                contexts.append(retrieved_texts)
                
                bot.reset() 
                success = True 
                
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "timeout" in error_msg:
                    current_key_idx = (current_key_idx + 1) % NUM_KEYS
                    tqdm.write(f"\n⚠️ Generation rate limit hit! Switching to API Key {current_key_idx + 1}...")
                    bot = setup_hybrid_bot(API_KEYS[current_key_idx])
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
        retry_count = 0 
        
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
                    retry_count += 1
                    if retry_count >= 3:
                        tqdm.write(f"\n⚠️ Skipping question {i+1} due to unresolvable LLM formatting error.")
                        blank_df = pd.DataFrame([{
                            "question": questions[i],
                            "answer": answers[i],
                            "contexts": contexts[i],
                            "ground_truth": ground_truths[i],
                            "faithfulness": None,
                            "answer_relevancy": None,
                            "context_precision": None,
                            "context_recall": None
                        }])
                        all_evaluation_results.append(blank_df)
                        success = True 
                    else:
                        tqdm.write(f"\n⚠️ Judge formatting error. Retrying... ({retry_count}/3)")
                        time.sleep(2)

    print("\n=== HYBRID SEARCH BENCHMARK RESULTS ===")
    
    final_results_df = pd.concat(all_evaluation_results, ignore_index=True)
    avg_scores = final_results_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean()
    print(avg_scores.to_string())
    
    output_file = "hybrid_70b_results.csv"
    final_results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to '{output_file}'")

if __name__ == "__main__":
    run_evaluation()