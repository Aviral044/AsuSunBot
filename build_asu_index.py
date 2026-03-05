import os
import pandas as pd
import chromadb

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext

# ====== CONFIG ======
PERSIST_DIR = r"C:\Stuff\asu-lib-bot\AsuLibBot\llamachromadb"
COLLECTION_NAME = "asulib"

# Ensure these match your actual data folder location
FAQS_CSV = r"C:\Stuff\asu-lib-bot\AsuLibBot\data\asu_library_faqs.csv"
DATABASES_CSV = r"C:\Stuff\asu-lib-bot\AsuLibBot\data\databases.csv"
GUIDES_CSV = r"C:\Stuff\asu-lib-bot\AsuLibBot\data\consolidated_guides.csv"
# --- NEW ---
TIMETABLE_CSV = r"C:\Stuff\asu-lib-bot\AsuLibBot\data\library_timetable_updated.csv" 

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def load_faq_docs(path: str) -> list[Document]:
    if not os.path.exists(path): return []
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        q = str(row.get("Question", "")).strip()
        a = str(row.get("Answer Text", "")).strip()
        url = str(row.get("Final URL", "")).strip()
        topics = str(row.get("Topics", "")).strip()
        if not q and not a: continue
        text = f"FAQ Question: {q}\n\nFAQ Answer:\n{a}"
        meta = {"type": "faq", "url": url, "topics": topics}
        docs.append(Document(text=text, metadata=meta))
    return docs

def load_database_docs(path: str) -> list[Document]:
    if not os.path.exists(path): return []
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        name = str(row.get("Database Name", "")).strip()
        desc = str(row.get("Description", "")).strip()
        url = str(row.get("URL", "")).strip()
        if not name and not desc: continue
        text = f"Database: {name}\n\nDescription:\n{desc}"
        meta = {"type": "database", "url": url}
        docs.append(Document(text=text, metadata=meta))
    return docs

def load_guide_docs(path: str) -> list[Document]:
    if not os.path.exists(path): return []
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        title = str(row.get("Guide Name", "")).strip()
        url = str(row.get("URL", "")).strip()
        subjects = str(row.get("Subjects", "")).strip()
        tags = str(row.get("Tags", "")).strip()
        desc = str(row.get("Description", "")).strip()
        if not title and not url: continue
        text = f"Research Guide (metadata): {title}\nSubjects: {subjects}\nTags: {tags}\nDescription: {desc}\nURL: {url}"
        meta = {"type": "guide", "url": url, "subjects": subjects, "tags": tags}
        docs.append(Document(text=text, metadata=meta))
    return docs

# --- NEW FUNCTION ---
def load_timetable_docs(path: str) -> list[Document]:
    if not os.path.exists(path): return []
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        location = str(row.get("Location", "")).strip()
        link = str(row.get("Link", "")).strip()
        if not location: continue
        
        # Format the schedule as a clear, readable string for the LLM
        schedule = (
            f"Sunday: {row.get('Sunday', 'Closed')}\n"
            f"Monday: {row.get('Monday', 'Closed')}\n"
            f"Tuesday: {row.get('Tuesday', 'Closed')}\n"
            f"Wednesday: {row.get('Wednesday', 'Closed')}\n"
            f"Thursday: {row.get('Thursday', 'Closed')}\n"
            f"Friday: {row.get('Friday', 'Closed')}\n"
            f"Saturday: {row.get('Saturday', 'Closed')}"
        )
        
        text = f"Library Location/Service: {location}\n\nWeekly Operating Hours:\n{schedule}\n\nMore Info: {link}"
        meta = {"type": "timetable", "url": link}
        docs.append(Document(text=text, metadata=meta))
    return docs

def main():
    print("Step 1: Initializing Embedding Model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device="cpu")

    print(f"Step 2: Setting up ChromaDB at {PERSIST_DIR}...")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Cleaned up old '{COLLECTION_NAME}' collection.")
    except:
        pass

    chroma_collection = chroma_client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    print("Step 3: Loading documents from CSVs...")
    
    # --- UPDATED DOCUMENT LOADER ---
    docs = (load_faq_docs(FAQS_CSV) + 
            load_database_docs(DATABASES_CSV) + 
            load_guide_docs(GUIDES_CSV) + 
            load_timetable_docs(TIMETABLE_CSV))
            
    print(f"Total documents prepared: {len(docs)}")

    if len(docs) == 0:
        print("Error: No documents found! Check your CSV file paths.")
        return

    print("Step 4: Building the Vector Index (This will take a moment)...")
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        docs, 
        storage_context=storage_context,
        show_progress=True
    )

    print("Success: Index build complete.")
    final_count = chroma_collection.count()
    print(f"Collection '{COLLECTION_NAME}' now contains {final_count} documents.")

if __name__ == "__main__":
    main()