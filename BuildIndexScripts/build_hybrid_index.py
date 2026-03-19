import os
import pandas as pd
import chromadb

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore

# ====== CONFIG ======
# We are making a dedicated folder for the hybrid index because we need to save JSON files alongside Chroma
PERSIST_DIR = r"C:\Stuff\AsuSunBot\llamachromadb_hybrid" 
COLLECTION_NAME = "asulib_hybrid" 

FAQS_CSV = r"C:\Stuff\AsuSunBot\data\asu_library_faqs.csv"
DATABASES_CSV = r"C:\Stuff\AsuSunBot\data\databases.csv"
GUIDES_CSV = r"C:\Stuff\AsuSunBot\data\consolidated_guides.csv"
TIMETABLE_CSV = r"C:\Stuff\AsuSunBot\data\library_timetable_updated.csv"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Document Loaders ---
def load_faq_docs(path: str) -> list[Document]:
    if not os.path.exists(path): return []
    df = pd.read_csv(path).dropna(subset=['Question', 'Answer Text'])
    docs = []
    for _, row in df.iterrows():
        text = f"FAQ Question: {row['Question']}\nAnswer: {row['Answer Text']}"
        docs.append(Document(text=text))
    return docs

def load_database_docs(path: str) -> list[Document]:
    if not os.path.exists(path): return []
    df = pd.read_csv(path).dropna(subset=['Database Name', 'Description', 'URL'])
    docs = []
    for _, row in df.iterrows():
        text = f"Database: {row['Database Name']}\nDescription: {row['Description']}\nLink: {row['URL']}"
        docs.append(Document(text=text))
    return docs

def load_guide_docs(path: str) -> list[Document]:
    if not os.path.exists(path): return []
    df = pd.read_csv(path).dropna(subset=['Guide Name', 'Guide URL'])
    docs = []
    for _, row in df.iterrows():
        text = f"Research Guide: {row['Guide Name']}\nLink: {row['Guide URL']}"
        docs.append(Document(text=text))
    return docs

def load_timetable_docs(path: str) -> list[Document]:
    if not os.path.exists(path): return []
    df = pd.read_csv(path).dropna(subset=['Location'])
    docs = []
    for _, row in df.iterrows():
        text = (f"Location: {row['Location']}\n"
                f"Sunday: {row.get('Sunday', 'Closed')}\n"
                f"Monday: {row.get('Monday', 'Closed')}\n"
                f"Tuesday: {row.get('Tuesday', 'Closed')}\n"
                f"Wednesday: {row.get('Wednesday', 'Closed')}\n"
                f"Thursday: {row.get('Thursday', 'Closed')}\n"
                f"Friday: {row.get('Friday', 'Closed')}\n"
                f"Saturday: {row.get('Saturday', 'Closed')}")
        docs.append(Document(text=text))
    return docs

def build_hybrid_index():
    print("Step 1: Loading documents from CSVs...")
    docs = (load_faq_docs(FAQS_CSV) + 
            load_database_docs(DATABASES_CSV) + 
            load_guide_docs(GUIDES_CSV) + 
            load_timetable_docs(TIMETABLE_CSV))
    print(f"Total documents prepared: {len(docs)}")

    print(f"Step 2: Loading Embedding Model: {EMBED_MODEL_NAME}...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device="cpu")

    print(f"Step 3: Setting up ChromaDB collection '{COLLECTION_NAME}'...")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except:
        pass

    chroma_collection = chroma_client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    print("Step 4: Setting up Document Store for Keyword Search (BM25)...")
    # NEW: We must save the raw text locally so BM25 can read it!
    docstore = SimpleDocumentStore()
    docstore.add_documents(docs)

    print("Step 5: Building the Hybrid Index...")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore
    )
    
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    print("Step 6: Saving to disk...")
    # NEW: This saves a `docstore.json` file inside your PERSIST_DIR
    storage_context.persist(persist_dir=PERSIST_DIR)

    print(f"\n✅ Success! Hybrid Index saved to {PERSIST_DIR} under '{COLLECTION_NAME}'")

if __name__ == "__main__":
    build_hybrid_index()