# AsuSunBot# 🔱 ASU Library Assistant (Kingbot)

An AI-powered, Retrieval-Augmented Generation (RAG) chatbot designed to act as a peer mentor for Arizona State University (ASU) library users. Built with Streamlit, LlamaIndex, Groq, and ChromaDB, this assistant answers questions about library hours, FAQs, databases, and research guides using verified local data.

## ✨ Features
* **Conversational AI:** Powered by the blazing-fast Groq API (`openai/gpt-oss-120b`).
* **Local Vector Database:** Uses ChromaDB and HuggingFace local embeddings (`BAAI/bge-small-en-v1.5`) for secure, fast document retrieval.
* **ASU Theming:** Custom Streamlit UI styled with ASU Maroon and Gold.
* **Context-Aware:** Reads the current day of the week to provide accurate, real-time library hours.

## 📋 Requirements
* **Python:** 3.9 or higher
* **API Key:** A valid [Groq API Key](https://console.groq.com/keys)
* **Dependencies:** Listed in `requirements.txt`

### Key Packages
* `streamlit`
* `llama-index`
* `llama-index-llms-groq`
* `llama-index-embeddings-huggingface`
* `llama-index-vector-stores-chroma`
* `chromadb`
* `pandas`

## 🚀 Setup Instructions

**1. Install Dependencies**
Open your terminal, navigate to the project directory, and install the required Python packages:

```bash
pip install -r requirements.txt
```

**2. Configure Your API Key**
The application requires a Groq API key to run the language model. 
* Create a folder named `.streamlit` in the root directory.
* Inside it, create a file named `secrets.toml`.
* Add your key to the file exactly like this:

```toml
GROQ_API_KEY = "gsk_your_actual_api_key_here"
```

**3. Build the Vector Database**
Before running the chatbot, you must process the library data (CSVs) into the ChromaDB vector format. Run the indexing script:

```bash
python build_asu_index.py
```
*Note: This will read the files in your data folder and generate a local database in the `/llamachromadb` directory.*

## 💻 Running the Application

Once the database is built and your API key is configured, you can launch the Streamlit interface:

```bash
python -m streamlit run llamainchatbot.py
```

The app will open automatically in your default web browser (usually at `http://localhost:8501`).

## 📁 Project Structure

```text
AsuLibBot/
│
├── llamainchatbot.py       # Main Streamlit application
├── build_asu_index.py      # Script to build the ChromaDB vector store
├── cbconfig.toml           # UI configuration and text assets
├── requirements.txt        # Python dependencies
│
├── .streamlit/             
│   └── secrets.toml        # Hidden API keys (Ignored by Git)
│
├── data/                   # Source knowledge base (CSVs)
│   ├── asu_library_faqs.csv
│   ├── databases.csv
│   ├── consolidated_guides.csv
│   └── library_timetable_updated.csv
│
└── llamachromadb/          # Generated vector database 
```