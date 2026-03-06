import os
import time, datetime
import streamlit as st
import toml
import chromadb
from sqlalchemy.sql import text
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_feedback import streamlit_feedback

# --- NEW IMPORTS ---
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer


cbconfig = toml.load("cbconfig.toml")
AVATARS = cbconfig['AVATARS']
ROLES = cbconfig['ROLES']

# Updated styling block to include ASU Maroon and Gold
ASU_STYLE = """
<style>
.stApp [data-testid="stHeader"] {
    display:none;
}

p img{
    margin-bottom: 0.6rem;
}

[data-testid="stSidebarCollapseButton"] {
    display:none;
}

[data-testid="baseButton-headerNoPadding"] {
    display:none;
}

.stChatInput button{
    display:none;
}

/* --- ASU THEME COLORS --- */
/* ASU Maroon: #8C1D40 | ASU Gold: #FFC627 */

/* Style action buttons */
.stButton > button {
    background-color: #8C1D40 !important;
    color: #FFFFFF !important;
    border: 2px solid #8C1D40 !important;
    border-radius: 8px;
    font-weight: bold;
}

.stButton > button:hover {
    border: 2px solid #FFC627 !important; /* Gold border on hover */
    color: #FFC627 !important; /* Gold text on hover */
    background-color: #7a1836 !important; /* Slightly darker maroon */
}

/* Style link buttons (like in the sidebar) */
a[data-testid="baseLinkButton"] {
    background-color: #FFC627 !important; /* Gold background */
    color: #000000 !important; /* Black text for contrast */
    border-radius: 8px;
    font-weight: bold;
    text-decoration: none;
}

a[data-testid="baseLinkButton"]:hover {
    background-color: #e5b223 !important;
}

/* Highlight the chat input box with ASU Gold when typing */
.stChatInputContainer:focus-within {
    border-color: #FFC627 !important;
    box-shadow: 0 0 0 1px #FFC627 !important;
}
</style>
"""

# --- 1. CACHE THE HEAVY MODELS ---
@st.cache_resource(show_spinner=False)
def get_ai_models():
    """Loads the AI models into memory only ONCE."""
    groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    llm = Groq(model="openai/gpt-oss-120b", api_key=groq_key)
    
        # embed_model = HuggingFaceEmbedding(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     device="cpu"
    # )
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu"
    )
    return llm, embed_model

# --- 2. CACHE THE DATABASE AND LINK EMBEDDING MODEL ---
@st.cache_resource(ttl="1d", show_spinner=False)
def getIndex(_embed_model): 
    client = chromadb.PersistentClient(path='./llamachromadb')
    
    # Use get_or_create to be safe
    collection = client.get_or_create_collection(name="asulib") 
    
    # We add a check here to see if the data actually exists
    count = collection.count()
    st.sidebar.write(f"📊 Database Count: {count} documents") # Helpful visual check
    
    # Initialize the Vector Store
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    # Create the index from the existing vector store
    # We use a StorageContext to ensure LlamaIndex maps the fields correctly
    from llama_index.core import StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=_embed_model
    )
    return index

def getBot(memory):    
    # Fetch models instantly from cache
    my_llm, my_embed_model = get_ai_models()
    
    # Pass the embedding model into the index generator
    index = getIndex(my_embed_model)       
    
    today = datetime.date.today().strftime('%A, %B %d, %Y')
    
    # NOTE: Your prompt currently mentions SJSU. You might want to update this to ASU!
    system_prompt = (
        f"You are ASU Sun bot, the AI assistant for the Arizona State University (ASU) Library. Respond supportively and professionally like a peer mentor. \n\n"
        f"Guidelines: \n"
        f"1. No creative content (stories, poems, tweets, code). \n"
        f"2. Simple jokes are allowed, but avoid jokes that could hurt any group. \n"
        f"3. Use up to two emojis when applicable. \n"
        f"4. Provide relevant search terms if asked. \n"
        f"5. Avoid providing information about celebrities, influential politicians, or state heads. \n"
        f"6. Keep responses detailed as possible\n"
        f"7. For unanswerable research questions, include the 'Ask A Librarian' URL: https://askalibrarian.asu.edu/ \n"
        f"8. Do not make assumptions or fabricate answers or URLs. \n"
        f"9. Use ONLY the retrieved context. If the database is insufficient, say you don't know and refer users to Ask a Librarian. \n"
        f"10. Do not provide specific book recommendations; instead, direct the user to search an ASU library database. \n"
        f"11. Please end your response with a reference URL from the source of the response content if available in the context. \n"
        f"12. CRITICAL: Today is {today}. If a user asks 'Is the library open today?' or asks about 'today's hours', figure out what day of the week it is based on {today}, and ONLY read the hours for that specific day of the week from the provided schedule context. \n"
        f"13. When users ask about broad research topics, recommend ASU Library OneSearch as a starting point. \n\n"
        "Context:\n"
        "{context}" # <--- CRITICAL: NO 'f' here! LlamaIndex needs this exactly as is to inject documents.
    )
    
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        llm=my_llm, # <-- EXPLICITLY set the LLM here!
        system_prompt=system_prompt,
        verbose=False,    
    )
    
    return chat_engine


def queryBot(user_query,bot,chip=''):        
    current = datetime.datetime.now()
    st.session_state.moment = current.isoformat()
    session_id = st.session_state.session_id
    today = current.date()
    now = current.time()
    answer = ''
    
    st.chat_message("user", avatar=AVATARS["user"]).write(user_query)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):  
        with st.spinner(text="In progress..."):
            response = bot.chat(user_query)
            answer = response.response
            st.write(answer)
            
            # --- DEBUGGER ---
            if not response.source_nodes:
                st.warning("⚠️ Debug: I couldn't find any documents in the database to answer this! Ensure your 'asulib' collection has data.")

if __name__ == "__main__":    

    # set up streamlit page
    st.set_page_config(page_title="ASU Library Assistant", page_icon="🔱", initial_sidebar_state="expanded")
    st.markdown(ASU_STYLE, unsafe_allow_html=True)
    
    # side
    st.sidebar.markdown(cbconfig['side']['title'])
    st.sidebar.markdown(cbconfig['side']['intro'])
    st.sidebar.markdown("\n\n")
    st.sidebar.link_button(cbconfig['side']['policylabel'],cbconfig['side']['policylink'])
    
    # main
    col1, col2, col3 = st.columns([0.25,0.1,0.65],vertical_alignment="bottom")
    with col2:
        st.markdown(cbconfig['main']['logo'])
    with col3:
        st.title(cbconfig['main']['title'])
    st.markdown("\n\n")
    st.markdown("\n\n")
  
    col21, col22, col23 = st.columns(3)
    with col21:
        button1 = st.button(cbconfig['button1']['label'])
    with col22:
        button2 = st.button(cbconfig['button2']['label'])    
    with col23:
        button3 = st.button(cbconfig['button3']['label'])    
    
    # lastest 5 messeges kept in memory for bot prompt
    if 'memory' not in st.session_state: 
        memory = ChatMemoryBuffer.from_defaults(token_limit=5000)
        st.session_state.memory = memory  
    memory = st.session_state.memory
    
    # get bot
    if 'mybot' not in st.session_state: 
        st.session_state.mybot = getBot(memory)  
    bot = st.session_state.mybot

    # get streamlit session 
    if 'session_id' not in st.session_state:
        session_id = get_script_run_ctx().session_id
        st.session_state.session_id = session_id
        
    if 'reference' not in st.session_state:
        st.session_state.reference = ''

    # messeges kept in streamlit session for display
    max_messages: int = 10  # Set the limit (K) of messages to keep
    allmsgs = memory.get()
    msgs = allmsgs[-max_messages:]
                      
    # display chat history
    for msg in msgs:
        st.chat_message(ROLES[msg.role],avatar=AVATARS[msg.role]).write(msg.content)

    # chip 
    if button1:
        queryBot(cbconfig['button1']['content'],bot,cbconfig['button1']['chip'])
    if button2:
        queryBot(cbconfig['button2']['content'],bot,cbconfig['button2']['chip'])
    if button3:
        queryBot(cbconfig['button3']['content'],bot,cbconfig['button3']['chip'])
            
    # chat (Updated placeholder)
    if user_query := st.chat_input(placeholder="Ask me about the ASU Library!"):
        queryBot(user_query,bot)
        
    # feedback, works outside user_query section     
    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "Optional. Please provide extra information",
    }
                        
    if 'moment' in st.session_state:
        currents = st.session_state.moment
        streamlit_feedback(
            **feedback_kwargs, args=(currents,), key=currents,
        )