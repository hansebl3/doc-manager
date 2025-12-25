import streamlit as st
import subprocess
from db_manager import DBManager
from llm_client import LLMClient
from sentence_transformers import SentenceTransformer

# Import Tabs
from ui.tab_upload import render_upload_tab
from ui.tab_batch import render_batch_tab
from ui.tab_review import render_review_tab
from ui.tab_search import render_search_tab

# Page Config
st.set_page_config(page_title="Documentation Manager", layout="wide")

# Initialize Session State
if "db" not in st.session_state:
    st.session_state.db = DBManager()
if "llm" not in st.session_state:
    st.session_state.llm = LLMClient()
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
if "categories" not in st.session_state:
    st.session_state.categories = ["General", "Personal", "CTC", "Proposal"]

# Sidebar - Settings & Prompt History
st.sidebar.title("Settings")
llm_url = st.sidebar.text_input("LLM Base URL", value="http://192.168.1.238:8080/v1")
st.session_state.llm.base_url = llm_url

st.sidebar.divider()
st.sidebar.subheader("Recent Prompts")
history = st.session_state.llm.get_history()
for h in history:
    if st.sidebar.button(h[:30] + "...", key=h):
        st.session_state.custom_prompt = h

# Tabs
tab_upload, tab_process, tab_review, tab_search = st.tabs(["1. Upload", "2. Batch Processing", "3. Review & Save", "4. Search & View"])

# --- Tab 1: Upload & Check ---
with tab_upload:
    render_upload_tab()

# --- Tab 2: Batch Processing ---
with tab_process:
    render_batch_tab()

# --- Tab 3: Review & Save ---
with tab_review:
    render_review_tab()

# --- Tab 4: Search & View ---
with tab_search:
    render_search_tab()
