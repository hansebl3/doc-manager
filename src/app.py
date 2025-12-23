import streamlit as st
import pandas as pd
import json
import os
from db_manager import DBManager
from llm_client import LLMClient
from utils.md_processor import MDProcessor
from sentence_transformers import SentenceTransformer

# Page Config
st.set_page_config(page_title="Documentation Manager", layout="wide")

# Initialize Session State
if "db" not in st.session_state:
    st.session_state.db = DBManager()
if "llm" not in st.session_state:
    st.session_state.llm = LLMClient()
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
if "doc_queue" not in st.session_state:
    st.session_state.doc_queue = {}  # {uuid: {content, filename, metadata, status}}
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = {} # {uuid: {meta_l, sum_l, meta_r, sum_r, ...}}
if "processing_active" not in st.session_state:
    st.session_state.processing_active = False

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
# --- Tab 1: Upload & Check ---
with tab_upload:
    st.header("Upload or Input")
    
    # 1. File Upload
    st.subheader("1. File Upload")
    uploaded_files = st.file_uploader("Drag and drop MD files", type=["md"], accept_multiple_files=True)
    
    valid_docs = []
    
    # 2. Manual Input
    st.subheader("2. Manual Text Input")
    manual_text = st.text_area("Enter text content directly", height=150)
    if st.button("Add Text to Processing Queue"):
        if manual_text.strip():
            # Generate ID and fake filename
            m_uuid = MDProcessor.generate_uuid_v7()
            m_filename = f"manual_input_{m_uuid[:8]}.md"
            
            # Upsert L0
            parent_emb = st.session_state.embedder.encode(manual_text).tolist()
            st.session_state.db.upsert_document(
                m_uuid, 
                "L0", 
                MDProcessor.prepare_metadata(manual_text), 
                manual_text, 
                parent_emb
            )
            
            # Add to Queue
            st.session_state.db.enqueue_task(m_uuid, config={"filename": m_filename})
            
            st.success(f"Added manual text ({m_filename}) to DB Queue!")
            # Retain text? Streamlit refreshes on button press usually clearing it unless we use session state for the widget. 
            # For now, clearing it (default behavior) is fine or we can let it reset.
        else:
            st.warning("Please enter some text.")

    if uploaded_files:
        st.divider()
        st.subheader("Detected Files")
        
        for u_file in uploaded_files:
            content = u_file.read().decode("utf-8")
            u_file.seek(0) # Reset pointer
            doc_uuid, clean_content = MDProcessor.extract_uuid(content, u_file.name)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**File:** {u_file.name}")
                if doc_uuid:
                    st.caption(f"UUID: {doc_uuid}")
                else:
                    st.warning("No UUID found - Will Generate New")
            
            with col2:
                # Check DB
                if doc_uuid:
                    existing = st.session_state.db.get_document(doc_uuid)
                    if existing:
                        st.warning("Already in DB")
                    else:
                        st.success("New Document")
            
            # If no UUID, generate one now
            if not doc_uuid:
                doc_uuid = MDProcessor.generate_uuid_v7()
                
            if doc_uuid:
                valid_docs.append({
                    "id": doc_uuid,
                    "filename": u_file.name,
                    "content": clean_content,
                    "metadata": MDProcessor.prepare_metadata(content)
                })
        
    if valid_docs:
        if st.button(f"Add {len(valid_docs)} Documents to DB Processing Queue"):
            count = 0
            for doc in valid_docs:
                # Add to DB Processing Queue
                st.session_state.db.enqueue_task(doc['id'], config={"filename": doc['filename']}) # Save filename in config for display
                
                # Upsert partial L0
                parent_emb = st.session_state.embedder.encode(doc['content']).tolist()
                st.session_state.db.upsert_document(doc['id'], "L0", doc['metadata'], doc['content'], parent_emb)
                
                count += 1
            st.success(f"Added {count} documents to DB Queue. Go to 'Batch Processing'.")
            count = 0
            for doc in valid_docs:
                # Add to DB Processing Queue
                st.session_state.db.enqueue_task(doc['id'], config={"filename": doc['filename']}) # Save filename in config for display
                
                # Upsert partial L0
                parent_emb = st.session_state.embedder.encode(doc['content']).tolist()
                st.session_state.db.upsert_document(doc['id'], "L0", doc['metadata'], doc['content'], parent_emb)
                
                count += 1
            st.success(f"Added {count} documents to DB Queue. Go to 'Batch Processing'.")

    # Show Queue status from DB
    st.divider()
    st.subheader("DB Processing Queue Status")
    
    # We can fetch count by status
    tasks_created = st.session_state.db.get_tasks_by_status('created')
    tasks_queued = st.session_state.db.get_tasks_by_status('queued')
    tasks_processing_l = st.session_state.db.get_tasks_by_status('processing_l')
    tasks_processing_r = st.session_state.db.get_tasks_by_status('processing_r')
    tasks_done = st.session_state.db.get_tasks_by_status('done')
    
    cols = st.columns(5)
    cols[0].metric("Created", len(tasks_created))
    cols[1].metric("Queued", len(tasks_queued))
    cols[2].metric("Processing L", len(tasks_processing_l))
    cols[3].metric("Processing R", len(tasks_processing_r))
    cols[4].metric("Done (Wait Review)", len(tasks_done))

# --- Tab 2: Batch Processing ---
with tab_process:
    st.header("Batch LLM Processing")
    
    # 1. Fetch 'created' tasks that need configuration
    tasks_to_config = st.session_state.db.get_tasks_by_status('created')
    
    if not tasks_to_config:
        st.info("No new tasks to configure. Upload more files or check Queue status.")
    else:
        st.write(f"Found {len(tasks_to_config)} documents waiting for configuration.")
        
        col_sets, col_ctrl = st.columns([2, 1])
        
        with col_sets:
            prompt_summary = st.text_area("Summary Prompt", 
                                       value=st.session_state.get("prompt_summary", "Summarize the following document concisely."),
                                       height=100)
            prompt_meta = st.text_area("Metadata Prompt", 
                                       value=st.session_state.get("prompt_meta", "Extract the document date and key technical keywords."),
                                       height=100)
        
        with col_ctrl:
            models = st.session_state.llm.get_available_models()
            
            # Helper to load/save prefs
            PREFS_FILE = "user_prefs.json"
            def load_prefs():
                if os.path.exists(PREFS_FILE):
                    try:
                         with open(PREFS_FILE, 'r') as f: return json.load(f)
                    except: return {}
                return {}
            
            def save_prefs(key, value):
                p = load_prefs()
                p[key] = value
                with open(PREFS_FILE, 'w') as f: json.dump(p, f)
            
            prefs = load_prefs()
            
            # Determine default indices based on prefs
            def get_index(model_list, pref_key, default_idx):
                if pref_key in prefs and prefs[pref_key] in model_list:
                    return model_list.index(prefs[pref_key])
                return default_idx

            idx_l = get_index(models, "model_l", 0)
            idx_r = get_index(models, "model_r", min(1, len(models)-1))

            model_l = st.selectbox("Left Model", models, index=idx_l, key="batch_model_l")
            model_r = st.selectbox("Right Model", models, index=idx_r, key="batch_model_r")
            
            # Save on change (using callback or just checking state vs prefs? Callback is cleaner but Streamlit reruns script on change)
            if model_l != prefs.get("model_l"): save_prefs("model_l", model_l)
            if model_r != prefs.get("model_r"): save_prefs("model_r", model_r)
            
            if st.button("Start Batch Execution", type="primary"):
                st.session_state.llm._save_history(prompt_summary)
                st.session_state.llm._save_history(prompt_meta)
                
                # Update all 'created' tasks to 'queued' with config
                count = 0
                for task in tasks_to_config:
                    new_config = task['config'] or {}
                    new_config.update({
                        "model_l": model_l,
                        "model_r": model_r,
                        "prompt_summary": prompt_summary,
                        "prompt_meta": prompt_meta
                    })
                    st.session_state.db.update_task(task['doc_id'], status='queued', config=new_config)
                    count += 1
                
                st.success(f"Queued {count} tasks! The background worker will pick them up.")
                st.rerun()

    # Monitoring Section
    st.divider()
    st.subheader("Live Monitor")
    
    if st.button("Refresh Status"):
        st.rerun()
        
    tasks_active = st.session_state.db.get_tasks_by_status('processing_l') + st.session_state.db.get_tasks_by_status('processing_r')
    if tasks_active:
        st.warning(f"Currently Processing: {len(tasks_active)} docs")
        for t in tasks_active:
             fname = t['config'].get('filename', str(t['doc_id']))
             st.text(f"Processing: {fname} [{t['status']}]")
    else:
        st.info("Worker is idle or all tasks done.")

# --- Tab 3: Review & Save ---
with tab_review:
    st.header("Review & Confirm")
    
    # Fetch done tasks from DB
    done_tasks = st.session_state.db.get_tasks_by_status('done')
    
    if not done_tasks:
        st.info("No documents waiting for review.")
    else:
        # DEBUG
        st.write(f"DEBUG: Found {len(done_tasks)} completed tasks")
        st.json([{"doc_id": str(t['doc_id']), "status": t['status'], "has_results": bool(t['results'])} for t in done_tasks])
        
        # Document Selection for Review
        # Map doc_id to display name (filename from config)
        
        # Helper to get name
        def get_name(t):
           return t['config'].get('filename', str(t['doc_id']))
        
        selected_task_id = st.selectbox("Select Pending Document", [t['doc_id'] for t in done_tasks], 
                                     format_func=lambda x: get_name(next(item for item in done_tasks if item["doc_id"] == x)))
        
        current_task = next(t for t in done_tasks if t['doc_id'] == selected_task_id)
        res = current_task['results']
        
        st.info(f"Reviewing: {get_name(current_task)}")
        
        # 1. Keywords Cross-Selection
        st.subheader("1. Keywords & Metadata")
        kw_l = res['meta_l'].get("keywords", [])
        kw_r = res['meta_r'].get("keywords", [])
        all_keywords = sorted(list(set(kw_l + kw_r)))
        
        col_k1, col_k2 = st.columns([3, 1])
        with col_k1:
            selected_keywords = st.multiselect("Select Keywords", all_keywords, default=all_keywords)
        with col_k2:
            final_date = st.text_input("Date", value=res['meta_l'].get("date", res['meta_r'].get("date", "")))

        st.divider()
        
        # 2. Summary Selection
        st.subheader("2. Summary Selection")
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.markdown(f"**Left Model ({current_task['config'].get('model_l')})**")
            st.text_area("Left Output", value=res['sum_l'], height=300, disabled=True, key=f"disp_l_{selected_task_id}")
            
        with col_r:
            st.markdown(f"**Right Model ({current_task['config'].get('model_r')})**")
            st.text_area("Right Output", value=res['sum_r'], height=300, disabled=True, key=f"disp_r_{selected_task_id}")

        # Choice
        choice = st.radio("Choose Base Summary", ["Left Model", "Right Model"], horizontal=True)
        base_text = res['sum_l'] if choice == "Left Model" else res['sum_r']
        
        st.divider()
        
        # 3. Final Edit
        st.subheader("3. Final Edit & Save")
        
        # Use a key that changes with choice so base_text updates correctly
        final_summary_text = st.text_area("Final Summary Content", value=base_text, height=300, key=f"final_{selected_task_id}_{choice}")
        
        if st.button("Confirm & Save to DB", type="primary"):
             with st.spinner("Saving..."):
                doc_id = str(current_task['doc_id']) # Ensure string
                summary_id = MDProcessor.generate_uuid_v7()
                
                # Fetch original to keep other metadata? 
                # We already have valid metadata in L0 doc from Tab 1 upload.
                existing_doc = st.session_state.db.get_document(doc_id)
                original_meta = existing_doc['metadata'] if existing_doc else {}
                
                # Metadata for L0 update
                final_meta = {
                    "keywords": selected_keywords,
                    "date": final_date,
                    "original_meta": original_meta # Keep existing meta structure
                }
                
                # Delete old summaries if exist (Re-summarization flow)
                # Requirement: "Delete existing summary... and delete summary uuid from document"
                if existing_doc and existing_doc.get('summary_uuids'):
                     for old_sum_id in existing_doc['summary_uuids']:
                        # 1. Delete the summary document
                        st.session_state.db.delete_document(old_sum_id)
                        # 2. Remove the link from the parent (L0)
                        st.session_state.db.remove_summary_link(doc_id, old_sum_id)
                
                # Upsert L0 (Update metadata)
                # Content is already in DB from Tab 1 upload
                st.session_state.db.upsert_document(doc_id, "L0", final_meta, existing_doc['content'])
                
                # Save Summary (L1)
                summary_emb = st.session_state.embedder.encode(final_summary_text).tolist()
                st.session_state.db.upsert_document(summary_id, "L1", {}, final_summary_text, summary_emb)
                
                # Link (Mutual)
                st.session_state.db.link_documents(doc_id, summary_id)
                
                # Delete Task from Queue
                st.session_state.db.delete_task(doc_id)
                
                st.success(f"Saved L1 ({summary_id}). Task removed from queue.")
                st.rerun()

# --- Tab 4: Search & View ---
with tab_search:
    st.header("Search Knowledge Base")
    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        search_query = st.text_input("Text Search")
    with col_s2:
        search_cat = st.selectbox("Category", ["ALL", "L0", "L1", "L2", "L3"])
    with col_s3:
        search_uuid = st.text_input("UUID Search")
    
    cat_filter = None if search_cat == "ALL" else search_cat
    uuid_filter = search_uuid if search_uuid else None
    
    results = st.session_state.db.search_documents(query_text=search_query, category=cat_filter, doc_id=uuid_filter)
    
    if results:
        df = pd.DataFrame(results)
        # Drop large columns for display
        if 'embedding' in df.columns:
            display_df = df.drop(columns=['embedding'])
        else:
            display_df = df
            
        st.dataframe(display_df, use_container_width=True)
        
        for idx, row in df.iterrows():
            # Indentation for L1
            if row['category'] == 'L1':
                c_indent, c_content = st.columns([1, 15])
                container = c_content
            else:
                container = st
            
            with container:
                with st.expander(f"[{row['category']}] {row['id']}"):
                    st.write(f"**Metadata:** {json.dumps(row['metadata'])}")
                    st.write(f"**Content Snippet:** {row['content'][:500]}...")
                    
                    # RE-SUMMARIZE (Only for L0)
                    if row['category'] == 'L0':
                        if st.button(f"Re-summarize {row['id']}", key=f"resum_{row['id']}"):
                            st.session_state.db.enqueue_task(row['id'], config={"filename": f"Re-Sum: {row['id']}"})
                            st.success(f"Added {row['id']} to Batch Processing Queue. Go to Tab 2.")
                            
                    # EDIT CONTENT (Only for L0)
                    if row['category'] == 'L0':
                        with st.expander(f"Edit Content"):
                            new_content = st.text_area("Update Content", value=row['content'], height=200, key=f"edit_{row['id']}")
                            if st.button("Save & Reset Summaries", key=f"save_{row['id']}"):
                                new_emb = st.session_state.embedder.encode(new_content).tolist()
                                st.session_state.db.upsert_document(
                                    row['id'], "L0", row['metadata'], new_content, new_emb
                                )
                                if row.get('summary_uuids'):
                                    for sum_id in row['summary_uuids']:
                                        st.session_state.db.delete_document(sum_id)
                                    st.session_state.db.clear_summary_links(row['id'])
                                st.success("Content updated and old summaries removed.")
                                st.rerun()

                    if row['category'] == 'L1':
                        parent_uuid = row['metadata'].get('parent_id') or row.get('metadata', {}).get('original_meta', {}).get('parent_id')
                        if parent_uuid and st.button(f"Show Parent (L0): {parent_uuid}", key=f"btn_{row['id']}"):
                            parent = st.session_state.db.get_document(parent_uuid)
                            if parent:
                                st.info(f"**Parent Content:**\n{parent['content']}")
                    
                    if row['summary_uuids'] and len(row['summary_uuids']) > 0:
                        st.write("**Related Summaries (L1):**")
                        for s_id in row['summary_uuids']:
                            s_doc = st.session_state.db.get_document(s_id)
                            if s_doc:
                                st.write(s_doc['content'])
                    
                    st.divider()
                    
                    # DELETE with Confirmation
                    del_key = f"confirm_del_{row['id']}"
                    if st.session_state.get(del_key, False):
                        st.warning("Are you sure you want to delete this document?")
                        col_conf, col_can = st.columns(2)
                        with col_conf:
                            if st.button("Yes, DELETE", key=f"yes_{row['id']}", type="primary"):
                                # Cascading Delete Logic
                                if row['category'] == 'L0' and row.get('summary_uuids'):
                                    st.warning(f"Deleting L0 {row['id']} and {len(row['summary_uuids'])} linked summaries.")
                                    for sum_id in row['summary_uuids']:
                                        st.session_state.db.delete_document(sum_id)
                                
                                st.session_state.db.delete_document(row['id'])
                                
                                if row['category'] == 'L1':
                                     if row.get('source_uuids'):
                                         for src in row['source_uuids']:
                                             st.session_state.db.remove_summary_link(src, row['id'])
                                
                                st.toast(f"Deleted {row['id']}")
                                st.session_state[del_key] = False # Reset
                                st.rerun()
                                
                        with col_can:
                            if st.button("Cancel", key=f"no_{row['id']}"):
                                st.session_state[del_key] = False
                                st.rerun()
                    else:
                        if st.button(f"Request Delete {row['id']}", key=f"req_del_{row['id']}"):
                            st.session_state[del_key] = True
                            st.rerun()
    else:
        st.info("No documents found.")

