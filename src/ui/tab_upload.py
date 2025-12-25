import streamlit as st
from datetime import datetime
from utils.md_processor import MDProcessor
import subprocess

def render_upload_tab():
    st.header("Upload or Input")
    
    # 1. File Upload
    st.subheader("1. File Upload")
    uploaded_files = st.file_uploader("Drag and drop MD files", type=["md"], accept_multiple_files=True)
    
    valid_docs = []
    
    # 2. Manual Input
    st.subheader("2. Manual Text Input")
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        doc_date = st.date_input("Document Date", value=datetime.today(), key="manual_date")
        manual_cat = st.selectbox("Category", st.session_state.categories, key="manual_cat")
    
    manual_text = st.text_area("Enter text content directly", height=150)
    manual_title_input = st.text_input("Document Title (Optional)", placeholder="Leave blank to use first 20 chars of content")
    
    if st.button("Add Text to Processing Queue"):
        if manual_text.strip():
            # Title logic: use input or first 20 chars
            m_title = manual_title_input.strip() if manual_title_input.strip() else manual_text.strip()[:20]
            
            # Convert date to timestamp (milliseconds for UUIDv7)
            # Use local time combine and timestamp
            dt_obj = datetime.combine(doc_date, datetime.min.time())
            ts_ms = int(dt_obj.timestamp() * 1000)
            
            # Generate ID and fake filename based on picked date
            m_uuid = MDProcessor.generate_uuid_v7(timestamp=ts_ms)
            m_filename = f"manual_input_{m_uuid[:8]}.md"
            
            # Add to Queue
            manual_meta = MDProcessor.prepare_metadata(manual_text)
            manual_meta['date'] = doc_date.strftime("%Y-%m-%d")

            # Upsert L0
            parent_emb = st.session_state.embedder.encode(manual_text).tolist()
            st.session_state.db.upsert_document(
                m_uuid, 
                manual_cat,
                "L0", 
                manual_meta, 
                manual_text, 
                parent_emb,
                title=m_title
            )
            
            # Add to Queue
            st.session_state.db.enqueue_task(m_uuid, config={"filename": m_filename, "title": m_title})
            
            st.success(f"Added manual text ({m_filename}) to DB Queue!")
            # Retain text? Streamlit refreshes on button press usually clearing it unless we use session state for the widget. 
            # For now, clearing it (default behavior) is fine or we can let it reset.
        else:
            st.warning("Please enter some text.")

    if uploaded_files:
        st.divider()
        st.subheader("Detected Files")
        upload_cat = st.selectbox("Set Category for All Uploads", st.session_state.categories, key="upload_cat")
        
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
                st.session_state.db.enqueue_task(doc['id'], config={"filename": doc['filename'], "title": doc['filename']}) # Save filename in config for display
                
                # Upsert partial L0
                parent_emb = st.session_state.embedder.encode(doc['content']).tolist()
                st.session_state.db.upsert_document(doc['id'], upload_cat, "L0", doc['metadata'], doc['content'], parent_emb, title=doc['filename'])
                
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

    # Manage Active Queue
    st.divider()
    st.subheader("Manage Active Queue")
    with st.expander("Show Detailed Queue Management"):
        all_task_list = tasks_created + tasks_queued + tasks_processing_l + tasks_processing_r + tasks_done
        if not all_task_list:
            st.info("No tasks in any status.")
        else:
            if tasks_done:
                if st.button("Clear All 'Done' Tasks"):
                    for t in tasks_done:
                        st.session_state.db.delete_task(t['doc_id'])
                    st.rerun()
            
            st.divider()
            for t in all_task_list:
                fname = t.get('config', {}).get('filename', str(t['doc_id']))
                with st.expander(f"**{fname}** - `{t['status']}`"):
                    doc = st.session_state.db.get_document(t['doc_id'])
                    if doc:
                        new_content = st.text_area("Edit Content", value=doc['content'], height=200, key=f"edit_q_{t['doc_id']}")
                        
                        c1, c2, c3 = st.columns([1, 1, 1])
                        if c1.button("Save Changes", key=f"save_q_{t['doc_id']}", type="primary"):
                            # Re-embed and update
                            new_emb = st.session_state.embedder.encode(new_content).tolist()
                            st.session_state.db.upsert_document(
                                doc['id'], doc['category'], doc['level'], doc['metadata'], new_content, new_emb
                            )
                            st.success("Changes saved!")
                            st.rerun()
                        
                        if c2.button("Cancel Task", key=f"can_q_{t['doc_id']}"):
                            st.session_state.db.delete_task(t['doc_id'])
                            st.rerun()
                        
                        if c3.button("Delete Doc & Task", key=f"del_dq_{t['doc_id']}"):
                            st.session_state.db.delete_task(t['doc_id'])
                            st.session_state.db.delete_document(t['doc_id'])
                            st.rerun()
                    else:
                        st.error("Document data not found.")
                        if st.button("Purge Orphaned Task", key=f"purge_{t['doc_id']}"):
                            st.session_state.db.delete_task(t['doc_id'])
                            st.rerun()

    # Worker Status Check
    def is_worker_running():
        try:
            cmd = "ps aux | grep '[w]orker.py'"
            result = subprocess.check_output(cmd, shell=True).decode()
            return len(result.strip()) > 0
        except:
            return False

    st.sidebar.divider()
    worker_ok = is_worker_running()
    if worker_ok:
        st.sidebar.success("✅ Worker: Running")
    else:
        st.sidebar.error("❌ Worker: Stopped")
        if st.sidebar.button("Try Start Worker"):
            subprocess.Popen(["python3", "src/worker.py"], start_new_session=True)
            st.rerun()
