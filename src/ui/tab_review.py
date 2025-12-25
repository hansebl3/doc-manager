import streamlit as st
from datetime import datetime
from utils.md_processor import MDProcessor

def render_review_tab():
    st.header("Review & Confirm")
    
    # Fetch done tasks from DB
    done_tasks = st.session_state.db.get_tasks_by_status('done')
    
    if not done_tasks:
        st.info("No documents waiting for review.")
    else:
        # DEBUG
        # st.write(f"DEBUG: Found {len(done_tasks)} completed tasks")
        
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
        
        # Smart Deduplication
        def normalize_kw(s):
            return "".join(s.lower().split())

        seen_normalized = {}
        unique_keywords = []
        for k in kw_l + kw_r:
            norm = normalize_kw(k)
            if norm not in seen_normalized:
                seen_normalized[norm] = k
                unique_keywords.append(k)
        
        all_keywords = sorted(unique_keywords)
        
        col_k1, col_k2 = st.columns([3, 1])
        with col_k1:
            selected_keywords = st.multiselect("Select Keywords", all_keywords, default=all_keywords)
        with col_k2:
            # Priority: 1. Parent Metadata > 2. LLM Extraction > 3. Today
            existing_doc_peek = st.session_state.db.get_document(selected_task_id)
            parent_metadata_date = None
            if existing_doc_peek and existing_doc_peek.get('metadata'):
                parent_metadata_date = existing_doc_peek['metadata'].get('date')
            
            extracted_date = res['meta_l'].get("date") or res['meta_r'].get("date")
            
            # If extracted is "unknown", ignore it
            if extracted_date == "unknown":
                extracted_date = None
                
            suggested_date = parent_metadata_date or extracted_date or datetime.today().strftime("%Y-%m-%d")
            
            final_date = st.text_input("Date", value=suggested_date)

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
        
        # Suggested L1 Title
        existing_doc = st.session_state.db.get_document(selected_task_id)
        parent_title = existing_doc.get('title', 'Document') if existing_doc else 'Document'
        
        # Recommendation Logic: Use AI title if available, otherwise L1_ + first 30 chars of summary
        ai_recommended_title = res['meta_l'].get("title") or res['meta_r'].get("title")
        if ai_recommended_title and ai_recommended_title != "unknown":
            # User request: Do not truncate AI title
            suggested_l1_title = f"L1_{ai_recommended_title}"
        else:
            # Fallback for L1: Use snippet of the summary itself
            content_snippet = base_text.strip().split('\n')[0][:20] if base_text.strip() else "Untitled"
            suggested_l1_title = f"L1_{content_snippet}"

        l1_title_input = st.text_input("Summary Title", value=suggested_l1_title, key=f"l1_title_{selected_task_id}_{choice}")

        # Use a key that changes with choice so base_text updates correctly
        final_summary_text = st.text_area("Final Summary Content", value=base_text, height=300, key=f"final_{selected_task_id}_{choice}")
        
        # Choice of Final Edit & Save Buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("Confirm & Save to DB", type="primary", use_container_width=True):
                 with st.spinner("Saving..."):
                    doc_id = str(current_task['doc_id']) # Ensure string
                    summary_id = MDProcessor.generate_uuid_v7()
                    
                    # existing_doc was fetched earlier or below
                    existing_doc = st.session_state.db.get_document(selected_task_id)
                    original_meta = existing_doc['metadata'] if existing_doc else {}
                    
                    final_meta_l1 = {
                        "keywords": selected_keywords,
                        "date": final_date,
                        "parent_id": doc_id
                    }
                    
                    final_meta_l0 = existing_doc['metadata'] if existing_doc else {}
                    final_meta_l0['date'] = final_date
                    
                    if existing_doc and existing_doc.get('summary_uuids'):
                         for old_sum_id in existing_doc['summary_uuids']:
                            st.session_state.db.delete_document(old_sum_id)
                            st.session_state.db.remove_summary_link(doc_id, old_sum_id)
                    
                    st.session_state.db.upsert_document(doc_id, existing_doc['category'], "L0", final_meta_l0, existing_doc['content'], title=existing_doc.get('title'))
                    
                    summary_emb = st.session_state.embedder.encode(final_summary_text).tolist()
                    l1_title = l1_title_input.strip() if l1_title_input.strip() else f"L1_{parent_title}"
                    st.session_state.db.upsert_document(summary_id, existing_doc['category'], "L1", final_meta_l1, final_summary_text, summary_emb, title=l1_title)
                    
                    st.session_state.db.link_documents(doc_id, summary_id)
                    st.session_state.db.delete_task(doc_id)
                    
                    st.success(f"Saved L1 ({summary_id}). Task removed from queue.")
                    st.rerun()

        with col_btn2:
            if st.button("Re-queue (Re-summarize)", use_container_width=True):
                st.session_state.db.update_task(selected_task_id, status='queued')
                st.info("Task returned to queue for re-processing!")
                st.rerun()

        with col_btn3:
            if st.button("Delete Doc & Task", type="secondary", use_container_width=True):
                st.session_state.db.delete_task(selected_task_id)
                st.session_state.db.delete_document(selected_task_id)
                st.warning("Document and task deleted.")
                st.rerun()
