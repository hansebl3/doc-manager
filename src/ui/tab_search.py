import streamlit as st
import pandas as pd
import json

def render_search_tab():
    st.header("Search Knowledge Base")
    col_s1, col_s2, col_s3, col_s4 = st.columns([2, 1, 1, 1])
    with col_s1:
        search_query = st.text_input("Text Search")
    with col_s2:
        search_cat = st.selectbox("Category Filter", ["ALL"] + st.session_state.categories)
    with col_s3:
        search_lvl = st.selectbox("Level Filter", ["ALL", "L0", "L1", "L2", "L3"])
    with col_s4:
        search_uuid = st.text_input("UUID Search")
    
    st.divider()
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        filter_no_task = st.checkbox("Show only documents WITHOUT an active task")
    with col_opt2:
        auto_expand_parent = st.checkbox("Automatically show parent content for summaries", value=True)
    
    cat_filter = None if search_cat == "ALL" else search_cat
    lvl_filter = None if search_lvl == "ALL" else search_lvl
    uuid_filter = search_uuid if search_uuid else None
    
    results = st.session_state.db.search_documents(query_text=search_query, category=cat_filter, level=lvl_filter, doc_id=uuid_filter)
    
    if filter_no_task and results:
        # Filter out results that have an active task
        new_results = []
        for r in results:
            if not st.session_state.db.get_task(r['id']):
                new_results.append(r)
        results = new_results

    if results:
        df = pd.DataFrame(results)
        # Drop large columns for display
        if 'embedding' in df.columns:
            display_df = df.drop(columns=['embedding'])
        else:
            display_df = df
            
        st.dataframe(display_df, use_container_width=True)
        
        for idx, row in df.iterrows():
            # Indentation for summaries
            is_summary = row['level'] in ['L1', 'L2', 'L3']
            if is_summary:
                c_indent, c_content = st.columns([1, 15])
                container = c_content
            else:
                container = st.container()
            
            with container:
                display_name = row['title'] if row['title'] else row['id']
                with st.expander(f"[{row['category']} / {row['level']}] {display_name}"):
                    c_info, c_down = st.columns([4, 1])
                    with c_down:
                        st.download_button(
                            label="Download MD",
                            data=row['content'],
                            file_name=f"{display_name}.md",
                            mime="text/markdown",
                            key=f"dl_{row['id']}"
                        )
                    
                    with c_info:
                        st.write(f"**Title:** {row['title']}")
                        st.write(f"**Metadata:** {json.dumps(row['metadata'], ensure_ascii=False)}")
                        st.write(f"**Content Snippet:** {row['content'][:500]}...")
                    
                    # Queue Status & Re-queue
                    task = st.session_state.db.get_task(row['id'])
                    if task:
                        st.info(f"**Task Status:** `{task['status']}`")
                    else:
                        st.warning("No Active Task in Queue")
                        if st.button(f"Add to Process Queue", key=f"re_q_{row['id']}"):
                            # Try to get filename from metadata or ID
                            fname = row['metadata'].get('filename') or str(row['id'])
                            st.session_state.db.enqueue_task(row['id'], config={"filename": fname})
                            st.success(f"Added {row['id']} to processing queue!")
                            st.rerun()

                    # EDIT CONTENT (Only for L0)
                    if row['category'] == 'L0':
                        with st.expander(f"Edit Content"):
                            new_content = st.text_area("Update Content", value=row['content'], height=200, key=f"edit_{row['id']}")
                            if st.button("Save & Reset Summaries", key=f"save_{row['id']}"):
                                new_emb = st.session_state.embedder.encode(new_content).tolist()
                                st.session_state.db.upsert_document(
                                    row['id'], row['category'], row['level'], row['metadata'], new_content, new_emb
                                )
                                if row.get('summary_uuids'):
                                    for sum_id in row['summary_uuids']:
                                        st.session_state.db.delete_document(sum_id)
                                    st.session_state.db.clear_summary_links(row['id'])
                                st.success("Content updated and old summaries removed.")
                                st.rerun()

                    if is_summary:
                        # Use source_uuids from DB if available, else fallback to metadata
                        parent_uuids = row.get('source_uuids') or []
                        
                        # Fallback for older records or if source_uuids is in metadata
                        if not parent_uuids:
                            m_p_id = row['metadata'].get('parent_id') or row.get('metadata', {}).get('original_meta', {}).get('parent_id')
                            if m_p_id:
                                parent_uuids = [m_p_id]
                        
                        if parent_uuids:
                            for p_uuid in parent_uuids:
                                if auto_expand_parent:
                                    parent = st.session_state.db.get_document(p_uuid)
                                    if parent:
                                        st.info(f"**Parent Context ({parent['category']} / {parent['level']}):**\n\n{parent['content']}")
                                else:
                                    if st.button(f"Show Parent ({p_uuid[:8]})", key=f"btn_{row['id']}_{p_uuid}"):
                                        parent = st.session_state.db.get_document(p_uuid)
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
                                if row['level'] == 'L0' and row.get('summary_uuids'):
                                    st.warning(f"Deleting L0 {row['id']} and {len(row['summary_uuids'])} linked summaries.")
                                    for sum_id in row['summary_uuids']:
                                        st.session_state.db.delete_document(sum_id)
                                
                                st.session_state.db.delete_document(row['id'])
                                
                                if row['level'] == 'L1':
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
