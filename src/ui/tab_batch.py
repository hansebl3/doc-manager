import streamlit as st
import os
import json

def render_batch_tab():
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
                                       value=st.session_state.get("prompt_meta", "Extract the document date, key technical keywords, and a short descriptive title (around 20 characters). Return ONLY a JSON object with 'date' (YYYY-MM-DD or similar), 'keywords' (list of strings), and 'title' (A short, descriptive title as a string. MANDATORY: DO NOT LEAVE EMPTY)."),
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
        
    tasks_active = st.session_state.db.get_tasks_by_status('processing_l') + st.session_state.db.get_tasks_by_status('processing_r') + st.session_state.db.get_tasks_by_status('processing')
    # Included 'processing' as well since we unified it in worker.py
    
    if tasks_active:
        st.warning(f"Currently Processing: {len(tasks_active)} docs")
        for t in tasks_active:
             fname = t['config'].get('filename', str(t['doc_id']))
             st.text(f"Processing: {fname} [{t['status']}]")
    else:
        st.info("Worker is idle or all tasks done.")
