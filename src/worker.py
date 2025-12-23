import time
import json
import logging
from db_manager import DBManager
from llm_client import LLMClient

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("worker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Worker")

class BackgroundWorker:
    def __init__(self):
        self.db = DBManager()
        self.llm = LLMClient()
        logger.info("Worker Initialized")
        self._recover_stuck_tasks()

    def _recover_stuck_tasks(self):
        """Reset any tasks stuck in processing state back to queued on startup."""
        stuck_l = self.db.get_tasks_by_status('processing_l')
        stuck_r = self.db.get_tasks_by_status('processing_r')
        count = 0
        
        for task in stuck_l + stuck_r:
            logger.warning(f"Recovering stuck task {task['doc_id']} from {task['status']} to queued")
            self.db.update_task(task['doc_id'], status='queued')
            count += 1
            
        if count > 0:
            logger.info(f"Recovered {count} stuck tasks.")

    def run(self):
        logger.info("Worker Interrupted. Starting loop...")
        while True:
            try:
                # 1. Fetch queued tasks
                tasks = self.db.get_tasks_by_status('queued')
                
                if not tasks:
                    time.sleep(2) # Wait if no tasks
                    continue
                
                logger.info(f"Found {len(tasks)} queued tasks. Starting batch...")
                
                # Update status to processing (to avoid double pick-up if we had multiple workers, 
                # though here we have one. Good practice.)
                # Also, we want to run Left Model first for ALL, then Right Model for ALL.
                
                # --- Phase 1: Left Model (All Docs) ---
                for task in tasks:
                    doc_id = task['doc_id']
                    config = task['config']
                    doc = self.db.get_document(doc_id)
                    
                    if not doc:
                        logger.error(f"Doc {doc_id} not found in documents table.")
                        self.db.update_task(doc_id, status='failed', results={"error": "Document not found"})
                        continue

                    # Update status to processing_l
                    self.db.update_task(doc_id, status='processing_l')
                    
                    try:
                        # Extract Meta Left
                        logger.info(f"Processing {doc_id} - Left Meta")
                        meta_l = self.llm.extract_metadata(doc['content'], config['model_l'], config['prompt_meta'])
                        
                        # Generate Summary Left
                        logger.info(f"Processing {doc_id} - Left Summary")
                        sum_l = self.llm.generate_content(doc['content'], config['model_l'], config['prompt_summary'])
                        
                        # Save Intermediate Results
                        # Note: We need to merge with existing results if any (though usually empty at start)
                        current_results = task['results'] or {}
                        current_results['meta_l'] = meta_l
                        current_results['sum_l'] = sum_l
                        
                        self.db.update_task(doc_id, results=current_results)
                        
                    except Exception as e:
                        logger.error(f"Error processing Left Model for {doc_id}: {e}")
                        self.db.update_task(doc_id, status='failed', results={"error": str(e)})

                # --- Phase 2: Right Model (All Docs) ---
                # Re-fetch or iterate same list? 
                # If we iterate same list, we assume they didn't fail in phase 1.
                # Let's re-check status just to be safe or just continue if status is not failed.
                
                for task in tasks:
                    doc_id = task['doc_id']
                    # Check current status from DB to see if it failed in Phase 1
                    current_task = self.db.get_task(doc_id)
                    if current_task['status'] == 'failed':
                        continue
                        
                    config = current_task['config']
                    # We already have results in DB, so we append to them.
                    current_results = current_task['results']
                    doc = self.db.get_document(doc_id)
                    
                    # Update status to processing_r
                    self.db.update_task(doc_id, status='processing_r')
                    
                    try:
                        # Extract Meta Right
                        logger.info(f"Processing {doc_id} - Right Meta")
                        meta_r = self.llm.extract_metadata(doc['content'], config['model_r'], config['prompt_meta'])
                        
                        # Generate Summary Right
                        logger.info(f"Processing {doc_id} - Right Summary")
                        sum_r = self.llm.generate_content(doc['content'], config['model_r'], config['prompt_summary'])
                        
                        # Save Final Results
                        current_results['meta_r'] = meta_r
                        current_results['sum_r'] = sum_r
                        
                        self.db.update_task(doc_id, status='done', results=current_results)
                        logger.info(f"Completed {doc_id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing Right Model for {doc_id}: {e}")
                        self.db.update_task(doc_id, status='failed', results={"error": str(e)})
                
                logger.info("Batch complete. Waiting for new tasks...")
                
            except Exception as main_e:
                logger.error(f"Worker Loop Error: {main_e}")
                time.sleep(5)

if __name__ == "__main__":
    worker = BackgroundWorker()
    worker.run()
