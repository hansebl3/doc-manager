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
                
                for task in tasks:
                    doc_id = task['doc_id']
                    config = task['config']
                    doc = self.db.get_document(doc_id)
                    
                    if not doc:
                        logger.error(f"Doc {doc_id} not found in documents table.")
                        self.db.update_task(doc_id, status='failed', results={"error": "Document not found"})
                        continue

                    # Update status to processing
                    self.db.update_task(doc_id, status='processing')
                    
                    try:
                        current_results = task['results'] or {}
                        
                        # --- Left Model ---
                        logger.info(f"Processing {doc_id} - Left Model ({config.get('model_l')})")
                        meta_l = self.llm.extract_metadata(doc['content'], config['model_l'], config['prompt_meta'])
                        sum_l = self.llm.generate_content(doc['content'], config['model_l'], config['prompt_summary'])
                        current_results['meta_l'] = meta_l
                        current_results['sum_l'] = sum_l
                        
                        # Save partial (optional, but good for debugging)
                        self.db.update_task(doc_id, results=current_results)
                        
                        # --- Right Model ---
                        logger.info(f"Processing {doc_id} - Right Model ({config.get('model_r')})")
                        meta_r = self.llm.extract_metadata(doc['content'], config['model_r'], config['prompt_meta'])
                        sum_r = self.llm.generate_content(doc['content'], config['model_r'], config['prompt_summary'])
                        current_results['meta_r'] = meta_r
                        current_results['sum_r'] = sum_r
                        
                        # Complete
                        self.db.update_task(doc_id, status='done', results=current_results)
                        logger.info(f"Completed {doc_id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {doc_id}: {e}")
                        self.db.update_task(doc_id, status='failed', results={"error": str(e)})
                
                logger.info("Batch complete. Waiting for new tasks...")
                
            except Exception as main_e:
                logger.error(f"Worker Loop Error: {main_e}")
                time.sleep(5)

if __name__ == "__main__":
    worker = BackgroundWorker()
    worker.run()
