import requests
import json
import logging
import os

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, base_url="http://192.168.1.238:8080/v1"):
        self.base_url = base_url
        self.history_file = "prompt_history.json"
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
            except:
                self.history = []
        else:
            self.history = []

    def _save_history(self, prompt):
        if prompt not in self.history:
            self.history.insert(0, prompt)
            self.history = self.history[:5]  # Keep last 5
            try:
                with open(self.history_file, "w") as f:
                    json.dump(self.history, f, indent=4)
            except:
                pass

    def get_history(self):
        return self.history

    def generate_content(self, content, model, prompt_template):
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"{prompt_template}\n\nContent:\n{content}"}],
            "temperature": 0.3,
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error calling LLM ({model}): {e}")
            return f"Error: {e}"

    def extract_metadata(self, content, model, prompt_template):
        """Extracts date and keywords as a JSON object"""
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        # Expecting JSON response
        full_prompt = f"{prompt_template}\n\nContent:\n{content}\n\nReturn ONLY a JSON object with 'date' (YYYY-MM-DD or similar) and 'keywords' (list of strings)."
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return json.loads(data['choices'][0]['message']['content'])
        except Exception as e:
            logger.error(f"Metadata extract error: {e}")
            return {"date": "unknown", "keywords": []}
            
    def get_available_models(self):
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            data = response.json()
            return [m['id'] for m in data['data']]
        except:
            return ["Qwen3-80b-Instruct", "llama-3-8b"] # Fallback
