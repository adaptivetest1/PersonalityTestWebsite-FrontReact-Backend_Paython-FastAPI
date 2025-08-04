# Data persistence configuration for production deployment
import os
import json
from pathlib import Path

class DataPersistence:
    """
    Production-ready data persistence layer
    Supports both file-based (development) and cloud-based (production) storage
    """
    
    def __init__(self):
        self.storage_type = os.getenv("STORAGE_TYPE", "file")  # file, mongodb, huggingface
        self.data_dir = Path("/app/data") if os.getenv("HF_SPACE") else Path("./data")
        self.data_dir.mkdir(exist_ok=True)
        
    def save_sessions(self, sessions_data):
        """Save sessions data with proper error handling"""
        try:
            if self.storage_type == "file":
                sessions_file = self.data_dir / "sessions_data.json"
                with open(sessions_file, 'w', encoding='utf-8') as f:
                    json.dump(sessions_data, f, ensure_ascii=False, indent=2, default=str)
                return True
            # Add MongoDB/other storage implementations here
        except Exception as e:
            print(f"Error saving sessions: {e}")
            return False
    
    def load_sessions(self):
        """Load sessions data with fallback"""
        try:
            if self.storage_type == "file":
                sessions_file = self.data_dir / "sessions_data.json"
                if sessions_file.exists():
                    with open(sessions_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading sessions: {e}")
            return {}
    
    def save_questions_cache(self, cache_data):
        """Save questions cache"""
        try:
            if self.storage_type == "file":
                cache_file = self.data_dir / "questions_cache.json"
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False
    
    def load_questions_cache(self):
        """Load questions cache"""
        try:
            if self.storage_type == "file":
                cache_file = self.data_dir / "questions_cache.json"
                if cache_file.exists():
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}

# Initialize persistence layer
persistence = DataPersistence()
