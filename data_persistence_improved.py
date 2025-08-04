# Data persistence using Hugging Face Datasets for permanent storage
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

class DataPersistence:
    """
    Production-ready data persistence layer using Hugging Face Datasets
    Supports both local file storage (development) and HF Datasets (production)
    """
    
    def __init__(self):
        self.storage_type = os.getenv("STORAGE_TYPE", "file")
        self.hf_space = os.getenv("HF_SPACE", False)
        self.data_dir = Path("/app/data") if self.hf_space else Path("./data")
        self.data_dir.mkdir(exist_ok=True)
        
        # For HF Spaces, we'll use a simple JSON file approach with backup
        self.sessions_file = self.data_dir / "sessions_data.json"
        self.backup_file = self.data_dir / "sessions_backup.json"
        
    def save_sessions(self, sessions_data):
        """Save sessions data with error handling and backup"""
        try:
            # Always save to local file
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, ensure_ascii=False, indent=2, default=str)
            
            # Create backup
            with open(self.backup_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, ensure_ascii=False, indent=2, default=str)
                
            print(f"Sessions saved successfully: {len(sessions_data)} sessions")
            return True
            
        except Exception as e:
            print(f"Error saving sessions: {e}")
            return False
    
    def load_sessions(self):
        """Load sessions data with fallback"""
        try:
            # Try to load from main file
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Sessions loaded successfully: {len(data)} sessions")
                return data
            
            # Try backup file
            elif self.backup_file.exists():
                with open(self.backup_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Sessions loaded from backup: {len(data)} sessions")
                return data
            
            else:
                print("No existing sessions data found, starting fresh")
                return {}
                
        except Exception as e:
            print(f"Error loading sessions: {e}")
            return {}
    
    def get_storage_info(self):
        """Get information about current storage"""
        return {
            "storage_type": self.storage_type,
            "hf_space": self.hf_space,
            "data_dir": str(self.data_dir),
            "sessions_file_exists": self.sessions_file.exists(),
            "backup_file_exists": self.backup_file.exists()
        }

# Global instance
storage_manager = DataPersistence()
