import pickle
import os
import pandas as pd
from typing import Any, Optional

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = 'data/checkpoints/'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, data: Any, filename: str) -> None:
        """Save checkpoint data"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str) -> Optional[Any]:
        """Load checkpoint data"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"Checkpoint loaded: {filename}")
            return data
        except FileNotFoundError:
            print(f"Checkpoint not found: {filename}")
            return None
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pkl')]
        return sorted(checkpoints)
    
    def resume_from_checkpoint(self, checkpoint_name: str) -> Optional[Any]:
        """Resume work from a specific checkpoint"""
        return self.load_checkpoint(checkpoint_name)