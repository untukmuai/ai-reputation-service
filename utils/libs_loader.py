"""
Universal JSON Data Loader - Load any JSON files from assets/libs/

Simple singleton to load and cache JSON configuration files at startup.
"""

import orjson
from pathlib import Path
from typing import Dict, Any


class LibsLoader:
    """
    Universal JSON loader for configuration files.
    
    Loads all JSON files from assets/libs/ once at startup and caches them.
    Access any loaded JSON via: libs_loader.get('filename')
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if LibsLoader._initialized:
            return
        
        # Setup paths
        self.project_root = Path(__file__).parent.parent.resolve()
        self.libs_dir = self.project_root / "assets" / "libs"
        self.libs_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded data
        self._data: Dict[str, Any] = {}
        
        LibsLoader._initialized = True
    
    def load_all(self) -> None:
        """Load all JSON files from assets/libs/ directory."""
        print(f"🔄 Loading JSON files from {self.libs_dir}")
        
        json_files = list(self.libs_dir.glob("*.json"))
        
        if not json_files:
            print("No JSON files found in assets/libs/")
            return
        
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    data = orjson.loads(content)
                    
                # Store with filename (without extension) as key
                key = json_file.stem
                self._data[key] = {
                    'raw': content,      # Raw string for LLM prompts
                    'parsed': data,      # Parsed dict for processing
                    'path': json_file    # Path reference
                }
                
                print(f"Loaded {json_file.name} ({len(content)} bytes)")
                
            except Exception as e:
                print(f"Failed to load {json_file.name}: {str(e)}")
        
        print(f"Loaded {len(self._data)} JSON file(s)")
    
    def get(self, name: str, parsed: bool = True) -> Any:
        """
        Get loaded JSON data by filename (without extension).
        
        Args:
            name: Filename without .json extension (e.g., 'persona', 'config')
            parsed: If True, returns parsed dict. If False, returns raw string.
        
        Returns:
            Parsed dict or raw string content
        
        Raises:
            KeyError: If file not found
        """
        if name not in self._data:
            available = list(self._data.keys())
            raise KeyError(f"'{name}.json' not loaded. Available: {available}")
        
        return self._data[name]['parsed'] if parsed else self._data[name]['raw']
    
    def get_raw(self, name: str) -> str:
        """Get raw string content (useful for LLM prompts)."""
        return self.get(name, parsed=False)
    
    def get_parsed(self, name: str) -> Dict[str, Any]:
        """Get parsed dict (useful for processing)."""
        return self.get(name, parsed=True)
    
    def list_loaded(self) -> list:
        """List all loaded JSON files."""
        return list(self._data.keys())
    
    def reload(self) -> None:
        """Reload all JSON files from disk."""
        print("🔄 Reloading JSON files...")
        self._data.clear()
        self.load_all()


# Singleton instance - import and use this
libs_loader = LibsLoader()

