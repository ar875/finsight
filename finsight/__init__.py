"""FinSight: question answering with citations over Indian company annual reports and earnings calls."""
from .config import deep_update, index_dir, index_key, load_config
from .pipeline import RAGPipeline

__all__ = ["RAGPipeline", "load_config", "deep_update", "index_key", "index_dir"]
__version__ = "1.7.1"
