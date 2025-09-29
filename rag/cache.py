"""
Caching layer for RAG system to improve performance.
Implements both in-memory and persistent caching strategies.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class SimpleCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of items to store
            default_ttl: Default time-to-live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps([args, sorted(kwargs.items())], sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if it exists and hasn't expired."""
        if key not in self.cache:
            return None
        
        item = self.cache[key]
        if time.time() > item['expires_at']:
            del self.cache[key]
            return None
        
        item['last_accessed'] = time.time()
        return item['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache with TTL."""
        # Evict oldest items if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            'value': value,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'expires_at': time.time() + ttl
        }
    
    def _evict_oldest(self):
        """Evict the least recently used item."""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), 
                        key=lambda k: self.cache[k]['last_accessed'])
        del self.cache[oldest_key]
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        active_items = sum(1 for item in self.cache.values() 
                          if item['expires_at'] > now)
        
        return {
            'total_items': len(self.cache),
            'active_items': active_items,
            'max_size': self.max_size,
            'cache_usage': len(self.cache) / self.max_size
        }

class PersistentCache:
    """Persistent cache using JSON files."""
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 86400):
        """
        Initialize persistent cache.
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from persistent cache."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if time.time() > data['expires_at']:
                file_path.unlink()  # Delete expired file
                return None
            
            return data['value']
            
        except (json.JSONDecodeError, KeyError, OSError):
            # Remove corrupted cache file
            if file_path.exists():
                file_path.unlink()
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in persistent cache."""
        file_path = self._get_file_path(key)
        ttl = ttl or self.default_ttl
        
        data = {
            'value': value,
            'created_at': time.time(),
            'expires_at': time.time() + ttl
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"Failed to write cache file {file_path}: {e}")
    
    def clear(self):
        """Clear all cached files."""
        for file_path in self.cache_dir.glob("*.json"):
            file_path.unlink()

# Global cache instances
_memory_cache = SimpleCache(max_size=500, default_ttl=1800)  # 30 minutes
_persistent_cache = PersistentCache(cache_dir="cache", default_ttl=86400)  # 24 hours

def cached_retrieval(cache_type: str = "memory", ttl: int = 1800):
    """
    Decorator for caching retrieval results.
    
    Args:
        cache_type: "memory" or "persistent"
        ttl: Time-to-live in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _memory_cache._generate_key(func.__name__, *args, **kwargs)
            
            # Select cache
            cache = _memory_cache if cache_type == "memory" else _persistent_cache
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            
            return result
        
        return wrapper
    return decorator

def cached_embedding(cache_type: str = "persistent", ttl: int = 604800):  # 1 week
    """
    Decorator for caching embedding computations.
    
    Args:
        cache_type: "memory" or "persistent"
        ttl: Time-to-live in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key based on input text and model
            text_input = str(args) + str(kwargs)
            cache_key = hashlib.md5(text_input.encode()).hexdigest()
            
            # Select cache
            cache = _memory_cache if cache_type == "memory" else _persistent_cache
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Embedding cache hit")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            logger.debug(f"Embedding cache miss, result cached")
            
            return result
        
        return wrapper
    return decorator

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches."""
    return {
        "memory_cache": _memory_cache.stats(),
        "persistent_cache": {
            "cache_dir": str(_persistent_cache.cache_dir),
            "cache_files": len(list(_persistent_cache.cache_dir.glob("*.json")))
        }
    }

def clear_all_caches():
    """Clear all caches."""
    _memory_cache.clear()
    _persistent_cache.clear()
    logger.info("All caches cleared")

# Example usage with existing retrievers
class CachedBM25Retriever:
    """BM25 retriever with caching."""
    
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
    
    @cached_retrieval(cache_type="memory", ttl=1800)
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Cached search method."""
        return self.base_retriever.search(query, top_k)

class CachedFaissTfidfRetriever:
    """FAISS TF-IDF retriever with caching."""
    
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
    
    @cached_retrieval(cache_type="memory", ttl=1800)
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Cached search method."""
        return self.base_retriever.search(query, top_k)
