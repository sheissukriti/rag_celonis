"""
Enhanced caching layer with Redis support for response caching and session management.
"""

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle
import time

# Optional Redis import with fallback
try:
    import redis
    from redis.exceptions import ConnectionError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    ConnectionError = Exception
    TimeoutError = Exception

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = None

class MemoryCache:
    """In-memory cache implementation as fallback."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        logger.info("Initialized MemoryCache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if entry.expires_at and datetime.now() > entry.expires_at:
            del self.cache[key]
            return None
        
        # Update access metadata
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        return entry.data
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache."""
        try:
            # Clean up if at max capacity
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            ttl = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
            
            entry = CacheEntry(
                key=key,
                data=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                metadata={}
            )
            
            self.cache[key] = entry
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        self.cache.clear()
        return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if key not in self.cache:
            return False
        
        entry = self.cache[key]
        if entry.expires_at and datetime.now() > entry.expires_at:
            del self.cache[key]
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = 0
        total_accesses = 0
        
        for entry in self.cache.values():
            if entry.expires_at and datetime.now() > entry.expires_at:
                expired_entries += 1
            total_accesses += entry.access_count
        
        return {
            'type': 'memory',
            'total_entries': total_entries,
            'expired_entries': expired_entries,
            'total_accesses': total_accesses,
            'max_size': self.max_size
        }
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed or self.cache[k].created_at
        )
        
        del self.cache[lru_key]
        logger.debug(f"Evicted LRU cache entry: {lru_key}")

class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, password: Optional[str] = None,
                 default_ttl: int = 3600, key_prefix: str = 'rag:'):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            default_ttl: Default TTL in seconds
            key_prefix: Prefix for all cache keys
        """
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.client = None
        self.connected = False
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Install redis-py.")
            return
        
        try:
            self.client = redis.Redis(
                host=host, port=port, db=db, password=password,
                decode_responses=False, socket_connect_timeout=5
            )
            
            # Test connection
            self.client.ping()
            self.connected = True
            logger.info(f"Connected to Redis at {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
            self.connected = False
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache."""
        if not self.connected:
            return None
        
        try:
            full_key = self._make_key(key)
            data = self.client.get(full_key)
            
            if data is None:
                return None
            
            # Deserialize data
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in Redis cache."""
        if not self.connected:
            return False
        
        try:
            full_key = self._make_key(key)
            ttl = ttl or self.default_ttl
            
            # Serialize data
            data = pickle.dumps(value)
            
            # Set with TTL
            if ttl > 0:
                result = self.client.setex(full_key, ttl, data)
            else:
                result = self.client.set(full_key, data)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from Redis cache."""
        if not self.connected:
            return False
        
        try:
            full_key = self._make_key(key)
            result = self.client.delete(full_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries with the prefix."""
        if not self.connected:
            return False
        
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.client.keys(pattern)
            
            if keys:
                result = self.client.delete(*keys)
                return result > 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self.connected:
            return False
        
        try:
            full_key = self._make_key(key)
            return bool(self.client.exists(full_key))
            
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        if not self.connected:
            return {'type': 'redis', 'connected': False}
        
        try:
            info = self.client.info()
            pattern = f"{self.key_prefix}*"
            keys_count = len(self.client.keys(pattern))
            
            return {
                'type': 'redis',
                'connected': True,
                'total_entries': keys_count,
                'used_memory': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'type': 'redis', 'connected': False, 'error': str(e)}

class CacheManager:
    """Unified cache manager with fallback support."""
    
    def __init__(self, cache_config: Dict[str, Any]):
        """
        Initialize cache manager.
        
        Args:
            cache_config: Cache configuration dictionary
        """
        self.config = cache_config
        self.cache = None
        
        # Try to initialize Redis cache first
        if cache_config.get('type') == 'redis' and REDIS_AVAILABLE:
            redis_config = cache_config.get('redis', {})
            self.cache = RedisCache(**redis_config)
            
            # Fallback to memory cache if Redis fails
            if not self.cache.connected:
                logger.warning("Redis connection failed, falling back to memory cache")
                self.cache = MemoryCache(**cache_config.get('memory', {}))
        else:
            # Use memory cache
            self.cache = MemoryCache(**cache_config.get('memory', {}))
        
        logger.info(f"Initialized CacheManager with {type(self.cache).__name__}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache."""
        return self.cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        return self.cache.delete(key)
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        return self.cache.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.cache.exists(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

class ResponseCache:
    """Specialized cache for RAG responses."""
    
    def __init__(self, cache_manager: CacheManager, 
                 default_ttl: int = 3600, enable_query_normalization: bool = True):
        """
        Initialize response cache.
        
        Args:
            cache_manager: Cache manager instance
            default_ttl: Default TTL for cached responses
            enable_query_normalization: Whether to normalize queries for caching
        """
        self.cache_manager = cache_manager
        self.default_ttl = default_ttl
        self.enable_query_normalization = enable_query_normalization
        
        logger.info("Initialized ResponseCache")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching."""
        if not self.enable_query_normalization:
            return query
        
        # Basic normalization: lowercase, strip whitespace, remove extra spaces
        normalized = ' '.join(query.lower().strip().split())
        return normalized
    
    def _generate_cache_key(self, query: str, retrieval_params: Dict[str, Any]) -> str:
        """Generate cache key for query and parameters."""
        normalized_query = self._normalize_query(query)
        
        # Create a consistent string representation of parameters
        param_str = json.dumps(retrieval_params, sort_keys=True)
        
        # Generate hash
        cache_input = f"{normalized_query}|{param_str}"
        cache_key = hashlib.md5(cache_input.encode('utf-8')).hexdigest()
        
        return f"response:{cache_key}"
    
    def get_response(self, query: str, retrieval_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response for query."""
        cache_key = self._generate_cache_key(query, retrieval_params)
        
        cached_response = self.cache_manager.get(cache_key)
        if cached_response:
            logger.info(f"Cache hit for query: {query[:50]}...")
            
            # Update access metadata
            cached_response['cache_metadata'] = {
                'cache_hit': True,
                'accessed_at': datetime.now().isoformat(),
                'original_query': query
            }
            
            return cached_response
        
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    
    def cache_response(self, query: str, retrieval_params: Dict[str, Any], 
                      response: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache response for query."""
        cache_key = self._generate_cache_key(query, retrieval_params)
        ttl = ttl or self.default_ttl
        
        # Add cache metadata
        cached_response = response.copy()
        cached_response['cache_metadata'] = {
            'cached_at': datetime.now().isoformat(),
            'ttl': ttl,
            'original_query': query,
            'cache_key': cache_key
        }
        
        success = self.cache_manager.set(cache_key, cached_response, ttl)
        
        if success:
            logger.info(f"Cached response for query: {query[:50]}...")
        else:
            logger.warning(f"Failed to cache response for query: {query[:50]}...")
        
        return success
    
    def invalidate_query(self, query: str, retrieval_params: Dict[str, Any]) -> bool:
        """Invalidate cached response for specific query."""
        cache_key = self._generate_cache_key(query, retrieval_params)
        return self.cache_manager.delete(cache_key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get response cache statistics."""
        base_stats = self.cache_manager.get_stats()
        base_stats['default_ttl'] = self.default_ttl
        base_stats['query_normalization'] = self.enable_query_normalization
        
        return base_stats

# Default cache configurations
DEFAULT_CACHE_CONFIGS = {
    'redis': {
        'type': 'redis',
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'default_ttl': 3600,
            'key_prefix': 'rag:'
        },
        'memory': {  # Fallback config
            'max_size': 1000,
            'default_ttl': 3600
        }
    },
    'memory': {
        'type': 'memory',
        'memory': {
            'max_size': 1000,
            'default_ttl': 3600
        }
    }
}