"""
Enhanced caching utilities with versioning and invalidation.

This module provides intelligent caching with version control,
TTL support, and automatic invalidation strategies.
"""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union
import pandas as pd

from src.config import get_config

T = TypeVar('T')


class CacheVersion:
    """Cache versioning for schema changes and data updates."""
    
    # Increment these when schemas change
    SCHEDULE_VERSION = "v1.0"
    FEATURES_VERSION = "v2.0"  # Incremented when feature engineering changes
    MODEL_VERSION = "v1.0"
    WEATHER_VERSION = "v1.0"
    
    @classmethod
    def get_version_hash(cls, *components: str) -> str:
        """
        Generate a hash from version components.
        
        Args:
            *components: Version strings to hash together
        
        Returns:
            Short hash string
        """
        combined = "|".join(components)
        return hashlib.md5(combined.encode()).hexdigest()[:8]


class CacheMetadata:
    """Metadata for cached items."""
    
    def __init__(
        self,
        version: str,
        created_at: datetime,
        ttl_hours: Optional[int] = None,
        data_hash: Optional[str] = None,
        extra: Optional[dict] = None
    ):
        self.version = version
        self.created_at = created_at
        self.ttl_hours = ttl_hours
        self.data_hash = data_hash
        self.extra = extra or {}
    
    def is_expired(self) -> bool:
        """Check if cache has expired based on TTL."""
        if self.ttl_hours is None:
            return False
        age = datetime.now() - self.created_at
        return age > timedelta(hours=self.ttl_hours)
    
    def is_valid(self, required_version: str) -> bool:
        """Check if cache is valid (not expired and correct version)."""
        return self.version == required_version and not self.is_expired()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "ttl_hours": self.ttl_hours,
            "data_hash": self.data_hash,
            "extra": self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CacheMetadata":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            ttl_hours=data.get("ttl_hours"),
            data_hash=data.get("data_hash"),
            extra=data.get("extra"),
        )


class SmartCache:
    """Smart caching with versioning and automatic invalidation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize smart cache.
        
        Args:
            cache_dir: Directory for cache files (defaults to config)
        """
        config = get_config()
        self.cache_dir = cache_dir or config.paths.processed_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = config.pipeline.enable_cache
        self.default_ttl = config.pipeline.cache_ttl_hours
    
    def _get_cache_path(self, key: str, suffix: str = ".pkl") -> Path:
        """Get path for cache file."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}{suffix}"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get path for metadata file."""
        return self._get_cache_path(key, ".meta.json")
    
    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of data for change detection."""
        if isinstance(data, pd.DataFrame):
            # Hash DataFrame shape and column names
            info = f"{data.shape}|{sorted(data.columns.tolist())}"
            return hashlib.md5(info.encode()).hexdigest()[:16]
        else:
            # Hash pickle representation
            try:
                pickled = pickle.dumps(data)
                return hashlib.md5(pickled).hexdigest()[:16]
            except Exception:
                return "unhashable"
    
    def get(
        self,
        key: str,
        version: str,
        default: Optional[T] = None
    ) -> Optional[T]:
        """
        Get item from cache if valid.
        
        Args:
            key: Cache key
            version: Required version
            default: Default value if not found or invalid
        
        Returns:
            Cached value or default
        """
        if not self.enabled:
            return default
        
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)
        
        if not cache_path.exists() or not meta_path.exists():
            return default
        
        try:
            # Load and validate metadata
            with open(meta_path, 'r') as f:
                meta = CacheMetadata.from_dict(json.load(f))
            
            if not meta.is_valid(version):
                # Invalid cache, clean up
                self.invalidate(key)
                return default
            
            # Load data
            if cache_path.suffix == ".parquet":
                data = pd.read_parquet(cache_path)
            else:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            
            return data
        
        except Exception as e:
            print(f"[cache] Error loading cache for {key}: {e}")
            self.invalidate(key)
            return default
    
    def set(
        self,
        key: str,
        value: Any,
        version: str,
        ttl_hours: Optional[int] = None,
        extra: Optional[dict] = None
    ) -> bool:
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            version: Cache version
            ttl_hours: Time-to-live in hours (None = no expiration)
            extra: Extra metadata
        
        Returns:
            True if successful
        """
        if not self.enabled:
            return False
        
        try:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            # Save data
            if isinstance(value, pd.DataFrame):
                cache_path = self._get_cache_path(key, ".parquet")
                value.to_parquet(cache_path)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            
            # Save metadata
            metadata = CacheMetadata(
                version=version,
                created_at=datetime.now(),
                ttl_hours=ttl_hours or self.default_ttl,
                data_hash=self._compute_data_hash(value),
                extra=extra
            )
            
            with open(meta_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"[cache] Error saving cache for {key}: {e}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate (delete) cache entry.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted
        """
        try:
            for suffix in [".pkl", ".parquet", ".meta.json"]:
                path = self._get_cache_path(key, suffix)
                if path.exists():
                    path.unlink()
            return True
        except Exception as e:
            print(f"[cache] Error invalidating cache for {key}: {e}")
            return False
    
    def clear_all(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = 0
        try:
            for path in self.cache_dir.glob("*"):
                if path.is_file():
                    path.unlink()
                    count += 1
        except Exception as e:
            print(f"[cache] Error clearing cache: {e}")
        return count
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = 0
        try:
            for meta_path in self.cache_dir.glob("*.meta.json"):
                try:
                    with open(meta_path, 'r') as f:
                        meta = CacheMetadata.from_dict(json.load(f))
                    
                    if meta.is_expired():
                        key = meta_path.stem.replace(".meta", "")
                        if self.invalidate(key):
                            count += 1
                except Exception:
                    continue
        except Exception as e:
            print(f"[cache] Error clearing expired cache: {e}")
        return count


def cached(
    version: str,
    ttl_hours: Optional[int] = None,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results.
    
    Args:
        version: Cache version string
        ttl_hours: Time-to-live in hours
        key_func: Function to generate cache key from args/kwargs
    
    Example:
        @cached(version="v1.0", ttl_hours=24)
        def expensive_function(season: int) -> pd.DataFrame:
            # ... expensive computation
            return result
    """
    def decorator(func: Callable) -> Callable:
        cache = SmartCache()
        
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: use function name and args
                args_str = f"{args}|{sorted(kwargs.items())}"
                cache_key = f"{func.__name__}:{args_str}"
            
            # Try to get from cache
            result = cache.get(cache_key, version)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result, version, ttl_hours)
            return result
        
        return wrapper
    return decorator


# Global cache instance
_cache: Optional[SmartCache] = None


def get_cache() -> SmartCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = SmartCache()
    return _cache

# Made with Bob
