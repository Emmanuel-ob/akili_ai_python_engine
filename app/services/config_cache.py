from typing import Dict, Any, Optional
import time
from app.core.logging_config import logger


class ConfigCache:
    """
    In-memory cache for database configurations

    Eliminates circular HTTP calls by caching configs sent from Laravel
    TTL: 5 minutes (configs rarely change)
    """

    def __init__(self, ttl: int = 300):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}
        self.ttl = ttl  # 5 minutes default

    def set(self, connection_id: str, config: Dict[str, Any]) -> None:
        """
        Cache database configuration

        Args:
            connection_id: Unique connection identifier
            config: Database configuration dict
        """
        self._cache[connection_id] = config
        self._timestamps[connection_id] = time.time()
        logger.debug(f"✅ Cached config for connection: {connection_id}")

    def get(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached configuration

        Returns None if not in cache or expired
        """
        if connection_id not in self._cache:
            logger.warning(f"⚠️ Config not in cache: {connection_id}")
            return None

        # Check if expired
        age = time.time() - self._timestamps[connection_id]
        if age > self.ttl:
            logger.warning(f"⚠️ Config expired for {connection_id} (age: {age}s)")
            del self._cache[connection_id]
            del self._timestamps[connection_id]
            return None

        logger.debug(f"✅ Retrieved cached config: {connection_id} (age: {age:.1f}s)")
        return self._cache[connection_id]

    def invalidate(self, connection_id: str) -> None:
        """Remove config from cache"""
        if connection_id in self._cache:
            del self._cache[connection_id]
            del self._timestamps[connection_id]
            logger.info(f"🗑️ Invalidated cache for: {connection_id}")

    def clear(self) -> None:
        """Clear entire cache"""
        self._cache.clear()
        self._timestamps.clear()
        logger.info("🗑️ Cleared entire config cache")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_configs": len(self._cache),
            "oldest_age": max(
                [time.time() - ts for ts in self._timestamps.values()], default=0
            ),
            "ttl": self.ttl,
        }
