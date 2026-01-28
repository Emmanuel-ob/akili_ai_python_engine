import time
from collections import deque
from typing import Dict
from app.core.logging_config import logger


class RateLimiter:
    """
    Simple rate limiter to track API requests per minute

    Used to switch from Gemini to Groq when rate limit is approached
    """

    def __init__(self, max_requests: int = 14, window_seconds: int = 60):
        """
        Initialize rate limiter

        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds (default 60 = 1 minute)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = {}  # Track requests per service

    def add_request(self, service: str = "gemini") -> None:
        """Record a new request"""
        if service not in self.requests:
            self.requests[service] = deque()

        current_time = time.time()
        self.requests[service].append(current_time)

        # Clean old requests outside the window
        self._clean_old_requests(service, current_time)

    def is_rate_limited(self, service: str = "gemini") -> bool:
        """
        Check if service has hit rate limit

        Returns:
            True if rate limit reached, False otherwise
        """
        if service not in self.requests:
            return False

        current_time = time.time()
        self._clean_old_requests(service, current_time)

        count = len(self.requests[service])
        is_limited = count >= self.max_requests

        if is_limited:
            logger.warning(
                f"Rate limit reached for {service}: {count}/{self.max_requests} requests in last {self.window_seconds}s"
            )

        return is_limited

    def get_request_count(self, service: str = "gemini") -> int:
        """Get current request count for a service"""
        if service not in self.requests:
            return 0

        current_time = time.time()
        self._clean_old_requests(service, current_time)
        return len(self.requests[service])

    def reset(self, service: str = "gemini") -> None:
        """Reset request counter for a service"""
        if service in self.requests:
            self.requests[service].clear()
            logger.info(f"Reset rate limiter for {service}")

    def _clean_old_requests(self, service: str, current_time: float) -> None:
        """Remove requests outside the time window"""
        if service not in self.requests:
            return

        cutoff_time = current_time - self.window_seconds

        # Remove old requests from the left side of deque
        while self.requests[service] and self.requests[service][0] < cutoff_time:
            self.requests[service].popleft()

    def get_stats(self) -> Dict[str, int]:
        """Get statistics for all services"""
        stats = {}
        current_time = time.time()

        for service in self.requests:
            self._clean_old_requests(service, current_time)
            stats[service] = {
                "current_count": len(self.requests[service]),
                "max_allowed": self.max_requests,
                "window_seconds": self.window_seconds,
                "is_limited": len(self.requests[service]) >= self.max_requests,
            }

        return stats


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=14, window_seconds=60)
