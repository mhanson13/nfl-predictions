"""
HTTP connection pooling and session management for improved performance.

This module provides connection pooling, session reuse, and performance
optimizations for HTTP requests across the application.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PoolConfig:
    """Configuration for connection pooling."""
    
    # Connection pool settings
    pool_connections: int = 10  # Number of connection pools to cache
    pool_maxsize: int = 20  # Max connections per pool
    max_retries: int = 3  # Max retry attempts
    pool_block: bool = False  # Block when pool is full
    
    # Timeout settings
    connect_timeout: float = 10.0  # Connection timeout
    read_timeout: float = 30.0  # Read timeout
    
    # Retry settings
    retry_on_status: tuple = (429, 500, 502, 503, 504)
    backoff_factor: float = 0.5
    
    # Keep-alive settings
    keep_alive: bool = True
    
    def to_retry(self) -> Retry:
        """Convert to urllib3 Retry object."""
        return Retry(
            total=self.max_retries,
            status_forcelist=self.retry_on_status,
            backoff_factor=self.backoff_factor,
            raise_on_status=False
        )


class HTTPSessionPool:
    """
    Manages a pool of HTTP sessions with connection pooling.
    
    Provides thread-safe access to reusable HTTP sessions with
    optimized connection pooling and retry logic.
    """
    
    def __init__(self, config: Optional[PoolConfig] = None):
        """
        Initialize session pool.
        
        Args:
            config: Pool configuration
        """
        self.config = config or PoolConfig()
        self._sessions: Dict[str, requests.Session] = {}
        self._lock = threading.Lock()
        self.logger = get_logger(f"{__name__}.HTTPSessionPool")
    
    def get_session(self, name: str = "default") -> requests.Session:
        """
        Get or create a session with connection pooling.
        
        Args:
            name: Session name for isolation
            
        Returns:
            Configured requests.Session
        """
        with self._lock:
            if name not in self._sessions:
                self._sessions[name] = self._create_session()
                self.logger.debug(f"Created new session: {name}")
            return self._sessions[name]
    
    def _create_session(self) -> requests.Session:
        """Create a new session with connection pooling."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = self.config.to_retry()
        
        # Create adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.config.pool_connections,
            pool_maxsize=self.config.pool_maxsize,
            max_retries=retry_strategy,
            pool_block=self.config.pool_block
        )
        
        # Mount adapter for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default timeout
        session.timeout = (self.config.connect_timeout, self.config.read_timeout)
        
        # Configure keep-alive
        if self.config.keep_alive:
            session.headers.update({"Connection": "keep-alive"})
        
        return session
    
    def close_session(self, name: str):
        """
        Close and remove a specific session.
        
        Args:
            name: Session name
        """
        with self._lock:
            if name in self._sessions:
                self._sessions[name].close()
                del self._sessions[name]
                self.logger.debug(f"Closed session: {name}")
    
    def close_all(self):
        """Close all sessions in the pool."""
        with self._lock:
            for name, session in self._sessions.items():
                session.close()
                self.logger.debug(f"Closed session: {name}")
            self._sessions.clear()
    
    @contextmanager
    def session(self, name: str = "default"):
        """
        Context manager for session usage.
        
        Args:
            name: Session name
            
        Yields:
            requests.Session
            
        Example:
            with pool.session("api") as session:
                response = session.get("https://api.example.com")
        """
        session = self.get_session(name)
        try:
            yield session
        finally:
            pass  # Don't close, keep in pool


class HTTPXSessionPool:
    """
    Manages a pool of httpx clients with connection pooling.
    
    Similar to HTTPSessionPool but for httpx (async-capable).
    """
    
    def __init__(self, config: Optional[PoolConfig] = None):
        """
        Initialize httpx session pool.
        
        Args:
            config: Pool configuration
        """
        self.config = config or PoolConfig()
        self._clients: Dict[str, httpx.Client] = {}
        self._lock = threading.Lock()
        self.logger = get_logger(f"{__name__}.HTTPXSessionPool")
    
    def get_client(self, name: str = "default", base_url: Optional[str] = None) -> httpx.Client:
        """
        Get or create an httpx client with connection pooling.
        
        Args:
            name: Client name for isolation
            base_url: Optional base URL for the client
            
        Returns:
            Configured httpx.Client
        """
        with self._lock:
            key = f"{name}:{base_url}" if base_url else name
            if key not in self._clients:
                self._clients[key] = self._create_client(base_url)
                self.logger.debug(f"Created new httpx client: {key}")
            return self._clients[key]
    
    def _create_client(self, base_url: Optional[str] = None) -> httpx.Client:
        """Create a new httpx client with connection pooling."""
        limits = httpx.Limits(
            max_connections=self.config.pool_maxsize,
            max_keepalive_connections=self.config.pool_connections,
            keepalive_expiry=30.0  # Keep connections alive for 30s
        )
        
        timeout = httpx.Timeout(
            connect=self.config.connect_timeout,
            read=self.config.read_timeout,
            write=self.config.read_timeout,
            pool=5.0
        )
        
        return httpx.Client(
            base_url=base_url,
            limits=limits,
            timeout=timeout,
            http2=True,  # Enable HTTP/2 for better performance
            follow_redirects=True
        )
    
    def close_client(self, name: str, base_url: Optional[str] = None):
        """
        Close and remove a specific client.
        
        Args:
            name: Client name
            base_url: Base URL if specified
        """
        with self._lock:
            key = f"{name}:{base_url}" if base_url else name
            if key in self._clients:
                self._clients[key].close()
                del self._clients[key]
                self.logger.debug(f"Closed httpx client: {key}")
    
    def close_all(self):
        """Close all clients in the pool."""
        with self._lock:
            for key, client in self._clients.items():
                client.close()
                self.logger.debug(f"Closed httpx client: {key}")
            self._clients.clear()
    
    @contextmanager
    def client(self, name: str = "default", base_url: Optional[str] = None):
        """
        Context manager for client usage.
        
        Args:
            name: Client name
            base_url: Optional base URL
            
        Yields:
            httpx.Client
            
        Example:
            with pool.client("api", "https://api.example.com") as client:
                response = client.get("/endpoint")
        """
        client = self.get_client(name, base_url)
        try:
            yield client
        finally:
            pass  # Don't close, keep in pool


# Global session pools
_requests_pool: Optional[HTTPSessionPool] = None
_httpx_pool: Optional[HTTPXSessionPool] = None
_pool_lock = threading.Lock()


def get_requests_pool(config: Optional[PoolConfig] = None) -> HTTPSessionPool:
    """
    Get global requests session pool.
    
    Args:
        config: Optional configuration (only used on first call)
        
    Returns:
        HTTPSessionPool instance
    """
    global _requests_pool
    with _pool_lock:
        if _requests_pool is None:
            _requests_pool = HTTPSessionPool(config)
        return _requests_pool


def get_httpx_pool(config: Optional[PoolConfig] = None) -> HTTPXSessionPool:
    """
    Get global httpx session pool.
    
    Args:
        config: Optional configuration (only used on first call)
        
    Returns:
        HTTPXSessionPool instance
    """
    global _httpx_pool
    with _pool_lock:
        if _httpx_pool is None:
            _httpx_pool = HTTPXSessionPool(config)
        return _httpx_pool


def close_all_pools():
    """Close all global session pools."""
    global _requests_pool, _httpx_pool
    with _pool_lock:
        if _requests_pool:
            _requests_pool.close_all()
            _requests_pool = None
        if _httpx_pool:
            _httpx_pool.close_all()
            _httpx_pool = None


# Convenience functions
@contextmanager
def pooled_session(name: str = "default", config: Optional[PoolConfig] = None):
    """
    Get a pooled requests session.
    
    Args:
        name: Session name
        config: Optional pool configuration
        
    Yields:
        requests.Session
        
    Example:
        with pooled_session("espn_api") as session:
            response = session.get("https://api.espn.com/data")
    """
    pool = get_requests_pool(config)
    with pool.session(name) as session:
        yield session


@contextmanager
def pooled_client(
    name: str = "default",
    base_url: Optional[str] = None,
    config: Optional[PoolConfig] = None
):
    """
    Get a pooled httpx client.
    
    Args:
        name: Client name
        base_url: Optional base URL
        config: Optional pool configuration
        
    Yields:
        httpx.Client
        
    Example:
        with pooled_client("sportradar", "https://api.sportradar.com") as client:
            response = client.get("/nfl/data")
    """
    pool = get_httpx_pool(config)
    with pool.client(name, base_url) as client:
        yield client

# Made with Bob
