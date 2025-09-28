"""
Utility functions for NoisePy DAS I/O
"""

import logging
import os
import posixpath
import time
from typing import Dict, Any
from urllib.parse import urlparse

import fsspec


def get_filesystem(path: str, storage_options: Dict[str, Any] = None) -> fsspec.AbstractFileSystem:
    """Construct an fsspec filesystem for the given path"""
    if storage_options is None:
        storage_options = {}
    
    url = urlparse(path)
    if url.scheme == "s3":
        return fsspec.filesystem(url.scheme, **storage_options)
    elif url.scheme == "https":
        return fsspec.filesystem(url.scheme)
    else:
        return fsspec.filesystem("file", **storage_options)


def fs_join(path1: str, path2: str) -> str:
    """Helper for joining two paths that can handle both S3 URLs and local file system paths"""
    url = urlparse(path1)
    if url.scheme == "s3":
        return posixpath.join(path1, path2)
    else:
        return os.path.join(path1, path2)


class TimeLogger:
    """
    A utility class to measure and log the time spent in code fragments. 
    """
    
    enabled: bool = True

    def __init__(self, logger: logging.Logger = None, level: int = logging.DEBUG, prefix: str = None):
        """Create an instance that will use the given logger and logging level to log the times"""
        self.logger = logger or logging.getLogger(__name__)
        self.level = level
        self.prefix = "" if prefix is None else f" {prefix}"
        self.reset()

    def reset(self) -> float:
        """Reset the time checkpoint"""
        self.time = time.time()
        return self.time

    def log(self, message: str = None, start: float = -1.0) -> float:
        """Log the time elapsed since the last checkpoint or since start"""
        stop = time.time()
        dt = stop - self.time if start <= 0 else stop - start
        self.reset()
        self.log_raw(message, dt)
        return self.time

    def log_raw(self, message: str, dt: float):
        """Log the raw time difference"""
        if self.enabled:
            self.logger.log(self.level, f"TIMING{self.prefix}: {dt:6.3f} secs for {message}")
        return self.time