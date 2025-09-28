"""
DAS-NoisePy: A fullstack of scripts for using NoisePy with DAS data

This package provides tools for processing Distributed Acoustic Sensing (DAS) data
using the NoisePy framework, with enhanced flexibility for channel pair selection
and cross-correlation analysis.
"""

__version__ = "0.1.0"
__author__ = "Denolle Lab"
__email__ = "denolle@uw.edu"

from .core.das_module import DASProcessor
from .core.channel_pairs import ChannelPairSelector
from .processing.xcorr import CrossCorrelator
from .utils.config import load_config, save_config, get_default_config
from .utils.io import load_das_data, save_results, create_synthetic_das_data

__all__ = [
    "DASProcessor",
    "ChannelPairSelector", 
    "CrossCorrelator",
    "load_config",
    "save_config",
    "get_default_config",
    "load_das_data",
    "save_results",
    "create_synthetic_das_data",
]