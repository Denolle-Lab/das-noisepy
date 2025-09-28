"""
Configuration management utilities
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Parameters:
    -----------
    config_path : str or Path
        Path to configuration file
        
    Returns:
    --------
    Dict[str, Any]
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    Save configuration to YAML or JSON file
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary
    config_path : str or Path
        Output file path
    """
    config_path = Path(config_path)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
    
    logger.info(f"Saved configuration to {config_path}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for DAS-NoisePy processing
    
    Returns:
    --------
    Dict[str, Any]
        Default configuration
    """
    config = {
        # DAS data parameters
        'das': {
            'sampling_rate': 1000.0,      # Hz
            'channel_spacing': 1.0,       # m
            'gauge_length': 10.0,         # m
            'data_format': 'hdf5',        # 'hdf5', 'segy', 'tdms'
        },
        
        # Preprocessing parameters
        'preprocessing': {
            'detrend': True,
            'taper': True,
            'taper_fraction': 0.05,
            'bandpass_filter': True,
            'freq_min': 0.01,             # Hz
            'freq_max': 10.0,             # Hz
            'decimate_factor': None,
            'quality_control': True,
            'snr_threshold': 2.0,
        },
        
        # Channel pair selection
        'channel_pairs': {
            'selection_method': 'distance',  # 'distance', 'quality', 'adaptive', 'custom'
            'min_separation': 10.0,          # m
            'max_separation': 1000.0,        # m
            'step_size': None,               # m, if None uses all separations
            'max_pairs_per_separation': None,
            'target_pairs_per_distance': 50,
            'distance_bins': 20,
        },
        
        # Cross-correlation parameters
        'cross_correlation': {
            'method': 'fft',              # 'fft' or 'time_domain'
            'max_lag_time': 10.0,         # seconds
            'normalization': 'cross',     # 'cross', 'auto', or 'none'
            'whitening': True,
            'whitening_freqs': [0.1, 10.0],  # Hz
            'time_window_length': 3600,   # seconds
            'time_window_overlap': 0.5,   # fraction
            'stack_method': 'linear',     # 'linear', 'pws', 'robust'
            'cc_threshold': 0.01,         # minimum correlation coefficient
        },
        
        # I/O parameters
        'io': {
            'output_directory': './output',
            'save_individual_correlations': False,
            'compression': 'gzip',
            'chunk_size': 1024,
        },
        
        # Processing parameters
        'processing': {
            'parallel': True,
            'n_processes': 4,
            'memory_limit': '8GB',
            'chunk_processing': True,
        },
    }
    
    return config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fill missing values in configuration
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Input configuration
        
    Returns:
    --------
    Dict[str, Any]
        Validated configuration with defaults filled
    """
    default_config = get_default_config()
    
    def merge_configs(default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults"""
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    validated_config = merge_configs(default_config, config)
    
    # Validate specific parameters
    das_config = validated_config['das']
    if das_config['sampling_rate'] <= 0:
        raise ValueError("Sampling rate must be positive")
    
    if das_config['channel_spacing'] <= 0:
        raise ValueError("Channel spacing must be positive")
    
    preproc_config = validated_config['preprocessing']
    if preproc_config['freq_min'] >= preproc_config['freq_max']:
        raise ValueError("freq_min must be less than freq_max")
    
    xcorr_config = validated_config['cross_correlation']
    if xcorr_config['max_lag_time'] <= 0:
        raise ValueError("max_lag_time must be positive")
    
    if xcorr_config['time_window_overlap'] < 0 or xcorr_config['time_window_overlap'] >= 1:
        raise ValueError("time_window_overlap must be between 0 and 1")
    
    logger.info("Configuration validated successfully")
    return validated_config


def create_example_config(output_path: Union[str, Path]):
    """
    Create an example configuration file
    
    Parameters:
    -----------
    output_path : str or Path
        Path where to save the example configuration
    """
    config = get_default_config()
    
    # Add comments as a separate structure for documentation
    comments = {
        'das': 'DAS-specific parameters',
        'preprocessing': 'Data preprocessing options',
        'channel_pairs': 'Channel pair selection strategy',
        'cross_correlation': 'Cross-correlation computation settings',
        'io': 'Input/output configuration',
        'processing': 'Processing and performance settings',
    }
    
    # Save configuration with comments
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write("# DAS-NoisePy Configuration File\n")
        f.write("# This file contains all parameters for DAS cross-correlation processing\n\n")
        
        for section, comment in comments.items():
            f.write(f"# {comment}\n")
            section_config = {section: config[section]}
            yaml.dump(section_config, f, default_flow_style=False, indent=2)
            f.write("\n")
    
    logger.info(f"Created example configuration at {output_path}")


def print_config_summary(config: Dict[str, Any]):
    """
    Print a summary of the configuration
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary
    """
    print("DAS-NoisePy Configuration Summary")
    print("=" * 50)
    
    das_config = config.get('das', {})
    print(f"DAS Parameters:")
    print(f"  Sampling rate: {das_config.get('sampling_rate')} Hz")
    print(f"  Channel spacing: {das_config.get('channel_spacing')} m")
    print(f"  Gauge length: {das_config.get('gauge_length')} m")
    print(f"  Data format: {das_config.get('data_format')}")
    
    preproc_config = config.get('preprocessing', {})
    print(f"\nPreprocessing:")
    print(f"  Detrend: {preproc_config.get('detrend')}")
    print(f"  Bandpass filter: {preproc_config.get('bandpass_filter')}")
    if preproc_config.get('bandpass_filter'):
        print(f"  Frequency range: {preproc_config.get('freq_min')}-{preproc_config.get('freq_max')} Hz")
    print(f"  Quality control: {preproc_config.get('quality_control')}")
    
    pair_config = config.get('channel_pairs', {})
    print(f"\nChannel Pair Selection:")
    print(f"  Method: {pair_config.get('selection_method')}")
    print(f"  Separation range: {pair_config.get('min_separation')}-{pair_config.get('max_separation')} m")
    
    xcorr_config = config.get('cross_correlation', {})
    print(f"\nCross-correlation:")
    print(f"  Method: {xcorr_config.get('method')}")
    print(f"  Max lag time: {xcorr_config.get('max_lag_time')} s")
    print(f"  Normalization: {xcorr_config.get('normalization')}")
    print(f"  Spectral whitening: {xcorr_config.get('whitening')}")
    print(f"  Stack method: {xcorr_config.get('stack_method')}")
    
    processing_config = config.get('processing', {})
    print(f"\nProcessing:")
    print(f"  Parallel: {processing_config.get('parallel')}")
    print(f"  Number of processes: {processing_config.get('n_processes')}")
    print(f"  Memory limit: {processing_config.get('memory_limit')}")
    print("")