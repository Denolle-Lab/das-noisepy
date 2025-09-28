"""
Input/Output utilities for DAS data and results
"""

import numpy as np
import xarray as xr
import h5py
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


def load_das_data(file_path: Union[str, Path], 
                 format: str = 'auto',
                 **kwargs) -> xr.Dataset:
    """
    Load DAS data from various formats
    
    Parameters:
    -----------
    file_path : str or Path
        Path to DAS data file
    format : str
        Data format ('auto', 'hdf5', 'segy', 'tdms', 'npy')
    **kwargs
        Additional arguments for specific loaders
        
    Returns:
    --------
    xr.Dataset
        Loaded DAS data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Auto-detect format if not specified
    if format == 'auto':
        format = _detect_format(file_path)
    
    logger.info(f"Loading DAS data from {file_path} (format: {format})")
    
    if format == 'hdf5':
        return _load_hdf5_das(file_path, **kwargs)
    elif format == 'segy':
        return _load_segy_das(file_path, **kwargs)
    elif format == 'tdms':
        return _load_tdms_das(file_path, **kwargs)
    elif format == 'npy':
        return _load_npy_das(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _detect_format(file_path: Path) -> str:
    """Auto-detect DAS data format from file extension"""
    suffix = file_path.suffix.lower()
    
    if suffix in ['.h5', '.hdf5', '.hdf']:
        return 'hdf5'
    elif suffix in ['.sgy', '.segy']:
        return 'segy'
    elif suffix == '.tdms':
        return 'tdms'
    elif suffix == '.npy':
        return 'npy'
    else:
        # Try to detect based on file content
        try:
            with h5py.File(file_path, 'r') as f:
                return 'hdf5'
        except:
            pass
        
        raise ValueError(f"Could not auto-detect format for {file_path}")


def _load_hdf5_das(file_path: Path, 
                   strain_key: str = 'strain',
                   time_key: str = 'time',
                   channel_key: str = 'channel',
                   **kwargs) -> xr.Dataset:
    """Load DAS data from HDF5 format"""
    
    with h5py.File(file_path, 'r') as f:
        # Load strain data
        if strain_key in f:
            strain = f[strain_key][:]
        else:
            # Try to find strain data automatically
            possible_keys = ['strain', 'data', 'strain_rate', 'das_data']
            strain_key = None
            for key in possible_keys:
                if key in f:
                    strain_key = key
                    break
            
            if strain_key is None:
                raise KeyError(f"Could not find strain data in {list(f.keys())}")
            
            strain = f[strain_key][:]
        
        # Get metadata from attributes
        attrs = dict(f.attrs)
        
        # Create coordinates
        n_time, n_channels = strain.shape
        
        # Try to load time coordinate
        if time_key in f:
            time = f[time_key][:]
        else:
            sampling_rate = attrs.get('sampling_rate', 1000.0)
            time = np.arange(n_time) / sampling_rate
        
        # Try to load channel coordinate
        if channel_key in f:
            channels = f[channel_key][:]
        else:
            channels = np.arange(n_channels)
        
        # Calculate distances
        channel_spacing = attrs.get('channel_spacing', 1.0)
        distances = channels * channel_spacing
        
        # Create xarray Dataset
        dataset = xr.Dataset(
            {
                'strain': (['time', 'channel'], strain),
                'distance': (['channel'], distances),
            },
            coords={
                'time': time,
                'channel': channels,
            },
            attrs=attrs
        )
        
        # Set default attributes if not present
        if 'sampling_rate' not in dataset.attrs:
            dataset.attrs['sampling_rate'] = 1000.0
        if 'channel_spacing' not in dataset.attrs:
            dataset.attrs['channel_spacing'] = 1.0
        if 'gauge_length' not in dataset.attrs:
            dataset.attrs['gauge_length'] = 10.0
        
        dataset.attrs['file_path'] = str(file_path)
    
    logger.info(f"Loaded HDF5 DAS data: {strain.shape} (time, channel)")
    return dataset


def _load_segy_das(file_path: Path, **kwargs) -> xr.Dataset:
    """Load DAS data from SEG-Y format"""
    try:
        import segyio
    except ImportError:
        raise ImportError("segyio package required for SEG-Y format")
    
    with segyio.open(file_path, 'r') as f:
        # Read traces
        strain = np.array([trace for trace in f.trace]).T
        
        # Get metadata
        n_time, n_channels = strain.shape
        
        # Extract sampling information
        sampling_rate = 1.0 / (f.bin[segyio.BinField.Interval] / 1e6)
        time = np.arange(n_time) / sampling_rate
        
        # Channel information
        channels = np.arange(n_channels)
        channel_spacing = 1.0  # Default, may need to be specified
        distances = channels * channel_spacing
        
        # Create dataset
        dataset = xr.Dataset(
            {
                'strain': (['time', 'channel'], strain),
                'distance': (['channel'], distances),
            },
            coords={
                'time': time,
                'channel': channels,
            },
            attrs={
                'sampling_rate': sampling_rate,
                'channel_spacing': channel_spacing,
                'gauge_length': 10.0,  # Default
                'file_path': str(file_path),
                'format': 'segy',
            }
        )
    
    logger.info(f"Loaded SEG-Y DAS data: {strain.shape} (time, channel)")
    return dataset


def _load_tdms_das(file_path: Path, **kwargs) -> xr.Dataset:
    """Load DAS data from TDMS format"""
    try:
        from nptdms import TdmsFile
    except ImportError:
        raise ImportError("nptdms package required for TDMS format")
    
    with TdmsFile.read(file_path) as tdms_file:
        # Get all groups and channels
        groups = tdms_file.groups()
        
        # Assume strain data is in the first group
        if not groups:
            raise ValueError("No groups found in TDMS file")
        
        group = groups[0]
        channels_data = group.channels()
        
        if not channels_data:
            raise ValueError("No channels found in TDMS group")
        
        # Read data from all channels
        strain_list = []
        for channel in channels_data:
            strain_list.append(channel[:])
        
        strain = np.array(strain_list).T
        n_time, n_channels = strain.shape
        
        # Get sampling rate from first channel properties
        first_channel = channels_data[0]
        dt = first_channel.properties.get('wf_increment', 1e-3)
        sampling_rate = 1.0 / dt
        
        time = np.arange(n_time) * dt
        channels = np.arange(n_channels)
        
        # Default spacing
        channel_spacing = 1.0
        distances = channels * channel_spacing
        
        # Create dataset
        dataset = xr.Dataset(
            {
                'strain': (['time', 'channel'], strain),
                'distance': (['channel'], distances),
            },
            coords={
                'time': time,
                'channel': channels,
            },
            attrs={
                'sampling_rate': sampling_rate,
                'channel_spacing': channel_spacing,
                'gauge_length': 10.0,
                'file_path': str(file_path),
                'format': 'tdms',
            }
        )
    
    logger.info(f"Loaded TDMS DAS data: {strain.shape} (time, channel)")
    return dataset


def _load_npy_das(file_path: Path, 
                  sampling_rate: float = 1000.0,
                  channel_spacing: float = 1.0,
                  **kwargs) -> xr.Dataset:
    """Load DAS data from NumPy format"""
    
    strain = np.load(file_path)
    
    # Ensure 2D array (time, channel)
    if strain.ndim == 1:
        strain = strain.reshape(-1, 1)
    
    n_time, n_channels = strain.shape
    
    # Create coordinates
    time = np.arange(n_time) / sampling_rate
    channels = np.arange(n_channels)
    distances = channels * channel_spacing
    
    # Create dataset
    dataset = xr.Dataset(
        {
            'strain': (['time', 'channel'], strain),
            'distance': (['channel'], distances),
        },
        coords={
            'time': time,
            'channel': channels,
        },
        attrs={
            'sampling_rate': sampling_rate,
            'channel_spacing': channel_spacing,
            'gauge_length': 10.0,
            'file_path': str(file_path),
            'format': 'npy',
        }
    )
    
    logger.info(f"Loaded NumPy DAS data: {strain.shape} (time, channel)")
    return dataset


def save_das_data(data: xr.Dataset, 
                 file_path: Union[str, Path],
                 format: str = 'hdf5',
                 compression: str = 'gzip',
                 **kwargs):
    """
    Save DAS data to file
    
    Parameters:
    -----------
    data : xr.Dataset
        DAS data to save
    file_path : str or Path
        Output file path
    format : str
        Output format ('hdf5', 'netcdf', 'npy')
    compression : str
        Compression method
    **kwargs
        Additional arguments for specific writers
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving DAS data to {file_path} (format: {format})")
    
    if format == 'hdf5':
        _save_hdf5_das(data, file_path, compression=compression, **kwargs)
    elif format == 'netcdf':
        data.to_netcdf(file_path, **kwargs)
    elif format == 'npy':
        np.save(file_path, data.strain.values, **kwargs)
    else:
        raise ValueError(f"Unsupported save format: {format}")


def _save_hdf5_das(data: xr.Dataset, 
                   file_path: Path,
                   compression: str = 'gzip',
                   **kwargs):
    """Save DAS data to HDF5 format"""
    
    with h5py.File(file_path, 'w') as f:
        # Save strain data
        f.create_dataset('strain', data=data.strain.values, 
                        compression=compression, **kwargs)
        
        # Save coordinates
        f.create_dataset('time', data=data.time.values, compression=compression)
        f.create_dataset('channel', data=data.channel.values, compression=compression)
        f.create_dataset('distance', data=data.distance.values, compression=compression)
        
        # Save attributes
        for key, value in data.attrs.items():
            try:
                f.attrs[key] = value
            except (TypeError, ValueError):
                # Convert to string if can't save directly
                f.attrs[key] = str(value)


def save_results(results: Dict[str, Any], 
                output_dir: Union[str, Path],
                format: str = 'hdf5',
                include_metadata: bool = True):
    """
    Save processing results to disk
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Processing results
    output_dir : str or Path
        Output directory
    format : str
        Output format ('hdf5', 'csv', 'json')
    include_metadata : bool
        Whether to save metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {output_dir}")
    
    # Save cross-correlation results
    if 'correlations' in results:
        if format == 'hdf5':
            cc_file = output_dir / 'cross_correlations.h5'
            _save_correlations_hdf5(results, cc_file)
        elif format == 'csv':
            _save_correlations_csv(results, output_dir)
        else:
            raise ValueError(f"Unsupported format for correlations: {format}")
    
    # Save metadata
    if include_metadata:
        metadata_file = output_dir / 'metadata.json'
        _save_metadata_json(results, metadata_file)
    
    logger.info("Results saved successfully")


def _save_correlations_hdf5(results: Dict, file_path: Path):
    """Save cross-correlations to HDF5"""
    with h5py.File(file_path, 'w') as f:
        # Save lag times
        if 'lag_times' in results:
            f.create_dataset('lag_times', data=results['lag_times'])
        
        # Save configuration
        if 'config' in results:
            config_grp = f.create_group('config')
            _save_dict_to_hdf5(results['config'], config_grp)
        
        # Save correlations
        correlations = results['correlations']
        corr_grp = f.create_group('correlations')
        
        for pair_id, pair_data in correlations.items():
            pair_grp = corr_grp.create_group(pair_id)
            
            # Save pair metadata
            pair = pair_data['pair']
            pair_grp.attrs['ch1'] = pair.ch1
            pair_grp.attrs['ch2'] = pair.ch2
            pair_grp.attrs['distance'] = pair.distance
            pair_grp.attrs['separation'] = pair.separation
            
            # Save correlation data
            for key, value in pair_data.items():
                if key != 'pair' and isinstance(value, np.ndarray):
                    pair_grp.create_dataset(key, data=value, compression='gzip')
                elif key != 'pair':
                    pair_grp.attrs[key] = value


def _save_dict_to_hdf5(d: Dict, group: h5py.Group):
    """Recursively save dictionary to HDF5 group"""
    for key, value in d.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            _save_dict_to_hdf5(value, subgroup)
        elif isinstance(value, (list, tuple)):
            group.attrs[key] = np.array(value)
        else:
            try:
                group.attrs[key] = value
            except (TypeError, ValueError):
                group.attrs[key] = str(value)


def _save_correlations_csv(results: Dict, output_dir: Path):
    """Save cross-correlations as CSV files"""
    correlations = results['correlations']
    
    # Save summary information
    summary_data = []
    for pair_id, pair_data in correlations.items():
        pair = pair_data['pair']
        summary_data.append({
            'pair_id': pair_id,
            'ch1': pair.ch1,
            'ch2': pair.ch2,
            'distance': pair.distance,
            'separation': pair.separation,
            'quality': pair_data.get('quality', np.nan),
            'n_windows': pair_data.get('n_windows', 1),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'correlation_summary.csv', index=False)
    
    # Save lag times
    if 'lag_times' in results:
        lag_df = pd.DataFrame({'lag_times': results['lag_times']})
        lag_df.to_csv(output_dir / 'lag_times.csv', index=False)
    
    # Save individual correlations if present
    correlations_dir = output_dir / 'correlations'
    correlations_dir.mkdir(exist_ok=True)
    
    for pair_id, pair_data in correlations.items():
        if 'correlation' in pair_data:
            # Single correlation
            corr_df = pd.DataFrame({
                'lag_times': results.get('lag_times', np.arange(len(pair_data['correlation']))),
                'correlation': pair_data['correlation']
            })
            corr_df.to_csv(correlations_dir / f'{pair_id}.csv', index=False)


def _save_metadata_json(results: Dict, file_path: Path):
    """Save metadata to JSON"""
    metadata = {}
    
    # Extract metadata, excluding large arrays
    for key, value in results.items():
        if key not in ['correlations', 'lag_times']:
            if isinstance(value, dict):
                metadata[key] = _dict_to_json_compatible(value)
            elif isinstance(value, np.ndarray):
                metadata[key] = f"Array shape: {value.shape}, dtype: {value.dtype}"
            else:
                metadata[key] = value
    
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def _dict_to_json_compatible(d: Dict) -> Dict:
    """Convert dictionary to JSON-compatible format"""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _dict_to_json_compatible(value)
        elif isinstance(value, np.ndarray):
            result[key] = f"Array shape: {value.shape}, dtype: {value.dtype}"
        elif isinstance(value, (np.integer, np.floating)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def create_synthetic_das_data(n_time: int = 10000,
                            n_channels: int = 1000,
                            sampling_rate: float = 1000.0,
                            channel_spacing: float = 1.0,
                            noise_level: float = 1.0,
                            signal_velocity: float = 3000.0,
                            source_location: float = 500.0) -> xr.Dataset:
    """
    Create synthetic DAS data for testing
    
    Parameters:
    -----------
    n_time : int
        Number of time samples
    n_channels : int
        Number of channels
    sampling_rate : float
        Sampling rate in Hz
    channel_spacing : float
        Channel spacing in meters
    noise_level : float
        Noise level (standard deviation)
    signal_velocity : float
        Signal velocity in m/s
    source_location : float
        Source location in meters
        
    Returns:
    --------
    xr.Dataset
        Synthetic DAS data
    """
    
    # Create coordinates
    time = np.arange(n_time) / sampling_rate
    channels = np.arange(n_channels)
    distances = channels * channel_spacing
    
    # Create synthetic signal
    # Simple pulse propagating along the array
    strain = np.random.normal(0, noise_level, (n_time, n_channels))
    
    # Add coherent signal
    pulse_time = 2.0  # seconds
    pulse_width = 0.1  # seconds
    pulse_samples = int(pulse_width * sampling_rate)
    
    for i, distance in enumerate(distances):
        travel_time = abs(distance - source_location) / signal_velocity
        arrival_sample = int((pulse_time + travel_time) * sampling_rate)
        
        if 0 <= arrival_sample < n_time - pulse_samples:
            # Add Ricker wavelet
            t_pulse = np.arange(pulse_samples) / sampling_rate
            f0 = 5.0  # Hz
            ricker = (1 - 2 * (np.pi * f0 * t_pulse)**2) * np.exp(-(np.pi * f0 * t_pulse)**2)
            amplitude = np.exp(-abs(distance - source_location) / 200.0)  # Geometric spreading
            
            end_sample = min(arrival_sample + pulse_samples, n_time)
            actual_samples = end_sample - arrival_sample
            strain[arrival_sample:end_sample, i] += amplitude * ricker[:actual_samples]
    
    # Create dataset
    dataset = xr.Dataset(
        {
            'strain': (['time', 'channel'], strain),
            'distance': (['channel'], distances),
        },
        coords={
            'time': time,
            'channel': channels,
        },
        attrs={
            'sampling_rate': sampling_rate,
            'channel_spacing': channel_spacing,
            'gauge_length': 10.0,
            'source_location': source_location,
            'signal_velocity': signal_velocity,
            'synthetic': True,
        }
    )
    
    logger.info(f"Created synthetic DAS data: {strain.shape} (time, channel)")
    return dataset