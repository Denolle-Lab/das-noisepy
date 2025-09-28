"""
Core DAS processing module

This module provides the main DAS data processing capabilities, adapted from 
DAS_module patterns and integrated with NoisePy ecosystem.
"""

import numpy as np
import xarray as xr
import h5py
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DASProcessor:
    """
    Main DAS data processor class
    
    This class handles DAS data preprocessing, quality control, and preparation
    for cross-correlation analysis using NoisePy-compatible methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DAS processor
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for DAS processing
        """
        self.config = config or self._default_config()
        self.data = None
        self.metadata = {}
        
    def _default_config(self) -> Dict:
        """Default configuration parameters"""
        return {
            'sampling_rate': 1000.0,  # Hz
            'channel_spacing': 1.0,   # m
            'gauge_length': 10.0,     # m
            'detrend': True,
            'taper': True,
            'taper_fraction': 0.05,
            'bandpass_filter': True,
            'freq_min': 0.01,         # Hz
            'freq_max': 10.0,         # Hz
            'decimate_factor': None,
            'time_chunk_size': 3600,  # seconds
            'quality_control': True,
            'snr_threshold': 2.0,
        }
    
    def load_data(self, data_path: Union[str, Path], format: str = 'hdf5') -> xr.Dataset:
        """
        Load DAS data from file
        
        Parameters:
        -----------
        data_path : str or Path
            Path to DAS data file
        format : str
            Data format ('hdf5', 'segy', 'tdms')
            
        Returns:
        --------
        xr.Dataset
            Loaded DAS data
        """
        data_path = Path(data_path)
        
        if format.lower() == 'hdf5':
            self.data = self._load_hdf5(data_path)
        elif format.lower() == 'segy':
            self.data = self._load_segy(data_path)
        elif format.lower() == 'tdms':
            self.data = self._load_tdms(data_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Loaded DAS data: {self.data.dims}")
        return self.data
    
    def _load_hdf5(self, file_path: Path) -> xr.Dataset:
        """Load DAS data from HDF5 format"""
        with h5py.File(file_path, 'r') as f:
            # Extract strain data
            strain = f['strain'][:]  # Shape: (time, channel)
            
            # Extract metadata
            sampling_rate = f.attrs.get('sampling_rate', self.config['sampling_rate'])
            channel_spacing = f.attrs.get('channel_spacing', self.config['channel_spacing'])
            gauge_length = f.attrs.get('gauge_length', self.config['gauge_length'])
            
            # Create time and channel coordinates
            n_time, n_channels = strain.shape
            time = np.arange(n_time) / sampling_rate
            channels = np.arange(n_channels)
            distance = channels * channel_spacing
            
            # Create xarray Dataset
            data = xr.Dataset(
                {
                    'strain': (['time', 'channel'], strain),
                    'distance': (['channel'], distance),
                },
                coords={
                    'time': time,
                    'channel': channels,
                },
                attrs={
                    'sampling_rate': sampling_rate,
                    'channel_spacing': channel_spacing,
                    'gauge_length': gauge_length,
                    'file_path': str(file_path),
                }
            )
            
        return data
    
    def _load_segy(self, file_path: Path) -> xr.Dataset:
        """Load DAS data from SEG-Y format"""
        # Placeholder for SEG-Y loading
        # Would use obspy or segyio for actual implementation
        raise NotImplementedError("SEG-Y loading not yet implemented")
    
    def _load_tdms(self, file_path: Path) -> xr.Dataset:
        """Load DAS data from TDMS format"""
        # Placeholder for TDMS loading
        # Would use nptdms package for actual implementation
        raise NotImplementedError("TDMS loading not yet implemented")
    
    def preprocess(self, data: Optional[xr.Dataset] = None) -> xr.Dataset:
        """
        Preprocess DAS data
        
        Parameters:
        -----------
        data : xr.Dataset, optional
            Input data. If None, uses self.data
            
        Returns:
        --------
        xr.Dataset
            Preprocessed data
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        logger.info("Starting DAS data preprocessing")
        
        # Detrend
        if self.config['detrend']:
            data = self._detrend(data)
            
        # Apply taper
        if self.config['taper']:
            data = self._taper(data)
            
        # Bandpass filter
        if self.config['bandpass_filter']:
            data = self._bandpass_filter(data)
            
        # Decimate
        if self.config['decimate_factor'] is not None:
            data = self._decimate(data)
            
        # Quality control
        if self.config['quality_control']:
            data = self._quality_control(data)
            
        logger.info("DAS preprocessing completed")
        return data
    
    def _detrend(self, data: xr.Dataset) -> xr.Dataset:
        """Remove linear trend from each channel"""
        from scipy.signal import detrend
        
        strain_detrended = detrend(data.strain.values, axis=0)
        data_out = data.copy()
        data_out['strain'].values = strain_detrended
        
        logger.debug("Applied detrending")
        return data_out
    
    def _taper(self, data: xr.Dataset) -> xr.Dataset:
        """Apply cosine taper to reduce edge effects"""
        from scipy.signal.windows import tukey
        
        n_time = len(data.time)
        taper_window = tukey(n_time, alpha=self.config['taper_fraction'])
        
        # Apply taper to each channel
        strain_tapered = data.strain.values * taper_window[:, np.newaxis]
        
        data_out = data.copy()
        data_out['strain'].values = strain_tapered
        
        logger.debug("Applied tapering")
        return data_out
    
    def _bandpass_filter(self, data: xr.Dataset) -> xr.Dataset:
        """Apply bandpass filter"""
        from scipy.signal import butter, filtfilt
        
        fs = data.attrs['sampling_rate']
        nyquist = fs / 2.0
        
        low = self.config['freq_min'] / nyquist
        high = self.config['freq_max'] / nyquist
        
        # Design filter
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter to each channel
        strain_filtered = np.zeros_like(data.strain.values)
        for i in range(data.strain.shape[1]):
            strain_filtered[:, i] = filtfilt(b, a, data.strain.values[:, i])
        
        data_out = data.copy()
        data_out['strain'].values = strain_filtered
        
        logger.debug(f"Applied bandpass filter: {self.config['freq_min']}-{self.config['freq_max']} Hz")
        return data_out
    
    def _decimate(self, data: xr.Dataset) -> xr.Dataset:
        """Decimate data to reduce sampling rate"""
        from scipy.signal import decimate
        
        factor = self.config['decimate_factor']
        
        # Decimate strain data
        strain_decimated = decimate(data.strain.values, factor, axis=0)
        
        # Update time coordinate
        new_fs = data.attrs['sampling_rate'] / factor
        n_time = strain_decimated.shape[0]
        new_time = np.arange(n_time) / new_fs
        
        # Create new dataset
        data_out = xr.Dataset(
            {
                'strain': (['time', 'channel'], strain_decimated),
                'distance': data.distance,
            },
            coords={
                'time': new_time,
                'channel': data.channel,
            },
            attrs=data.attrs.copy()
        )
        data_out.attrs['sampling_rate'] = new_fs
        
        logger.debug(f"Decimated by factor {factor}, new sampling rate: {new_fs} Hz")
        return data_out
    
    def _quality_control(self, data: xr.Dataset) -> xr.Dataset:
        """Apply quality control checks"""
        # Calculate signal-to-noise ratio for each channel
        strain = data.strain.values
        
        # Simple SNR estimation: signal power / noise power
        signal_power = np.var(strain, axis=0)
        # Use high-frequency content as noise proxy
        from scipy.signal import butter, filtfilt
        fs = data.attrs['sampling_rate']
        nyquist = fs / 2.0
        
        # High-pass filter for noise estimation
        b, a = butter(4, 5.0 / nyquist, btype='high')
        noise_strain = np.zeros_like(strain)
        for i in range(strain.shape[1]):
            noise_strain[:, i] = filtfilt(b, a, strain[:, i])
        
        noise_power = np.var(noise_strain, axis=0)
        snr = signal_power / (noise_power + 1e-12)  # Avoid division by zero
        
        # Mark channels with low SNR
        good_channels = snr > self.config['snr_threshold']
        
        data_out = data.copy()
        data_out.attrs['good_channels'] = good_channels
        data_out.attrs['snr'] = snr
        
        n_good = np.sum(good_channels)
        n_total = len(good_channels)
        logger.info(f"Quality control: {n_good}/{n_total} channels passed SNR threshold")
        
        return data_out
    
    def get_channel_info(self) -> Dict:
        """Get information about channels"""
        if self.data is None:
            raise ValueError("No data loaded")
            
        info = {
            'n_channels': len(self.data.channel),
            'channel_spacing': self.data.attrs.get('channel_spacing', self.config['channel_spacing']),
            'total_length': len(self.data.channel) * self.data.attrs.get('channel_spacing', self.config['channel_spacing']),
            'sampling_rate': self.data.attrs['sampling_rate'],
            'duration': len(self.data.time) / self.data.attrs['sampling_rate'],
        }
        
        if 'good_channels' in self.data.attrs:
            info['good_channels'] = np.sum(self.data.attrs['good_channels'])
            info['bad_channels'] = len(self.data.channel) - info['good_channels']
            
        return info