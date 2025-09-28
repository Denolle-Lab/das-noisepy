"""
Cross-correlation processing for DAS data

This module implements cross-correlation analysis adapted from Mt. Rainier 
notebook functionality, integrated with NoisePy ecosystem patterns.
"""

import numpy as np
import xarray as xr
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging
from scipy.signal import correlate, hilbert
from scipy.fft import fft, ifft, fftshift
import h5py

from ..core.channel_pairs import ChannelPair, ChannelPairSelector

logger = logging.getLogger(__name__)


class CrossCorrelator:
    """
    Cross-correlation processor for DAS data
    
    This class implements cross-correlation analysis with flexible channel pair
    selection, time-domain and frequency-domain processing options.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize cross-correlator
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters
        """
        self.config = config or self._default_config()
        self.results = {}
        
    def _default_config(self) -> Dict:
        """Default configuration parameters"""
        return {
            'method': 'fft',              # 'fft' or 'time_domain'
            'max_lag_time': 10.0,         # seconds
            'normalization': 'cross',     # 'cross', 'auto', or 'none'
            'whitening': True,
            'whitening_freqs': (0.1, 10.0),  # Hz
            'time_window_length': 3600,    # seconds
            'time_window_overlap': 0.5,    # fraction
            'save_individual_correlations': False,
            'stack_method': 'linear',      # 'linear', 'pws', 'robust'
            'quality_control': True,
            'cc_threshold': 0.01,          # minimum correlation coefficient
            'remove_response': False,
        }
    
    def compute_correlations(self,
                           data: xr.Dataset,
                           pairs: List[ChannelPair],
                           time_chunked: bool = True) -> Dict:
        """
        Compute cross-correlations for given channel pairs
        
        Parameters:
        -----------
        data : xr.Dataset
            DAS data
        pairs : List[ChannelPair]
            Channel pairs to correlate
        time_chunked : bool
            Whether to process in time chunks
            
        Returns:
        --------
        Dict
            Cross-correlation results
        """
        logger.info(f"Computing correlations for {len(pairs)} channel pairs")
        
        sampling_rate = data.attrs['sampling_rate']
        max_lag_samples = int(self.config['max_lag_time'] * sampling_rate)
        
        if time_chunked:
            return self._compute_correlations_chunked(data, pairs, max_lag_samples)
        else:
            return self._compute_correlations_whole(data, pairs, max_lag_samples)
    
    def _compute_correlations_chunked(self,
                                    data: xr.Dataset,
                                    pairs: List[ChannelPair],
                                    max_lag_samples: int) -> Dict:
        """Compute correlations in time chunks"""
        sampling_rate = data.attrs['sampling_rate']
        window_samples = int(self.config['time_window_length'] * sampling_rate)
        overlap_samples = int(window_samples * self.config['time_window_overlap'])
        step_samples = window_samples - overlap_samples
        
        n_time = len(data.time)
        n_windows = (n_time - overlap_samples) // step_samples
        
        logger.info(f"Processing {n_windows} time windows")
        
        # Initialize result arrays
        lag_times = np.arange(-max_lag_samples, max_lag_samples + 1) / sampling_rate
        
        correlations = {}
        for pair in pairs:
            pair_id = f"{pair.ch1:04d}_{pair.ch2:04d}"
            correlations[pair_id] = {
                'pair': pair,
                'correlations': np.zeros((n_windows, len(lag_times))),
                'times': np.zeros(n_windows),
                'quality': np.zeros(n_windows),
            }
        
        # Process each time window
        for i in range(n_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            
            if end_idx > n_time:
                break
                
            window_data = data.isel(time=slice(start_idx, end_idx))
            window_time = data.time[start_idx:end_idx].mean().values
            
            # Preprocess window
            if self.config['whitening']:
                window_data = self._spectral_whitening(window_data)
            
            # Compute correlations for all pairs in this window
            for pair in pairs:
                pair_id = f"{pair.ch1:04d}_{pair.ch2:04d}"
                
                trace1 = window_data.strain.isel(channel=pair.ch1).values
                trace2 = window_data.strain.isel(channel=pair.ch2).values
                
                # Compute cross-correlation
                if self.config['correlation_method'] == 'fft':
                    cc = self._fft_correlate(trace1, trace2, max_lag_samples)
                else:
                    cc = self._time_domain_correlate(trace1, trace2, max_lag_samples)
                
                # Apply normalization
                if self.config['normalization'] != 'none':
                    cc = self._normalize_correlation(cc, trace1, trace2)
                
                # Store results
                correlations[pair_id]['correlations'][i, :] = cc
                correlations[pair_id]['times'][i] = window_time
                
                # Compute quality metric
                max_cc = np.max(np.abs(cc))
                correlations[pair_id]['quality'][i] = max_cc
        
        # Add metadata
        results = {
            'correlations': correlations,
            'lag_times': lag_times,
            'config': self.config.copy(),
            'n_windows': n_windows,
            'sampling_rate': sampling_rate,
        }
        
        logger.info("Cross-correlation computation completed")
        return results
    
    def _compute_correlations_whole(self,
                                  data: xr.Dataset,
                                  pairs: List[ChannelPair],
                                  max_lag_samples: int) -> Dict:
        """Compute correlations for entire time series"""
        sampling_rate = data.attrs['sampling_rate']
        lag_times = np.arange(-max_lag_samples, max_lag_samples + 1) / sampling_rate
        
        # Preprocess data
        if self.config['whitening']:
            data = self._spectral_whitening(data)
        
        correlations = {}
        
        for pair in pairs:
            pair_id = f"{pair.ch1:04d}_{pair.ch2:04d}"
            
            trace1 = data.strain.isel(channel=pair.ch1).values
            trace2 = data.strain.isel(channel=pair.ch2).values
            
            # Compute cross-correlation
            if self.config['method'] == 'fft':
                cc = self._fft_correlate(trace1, trace2, max_lag_samples)
            else:
                cc = self._time_domain_correlate(trace1, trace2, max_lag_samples)
            
            # Apply normalization
            if self.config['normalization'] != 'none':
                cc = self._normalize_correlation(cc, trace1, trace2)
            
            # Store results
            correlations[pair_id] = {
                'pair': pair,
                'correlation': cc,
                'quality': np.max(np.abs(cc)),
            }
        
        results = {
            'correlations': correlations,
            'lag_times': lag_times,
            'config': self.config.copy(),
            'sampling_rate': sampling_rate,
        }
        
        return results
    
    def _fft_correlate(self, trace1: np.ndarray, trace2: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute cross-correlation using FFT"""
        n = len(trace1)
        
        # Zero-pad to avoid circular correlation
        nfft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        
        # FFT of both traces
        f1 = fft(trace1, nfft)
        f2 = fft(trace2, nfft)
        
        # Cross-correlation in frequency domain
        cc_fft = f1 * np.conj(f2)
        cc_full = np.real(ifft(cc_fft))
        
        # Rearrange to center zero lag
        cc_full = fftshift(cc_full)
        
        # Extract desired lag range
        center = len(cc_full) // 2
        start_idx = center - max_lag
        end_idx = center + max_lag + 1
        
        cc = cc_full[start_idx:end_idx]
        
        return cc
    
    def _time_domain_correlate(self, trace1: np.ndarray, trace2: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute cross-correlation in time domain"""
        cc_full = correlate(trace1, trace2, mode='full')
        
        # Extract desired lag range
        center = len(cc_full) // 2
        start_idx = center - max_lag
        end_idx = center + max_lag + 1
        
        cc = cc_full[start_idx:end_idx]
        
        return cc
    
    def _normalize_correlation(self, cc: np.ndarray, trace1: np.ndarray, trace2: np.ndarray) -> np.ndarray:
        """Apply normalization to cross-correlation"""
        if self.config['normalization'] == 'cross':
            # Cross-correlation normalization
            norm = np.sqrt(np.sum(trace1**2) * np.sum(trace2**2))
            if norm > 0:
                cc = cc / norm
        elif self.config['normalization'] == 'auto':
            # Auto-correlation normalization
            auto1 = np.sum(trace1**2)
            auto2 = np.sum(trace2**2)
            norm = np.sqrt(auto1 * auto2)
            if norm > 0:
                cc = cc / norm
        
        return cc
    
    def _spectral_whitening(self, data: xr.Dataset) -> xr.Dataset:
        """Apply spectral whitening to data"""
        sampling_rate = data.attrs['sampling_rate']
        freq_min, freq_max = self.config['whitening_freqs']
        
        # Convert to frequency limits for whitening
        n = len(data.time)
        freqs = np.fft.fftfreq(n, 1.0/sampling_rate)
        freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        
        # Apply whitening to each channel
        whitened_strain = np.zeros_like(data.strain.values)
        
        for i in range(data.strain.shape[1]):
            trace = data.strain.isel(channel=i).values
            
            # FFT
            trace_fft = fft(trace)
            
            # Spectral whitening
            amplitude = np.abs(trace_fft)
            amplitude[amplitude == 0] = 1e-12  # Avoid division by zero
            
            # Apply whitening only in specified frequency band
            whitened_fft = trace_fft.copy()
            whitened_fft[freq_mask] = trace_fft[freq_mask] / amplitude[freq_mask]
            
            # IFFT
            whitened_strain[:, i] = np.real(ifft(whitened_fft))
        
        # Create new dataset
        data_whitened = data.copy()
        data_whitened['strain'].values = whitened_strain
        
        return data_whitened
    
    def stack_correlations(self, results: Dict, method: str = None) -> Dict:
        """
        Stack cross-correlations across time windows
        
        Parameters:
        -----------
        results : Dict
            Cross-correlation results from compute_correlations
        method : str, optional
            Stacking method ('linear', 'pws', 'robust')
            
        Returns:
        --------
        Dict
            Stacked correlation results
        """
        if method is None:
            method = self.config['stack_method']
            
        if 'n_windows' not in results:
            logger.warning("Input appears to be single-window results, no stacking needed")
            return results
        
        logger.info(f"Stacking correlations using {method} method")
        
        correlations = results['correlations']
        stacked = {}
        
        for pair_id, pair_data in correlations.items():
            cc_matrix = pair_data['correlations']
            quality = pair_data['quality']
            
            # Apply quality control
            if self.config['quality_control']:
                good_windows = quality > self.config['cc_threshold']
                if np.sum(good_windows) == 0:
                    logger.warning(f"No good windows for pair {pair_id}")
                    continue
                cc_matrix = cc_matrix[good_windows, :]
            
            # Stack correlations
            if method == 'linear':
                stacked_cc = np.mean(cc_matrix, axis=0)
            elif method == 'pws':
                stacked_cc = self._phase_weighted_stack(cc_matrix)
            elif method == 'robust':
                stacked_cc = np.median(cc_matrix, axis=0)
            else:
                raise ValueError(f"Unknown stacking method: {method}")
            
            stacked[pair_id] = {
                'pair': pair_data['pair'],
                'correlation': stacked_cc,
                'n_windows': cc_matrix.shape[0],
                'quality': np.mean(quality),
            }
        
        # Create stacked results
        stacked_results = {
            'correlations': stacked,
            'lag_times': results['lag_times'],
            'config': results['config'].copy(),
            'sampling_rate': results['sampling_rate'],
            'stack_method': method,
        }
        
        logger.info(f"Stacked {len(stacked)} correlation pairs")
        return stacked_results
    
    def _phase_weighted_stack(self, cc_matrix: np.ndarray) -> np.ndarray:
        """Apply phase-weighted stacking"""
        # Convert to complex representation
        analytic_cc = hilbert(cc_matrix, axis=1)
        
        # Compute phase weights
        phases = np.angle(analytic_cc)
        phase_coherence = np.abs(np.mean(np.exp(1j * phases), axis=0))
        
        # Apply weights to linear stack
        linear_stack = np.mean(cc_matrix, axis=0)
        pws_stack = linear_stack * phase_coherence
        
        return pws_stack
    
    def save_results(self, results: Dict, filepath: Union[str, Path]):
        """
        Save cross-correlation results to HDF5 file
        
        Parameters:
        -----------
        results : Dict
            Cross-correlation results
        filepath : str or Path
            Output file path
        """
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            f.attrs['sampling_rate'] = results['sampling_rate']
            f.attrs['stack_method'] = results.get('stack_method', 'none')
            
            # Save configuration
            config_grp = f.create_group('config')
            for key, value in results['config'].items():
                config_grp.attrs[key] = value
            
            # Save lag times
            f.create_dataset('lag_times', data=results['lag_times'])
            
            # Save correlations
            correlations_grp = f.create_group('correlations')
            
            for pair_id, pair_data in results['correlations'].items():
                pair_grp = correlations_grp.create_group(pair_id)
                
                # Save pair info
                pair_grp.attrs['ch1'] = pair_data['pair'].ch1
                pair_grp.attrs['ch2'] = pair_data['pair'].ch2
                pair_grp.attrs['distance'] = pair_data['pair'].distance
                pair_grp.attrs['separation'] = pair_data['pair'].separation
                
                # Save correlation data
                if 'correlation' in pair_data:
                    # Stacked result
                    pair_grp.create_dataset('correlation', data=pair_data['correlation'])
                    pair_grp.attrs['quality'] = pair_data['quality']
                    pair_grp.attrs['n_windows'] = pair_data.get('n_windows', 1)
                else:
                    # Time-windowed result
                    pair_grp.create_dataset('correlations', data=pair_data['correlations'])
                    pair_grp.create_dataset('times', data=pair_data['times'])
                    pair_grp.create_dataset('quality', data=pair_data['quality'])
        
        logger.info(f"Saved cross-correlation results to {filepath}")
    
    def load_results(self, filepath: Union[str, Path]) -> Dict:
        """
        Load cross-correlation results from HDF5 file
        
        Parameters:
        -----------
        filepath : str or Path
            Input file path
            
        Returns:
        --------
        Dict
            Cross-correlation results
        """
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'r') as f:
            # Load metadata
            sampling_rate = f.attrs['sampling_rate']
            stack_method = f.attrs.get('stack_method', 'none')
            
            # Load configuration
            config = {}
            for key, value in f['config'].attrs.items():
                config[key] = value
            
            # Load lag times
            lag_times = f['lag_times'][:]
            
            # Load correlations
            correlations = {}
            correlations_grp = f['correlations']
            
            for pair_id in correlations_grp.keys():
                pair_grp = correlations_grp[pair_id]
                
                # Load pair info
                pair = ChannelPair(
                    ch1=pair_grp.attrs['ch1'],
                    ch2=pair_grp.attrs['ch2'],
                    distance=pair_grp.attrs['distance'],
                    separation=pair_grp.attrs['separation']
                )
                
                # Load correlation data
                if 'correlation' in pair_grp:
                    # Stacked result
                    correlations[pair_id] = {
                        'pair': pair,
                        'correlation': pair_grp['correlation'][:],
                        'quality': pair_grp.attrs['quality'],
                        'n_windows': pair_grp.attrs['n_windows'],
                    }
                else:
                    # Time-windowed result
                    correlations[pair_id] = {
                        'pair': pair,
                        'correlations': pair_grp['correlations'][:],
                        'times': pair_grp['times'][:],
                        'quality': pair_grp['quality'][:],
                    }
        
        # Construct results dictionary
        results = {
            'correlations': correlations,
            'lag_times': lag_times,
            'config': config,
            'sampling_rate': sampling_rate,
        }
        
        if stack_method != 'none':
            results['stack_method'] = stack_method
        
        logger.info(f"Loaded cross-correlation results from {filepath}")
        return results