"""
Flexible channel pair selection for DAS cross-correlation

This module provides enhanced flexibility for selecting channel pairs for cross-correlation
analysis, addressing the limitation mentioned in the original NoisePy implementation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Callable
import logging
from itertools import combinations
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChannelPair:
    """
    Represents a channel pair for cross-correlation
    """
    ch1: int
    ch2: int
    distance: float
    separation: float
    
    def __post_init__(self):
        # Ensure ch1 <= ch2 for consistency
        if self.ch1 > self.ch2:
            self.ch1, self.ch2 = self.ch2, self.ch1


class ChannelPairSelector:
    """
    Flexible channel pair selection for DAS cross-correlation analysis
    
    This class provides various strategies for selecting channel pairs,
    including distance-based, quality-based, and custom selection criteria.
    """
    
    def __init__(self, 
                 channel_spacing: float = 1.0,
                 quality_weights: Optional[np.ndarray] = None):
        """
        Initialize channel pair selector
        
        Parameters:
        -----------
        channel_spacing : float
            Spacing between channels in meters
        quality_weights : np.ndarray, optional
            Quality weights for each channel (0-1)
        """
        self.channel_spacing = channel_spacing
        self.quality_weights = quality_weights
        self.pairs = []
        
    def select_pairs_by_distance(self, 
                                n_channels: int,
                                min_separation: float = 0.0,
                                max_separation: float = np.inf,
                                step_size: float = None,
                                max_pairs_per_separation: int = None) -> List[ChannelPair]:
        """
        Select channel pairs based on separation distance
        
        Parameters:
        -----------
        n_channels : int
            Total number of channels
        min_separation : float
            Minimum separation distance in meters
        max_separation : float  
            Maximum separation distance in meters
        step_size : float, optional
            If provided, only select pairs at specific separation steps
        max_pairs_per_separation : int, optional
            Maximum number of pairs per separation distance
            
        Returns:
        --------
        List[ChannelPair]
            Selected channel pairs
        """
        pairs = []
        
        # Convert distances to channel indices
        min_ch_sep = int(np.ceil(min_separation / self.channel_spacing))
        max_ch_sep = int(np.floor(max_separation / self.channel_spacing))
        
        if step_size is not None:
            step_ch = int(np.round(step_size / self.channel_spacing))
            separations = range(min_ch_sep, max_ch_sep + 1, step_ch)
        else:
            separations = range(min_ch_sep, max_ch_sep + 1)
        
        for sep in separations:
            if sep == 0:
                continue  # Skip zero separation (auto-correlation)
                
            # Find all pairs with this separation
            separation_pairs = []
            for ch1 in range(n_channels - sep):
                ch2 = ch1 + sep
                if ch2 < n_channels:
                    distance1 = ch1 * self.channel_spacing
                    distance2 = ch2 * self.channel_spacing
                    avg_distance = (distance1 + distance2) / 2
                    separation_dist = sep * self.channel_spacing
                    
                    pair = ChannelPair(ch1, ch2, avg_distance, separation_dist)
                    separation_pairs.append(pair)
            
            # Limit pairs per separation if requested
            if max_pairs_per_separation is not None and len(separation_pairs) > max_pairs_per_separation:
                # Sample uniformly across the array
                indices = np.linspace(0, len(separation_pairs)-1, max_pairs_per_separation, dtype=int)
                separation_pairs = [separation_pairs[i] for i in indices]
            
            pairs.extend(separation_pairs)
        
        logger.info(f"Selected {len(pairs)} pairs by distance criteria")
        self.pairs = pairs
        return pairs
    
    def select_pairs_by_quality(self,
                               n_channels: int,
                               snr: np.ndarray,
                               min_snr: float = 2.0,
                               min_separation: float = 0.0,
                               max_separation: float = np.inf) -> List[ChannelPair]:
        """
        Select channel pairs based on signal quality
        
        Parameters:
        -----------
        n_channels : int
            Total number of channels
        snr : np.ndarray
            Signal-to-noise ratio for each channel
        min_snr : float
            Minimum SNR threshold for both channels in pair
        min_separation : float
            Minimum separation distance
        max_separation : float
            Maximum separation distance
            
        Returns:
        --------
        List[ChannelPair]
            Selected channel pairs
        """
        # First get good channels
        good_channels = np.where(snr >= min_snr)[0]
        logger.info(f"Found {len(good_channels)} channels with SNR >= {min_snr}")
        
        pairs = []
        min_ch_sep = int(np.ceil(min_separation / self.channel_spacing))
        max_ch_sep = int(np.floor(max_separation / self.channel_spacing))
        
        # Generate all combinations of good channels
        for ch1, ch2 in combinations(good_channels, 2):
            separation = abs(ch2 - ch1)
            
            if min_ch_sep <= separation <= max_ch_sep:
                distance1 = ch1 * self.channel_spacing
                distance2 = ch2 * self.channel_spacing
                avg_distance = (distance1 + distance2) / 2
                separation_dist = separation * self.channel_spacing
                
                pair = ChannelPair(ch1, ch2, avg_distance, separation_dist)
                pairs.append(pair)
        
        logger.info(f"Selected {len(pairs)} pairs by quality criteria")
        self.pairs = pairs
        return pairs
    
    def select_pairs_custom(self,
                           n_channels: int,
                           selection_func: Callable[[int, int, Dict], bool],
                           metadata: Dict = None) -> List[ChannelPair]:
        """
        Select channel pairs using custom selection function
        
        Parameters:
        -----------
        n_channels : int
            Total number of channels
        selection_func : Callable
            Function that takes (ch1, ch2, metadata) and returns bool
        metadata : Dict
            Additional metadata for selection function
            
        Returns:
        --------
        List[ChannelPair]
            Selected channel pairs
        """
        pairs = []
        metadata = metadata or {}
        
        for ch1 in range(n_channels):
            for ch2 in range(ch1 + 1, n_channels):
                if selection_func(ch1, ch2, metadata):
                    distance1 = ch1 * self.channel_spacing
                    distance2 = ch2 * self.channel_spacing
                    avg_distance = (distance1 + distance2) / 2
                    separation = abs(ch2 - ch1) * self.channel_spacing
                    
                    pair = ChannelPair(ch1, ch2, avg_distance, separation)
                    pairs.append(pair)
        
        logger.info(f"Selected {len(pairs)} pairs by custom criteria")
        self.pairs = pairs
        return pairs
    
    def select_pairs_regular_grid(self,
                                 n_channels: int,
                                 separation_distances: List[float],
                                 spacing_between_pairs: int = 1) -> List[ChannelPair]:
        """
        Select pairs on a regular grid of separations and positions
        
        Parameters:
        -----------
        n_channels : int
            Total number of channels
        separation_distances : List[float]
            List of separation distances to include
        spacing_between_pairs : int
            Spacing between starting positions of pairs
            
        Returns:
        --------
        List[ChannelPair]
            Selected channel pairs
        """
        pairs = []
        
        for sep_dist in separation_distances:
            sep_ch = int(np.round(sep_dist / self.channel_spacing))
            if sep_ch == 0:
                continue
                
            for ch1 in range(0, n_channels - sep_ch, spacing_between_pairs):
                ch2 = ch1 + sep_ch
                if ch2 < n_channels:
                    distance1 = ch1 * self.channel_spacing
                    distance2 = ch2 * self.channel_spacing
                    avg_distance = (distance1 + distance2) / 2
                    
                    pair = ChannelPair(ch1, ch2, avg_distance, sep_dist)
                    pairs.append(pair)
        
        logger.info(f"Selected {len(pairs)} pairs on regular grid")
        self.pairs = pairs
        return pairs
    
    def select_pairs_adaptive(self,
                             n_channels: int,
                             snr: np.ndarray,
                             target_pairs_per_distance: int = 50,
                             distance_bins: int = 20) -> List[ChannelPair]:
        """
        Adaptively select pairs to achieve uniform sampling across distances
        while prioritizing high-quality channels
        
        Parameters:
        -----------
        n_channels : int
            Total number of channels
        snr : np.ndarray
            Signal-to-noise ratio for each channel
        target_pairs_per_distance : int
            Target number of pairs per distance bin
        distance_bins : int
            Number of distance bins
            
        Returns:
        --------
        List[ChannelPair]
            Selected channel pairs
        """
        # Create distance bins
        max_separation = (n_channels - 1) * self.channel_spacing
        distance_edges = np.linspace(0, max_separation, distance_bins + 1)
        
        pairs = []
        
        for i in range(distance_bins):
            min_dist = distance_edges[i]
            max_dist = distance_edges[i + 1]
            
            if min_dist == 0:
                min_dist = self.channel_spacing  # Skip zero separation
            
            # Find all possible pairs in this distance range
            bin_pairs = []
            
            min_ch_sep = int(np.ceil(min_dist / self.channel_spacing))
            max_ch_sep = int(np.floor(max_dist / self.channel_spacing))
            
            for ch1 in range(n_channels):
                for sep in range(min_ch_sep, max_ch_sep + 1):
                    ch2 = ch1 + sep
                    if ch2 < n_channels:
                        # Calculate combined quality score
                        quality_score = (snr[ch1] + snr[ch2]) / 2
                        separation_dist = sep * self.channel_spacing
                        
                        if min_dist <= separation_dist <= max_dist:
                            distance1 = ch1 * self.channel_spacing
                            distance2 = ch2 * self.channel_spacing
                            avg_distance = (distance1 + distance2) / 2
                            
                            pair = ChannelPair(ch1, ch2, avg_distance, separation_dist)
                            bin_pairs.append((pair, quality_score))
            
            # Sort by quality and select top pairs
            bin_pairs.sort(key=lambda x: x[1], reverse=True)
            selected_pairs = [pair for pair, _ in bin_pairs[:target_pairs_per_distance]]
            pairs.extend(selected_pairs)
        
        logger.info(f"Selected {len(pairs)} pairs adaptively")
        self.pairs = pairs
        return pairs
    
    def get_pairs_dataframe(self) -> pd.DataFrame:
        """
        Convert selected pairs to pandas DataFrame
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with pair information
        """
        if not self.pairs:
            return pd.DataFrame()
        
        data = []
        for pair in self.pairs:
            data.append({
                'ch1': pair.ch1,
                'ch2': pair.ch2,
                'distance': pair.distance,
                'separation': pair.separation,
                'pair_id': f"{pair.ch1:04d}_{pair.ch2:04d}",
            })
        
        return pd.DataFrame(data)
    
    def save_pairs(self, filepath: str):
        """Save selected pairs to file"""
        df = self.get_pairs_dataframe()
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.h5'):
            df.to_hdf(filepath, key='pairs', mode='w')
        else:
            raise ValueError("Unsupported file format. Use .csv or .h5")
        
        logger.info(f"Saved {len(self.pairs)} pairs to {filepath}")
    
    def load_pairs(self, filepath: str) -> List[ChannelPair]:
        """Load pairs from file"""
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.h5'):
            df = pd.read_hdf(filepath, key='pairs')
        else:
            raise ValueError("Unsupported file format. Use .csv or .h5")
        
        pairs = []
        for _, row in df.iterrows():
            pair = ChannelPair(
                ch1=int(row['ch1']),
                ch2=int(row['ch2']),
                distance=row['distance'],
                separation=row['separation']
            )
            pairs.append(pair)
        
        self.pairs = pairs
        logger.info(f"Loaded {len(self.pairs)} pairs from {filepath}")
        return pairs
    
    def plot_pair_distribution(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot the distribution of selected channel pairs
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size (width, height)
        """
        import matplotlib.pyplot as plt
        
        if not self.pairs:
            logger.warning("No pairs selected. Use a selection method first.")
            return
        
        df = self.get_pairs_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Separation distance histogram
        axes[0, 0].hist(df['separation'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Separation Distance (m)')
        axes[0, 0].set_ylabel('Number of Pairs')
        axes[0, 0].set_title('Distribution of Pair Separations')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Position along array
        axes[0, 1].hist(df['distance'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Average Position (m)')
        axes[0, 1].set_ylabel('Number of Pairs')
        axes[0, 1].set_title('Distribution of Pair Positions')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2D histogram: position vs separation
        h = axes[1, 0].hist2d(df['distance'], df['separation'], bins=30, cmap='Blues')
        axes[1, 0].set_xlabel('Average Position (m)')
        axes[1, 0].set_ylabel('Separation Distance (m)')
        axes[1, 0].set_title('Pairs: Position vs Separation')
        plt.colorbar(h[3], ax=axes[1, 0], label='Number of Pairs')
        
        # Pair connectivity plot
        max_ch = max(max(df['ch1']), max(df['ch2']))
        connectivity = np.zeros((max_ch + 1, max_ch + 1))
        for _, row in df.iterrows():
            connectivity[int(row['ch1']), int(row['ch2'])] = 1
            connectivity[int(row['ch2']), int(row['ch1'])] = 1
        
        im = axes[1, 1].imshow(connectivity, cmap='Blues', aspect='auto')
        axes[1, 1].set_xlabel('Channel Index')
        axes[1, 1].set_ylabel('Channel Index')
        axes[1, 1].set_title('Pair Connectivity Matrix')
        
        plt.tight_layout()
        return fig