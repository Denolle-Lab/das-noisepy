"""
Basic tests for DAS-NoisePy functionality
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from das_noisepy import (
    DASProcessor,
    ChannelPairSelector,
    CrossCorrelator,
    create_synthetic_das_data,
    get_default_config
)


@pytest.fixture
def synthetic_data():
    """Create synthetic DAS data for testing"""
    return create_synthetic_das_data(
        n_time=1000,
        n_channels=100,
        sampling_rate=100.0,
        channel_spacing=1.0
    )


@pytest.fixture
def config():
    """Get default configuration"""
    return get_default_config()


def test_synthetic_data_creation():
    """Test synthetic data creation"""
    data = create_synthetic_das_data(n_time=500, n_channels=50)
    
    assert data.strain.shape == (500, 50)
    assert len(data.time) == 500
    assert len(data.channel) == 50
    assert len(data.distance) == 50
    assert 'sampling_rate' in data.attrs
    assert data.attrs['synthetic'] is True


def test_das_processor_initialization(config):
    """Test DAS processor initialization"""
    processor = DASProcessor(config['preprocessing'])
    
    assert processor.config is not None
    assert processor.config['detrend'] is True
    assert processor.config['bandpass_filter'] is True


def test_das_preprocessing(synthetic_data, config):
    """Test DAS data preprocessing"""
    processor = DASProcessor(config['preprocessing'])
    processor.data = synthetic_data
    
    processed_data = processor.preprocess()
    
    assert processed_data.strain.shape == synthetic_data.strain.shape
    assert 'good_channels' in processed_data.attrs
    assert 'snr' in processed_data.attrs
    
    # Check that some channels are marked as good
    assert np.sum(processed_data.attrs['good_channels']) > 0


def test_channel_pair_selector():
    """Test channel pair selection"""
    selector = ChannelPairSelector(channel_spacing=1.0)
    
    # Test distance-based selection
    pairs = selector.select_pairs_by_distance(
        n_channels=50,
        min_separation=5.0,
        max_separation=20.0
    )
    
    assert len(pairs) > 0
    
    # Check pair properties
    for pair in pairs[:5]:  # Check first few pairs
        assert pair.ch1 < pair.ch2  # Ordered pairs
        assert 5.0 <= pair.separation <= 20.0  # Within specified range


def test_quality_based_pair_selection():
    """Test quality-based pair selection"""
    selector = ChannelPairSelector(channel_spacing=1.0)
    
    # Create mock SNR data
    snr = np.random.uniform(1.0, 10.0, 50)
    
    pairs = selector.select_pairs_by_quality(
        n_channels=50,
        snr=snr,
        min_snr=3.0,
        min_separation=5.0,
        max_separation=20.0
    )
    
    assert len(pairs) > 0
    
    # Check that selected channels have good SNR
    for pair in pairs[:5]:
        assert snr[pair.ch1] >= 3.0
        assert snr[pair.ch2] >= 3.0


def test_adaptive_pair_selection():
    """Test adaptive pair selection"""
    selector = ChannelPairSelector(channel_spacing=1.0)
    
    # Create mock SNR data with some variation
    snr = 3.0 + np.random.uniform(0, 5.0, 50)
    
    pairs = selector.select_pairs_adaptive(
        n_channels=50,
        snr=snr,
        target_pairs_per_distance=5,
        distance_bins=5
    )
    
    assert len(pairs) > 0
    
    # Should have reasonable distribution across distances
    separations = [pair.separation for pair in pairs]
    assert np.std(separations) > 0  # Some variation in separations


def test_cross_correlator_initialization(config):
    """Test cross-correlator initialization"""
    correlator = CrossCorrelator(config['cross_correlation'])
    
    assert correlator.config is not None
    assert correlator.config['method'] == 'fft'


def test_cross_correlation_computation(synthetic_data, config):
    """Test cross-correlation computation"""
    # Preprocess data
    processor = DASProcessor(config['preprocessing'])
    processor.data = synthetic_data
    processed_data = processor.preprocess()
    
    # Select a few pairs
    selector = ChannelPairSelector(channel_spacing=1.0)
    pairs = selector.select_pairs_by_distance(
        n_channels=len(processed_data.channel),
        min_separation=5.0,
        max_separation=15.0,
        max_pairs_per_separation=3
    )[:5]  # Limit to 5 pairs for quick test
    
    # Compute correlations
    correlator = CrossCorrelator(config['cross_correlation'])
    
    results = correlator.compute_correlations(
        data=processed_data,
        pairs=pairs,
        time_chunked=False
    )
    
    assert 'correlations' in results
    assert 'lag_times' in results
    assert len(results['correlations']) == len(pairs)
    
    # Check correlation properties
    for pair_id, pair_data in results['correlations'].items():
        assert 'correlation' in pair_data
        assert 'pair' in pair_data
        assert 'quality' in pair_data
        assert len(pair_data['correlation']) == len(results['lag_times'])


def test_save_load_results():
    """Test saving and loading results"""
    # Create minimal test data
    from das_noisepy.core.channel_pairs import ChannelPair
    
    # Mock results
    lag_times = np.linspace(-1, 1, 21)
    pair = ChannelPair(ch1=10, ch2=20, distance=15.0, separation=10.0)
    correlation = np.random.normal(0, 0.1, len(lag_times))
    
    results = {
        'correlations': {
            '0010_0020': {
                'pair': pair,
                'correlation': correlation,
                'quality': 0.5
            }
        },
        'lag_times': lag_times,
        'config': get_default_config()['cross_correlation'],
        'sampling_rate': 100.0
    }
    
    # Save and load
    correlator = CrossCorrelator()
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_file = Path(f.name)
    
    try:
        correlator.save_results(results, temp_file)
        loaded_results = correlator.load_results(temp_file)
        
        assert 'correlations' in loaded_results
        assert '0010_0020' in loaded_results['correlations']
        assert np.allclose(loaded_results['lag_times'], lag_times)
        
    finally:
        temp_file.unlink()  # Clean up


def test_config_validation():
    """Test configuration validation"""
    from das_noisepy.utils.config import validate_config
    
    # Test valid config
    config = get_default_config()
    validated = validate_config(config)
    
    assert 'das' in validated
    assert 'preprocessing' in validated
    assert validated['das']['sampling_rate'] > 0
    
    # Test invalid config
    invalid_config = {'das': {'sampling_rate': -1000}}
    
    with pytest.raises(ValueError, match="Sampling rate must be positive"):
        validate_config(invalid_config)


def test_pair_dataframe_conversion():
    """Test conversion of pairs to DataFrame"""
    selector = ChannelPairSelector(channel_spacing=2.0)
    
    pairs = selector.select_pairs_by_distance(
        n_channels=20,
        min_separation=4.0,
        max_separation=10.0
    )
    
    df = selector.get_pairs_dataframe()
    
    assert len(df) == len(pairs)
    assert 'ch1' in df.columns
    assert 'ch2' in df.columns
    assert 'distance' in df.columns
    assert 'separation' in df.columns
    assert 'pair_id' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])