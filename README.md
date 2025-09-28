# DAS-NoisePy

A comprehensive toolkit for cross-correlation analysis of Distributed Acoustic Sensing (DAS) data, integrating the best features of NoisePy with enhanced flexibility for DAS-specific processing workflows.

## Key Features

🎯 **Flexible Channel Pair Selection** - Multiple strategies for optimal channel pair selection, addressing limitations in original NoisePy
📊 **DAS-Specific Processing** - Tailored preprocessing pipeline for DAS data characteristics  
🔄 **NoisePy Integration** - Maintains compatibility with NoisePy ecosystem and workflows
📓 **Mt. Rainier Workflow** - Includes advanced cross-correlation notebook inspired by Mt. Rainier DAS analysis
⚡ **High Performance** - Optimized for large-scale DAS datasets with parallel processing support
🛠️ **Command Line Tools** - Easy-to-use CLI for automated processing workflows

## Quick Start

### Installation

```bash
git clone https://github.com/Denolle-Lab/das-noisepy.git
cd das-noisepy
pip install -e .
```

### Basic Usage

```python
from das_noisepy import (
    DASProcessor, 
    ChannelPairSelector, 
    CrossCorrelator,
    create_synthetic_das_data
)

# Create example data
data = create_synthetic_das_data(n_channels=1000, n_time=60000)

# Preprocess
processor = DASProcessor()
processed_data = processor.preprocess(data)

# Select channel pairs (flexible strategies!)
pair_selector = ChannelPairSelector(channel_spacing=2.0)
pairs = pair_selector.select_pairs_adaptive(
    n_channels=len(data.channel),
    snr=processed_data.attrs['snr']
)

# Compute cross-correlations
correlator = CrossCorrelator()
results = correlator.compute_correlations(processed_data, pairs)
```

### Command Line Interface

```bash
# Create default configuration
das-noisepy create-config -o config.yaml

# Process DAS data
das-noisepy process data.h5 -c config.yaml -o ./results

# Generate plots
das-noisepy plot ./results/cross_correlations.h5 -o ./plots

# Validate data and config
das-noisepy validate data.h5 -c config.yaml
```

## Channel Pair Selection Strategies

One of the key improvements in DAS-NoisePy is flexible channel pair selection:

### 1. **Distance-Based Selection**
```python
pairs = pair_selector.select_pairs_by_distance(
    n_channels=n_channels,
    min_separation=50.0,      # Minimum separation in meters
    max_separation=500.0,     # Maximum separation in meters
    step_size=25.0,          # Separation step size
    max_pairs_per_separation=30  # Limit pairs per distance
)
```

### 2. **Quality-Based Selection**
```python
pairs = pair_selector.select_pairs_by_quality(
    n_channels=n_channels,
    snr=channel_snr,         # Signal-to-noise ratio per channel
    min_snr=3.0,            # Minimum SNR threshold
    min_separation=50.0,
    max_separation=500.0
)
```

### 3. **Adaptive Selection** (Recommended)
```python
pairs = pair_selector.select_pairs_adaptive(
    n_channels=n_channels,
    snr=channel_snr,
    target_pairs_per_distance=20,  # Uniform spatial sampling
    distance_bins=15              # Number of distance bins
)
```

### 4. **Custom Selection**
```python
def custom_criteria(ch1, ch2, metadata):
    # Your custom selection logic
    return (abs(ch2 - ch1) >= 25) and (metadata['quality'][ch1] > 2.0)

pairs = pair_selector.select_pairs_custom(
    n_channels=n_channels,
    selection_func=custom_criteria,
    metadata={'quality': channel_quality}
)
```

## Mt. Rainier Notebook

The package includes a comprehensive Jupyter notebook (`notebooks/MtRainier_DAS_CrossCorrelation.ipynb`) that demonstrates:

- Complete DAS processing workflow
- Flexible channel pair selection strategies
- Cross-correlation analysis with quality control
- Velocity analysis and visualization
- Advanced stacking methods (linear, phase-weighted, robust)

## Examples

### Basic Workflow
```bash
python examples/basic_workflow.py
```

### Channel Pair Strategy Comparison
```bash
python examples/channel_pair_strategies.py
```

## Configuration

DAS-NoisePy uses YAML configuration files for flexible parameter management:

```yaml
# DAS-specific parameters
das:
  sampling_rate: 1000.0      # Hz
  channel_spacing: 2.0       # m
  gauge_length: 20.0         # m

# Preprocessing options  
preprocessing:
  detrend: true
  bandpass_filter: true
  freq_min: 0.5             # Hz
  freq_max: 25.0            # Hz
  quality_control: true
  snr_threshold: 2.0

# Channel pair selection
channel_pairs:
  selection_method: adaptive
  min_separation: 10.0       # m
  max_separation: 1000.0     # m
  target_pairs_per_distance: 50

# Cross-correlation parameters
cross_correlation:
  method: fft
  max_lag_time: 20.0        # seconds
  normalization: cross
  whitening: true
  stack_method: linear
```

## Supported Data Formats

- **HDF5** - Primary format with full metadata support
- **SEG-Y** - Seismic industry standard (requires `segyio`)
- **TDMS** - National Instruments format (requires `nptdms`) 
- **NumPy** - Simple array format for testing

## Processing Pipeline

1. **Data Loading** - Multi-format support with automatic detection
2. **Quality Control** - SNR-based channel assessment
3. **Preprocessing** - Detrending, filtering, spectral whitening
4. **Pair Selection** - Flexible strategies for optimal channel pairs
5. **Cross-Correlation** - FFT-based or time-domain correlation
6. **Stacking** - Multiple stacking methods with quality control
7. **Analysis** - Velocity analysis and visualization tools

## Key Improvements over Original NoisePy

✅ **Flexible Channel Pair Selection** - Multiple strategies instead of fixed approach  
✅ **DAS-Specific Preprocessing** - Tailored for DAS data characteristics  
✅ **Quality-Based Processing** - SNR-aware pair selection and stacking  
✅ **Comprehensive Visualization** - Rich plotting and analysis tools  
✅ **Configuration Management** - YAML-based parameter control  
✅ **Command Line Interface** - Automated processing workflows  
✅ **Better Documentation** - Complete examples and tutorials  

## Requirements

```
numpy>=1.20.0
scipy>=1.7.0  
matplotlib>=3.3.0
pandas>=1.3.0
obspy>=1.2.0
h5py>=3.1.0
xarray>=0.18.0
pyyaml>=5.4.0
tqdm>=4.60.0
click>=8.0.0
jupyter>=1.0.0
```

Optional dependencies:
- `segyio` - For SEG-Y format support
- `nptdms` - For TDMS format support
- `mpi4py` - For parallel processing
- `numba` - For performance optimization

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Submitting bug reports and feature requests
- Setting up development environment
- Code style and testing requirements
- Submitting pull requests

## Citation

If you use DAS-NoisePy in your research, please cite:

```bibtex
@software{das_noisepy,
  title={DAS-NoisePy: Cross-correlation analysis for Distributed Acoustic Sensing data},
  author={Denolle Lab},
  year={2024},
  url={https://github.com/Denolle-Lab/das-noisepy}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original NoisePy development team
- Mt. Rainier DAS deployment contributors
- NoisePy4DAS-SeaDAS project
- University of Washington Fiber Lab

## Related Projects

- [NoisePy](https://github.com/mdenolle/NoisePy) - Original ambient noise seismology toolkit
- [ObsPy](https://github.com/obspy/obspy) - Seismological data processing framework
- [SeisNoise.jl](https://github.com/tclements/SeisNoise.jl) - Julia implementation of noise correlation

---

**Contact**: [denolle@uw.edu](mailto:denolle@uw.edu)  
**Documentation**: [https://das-noisepy.readthedocs.io](https://das-noisepy.readthedocs.io)  
**Issues**: [https://github.com/Denolle-Lab/das-noisepy/issues](https://github.com/Denolle-Lab/das-noisepy/issues)
