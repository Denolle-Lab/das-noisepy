# das-noisepy

A fullstack of scripts that demonstrate using noisepy and DAS (Distributed Acoustic Sensing) data.

This repository provides tools for working with DAS data stored in HDF5 format, including a standalone implementation of the DASH5DataStore adapted from the noisepy-io project.

## Features

- **DASH5DataStore**: A data store implementation for reading DAS data from HDF5 files
- **File Discovery**: Use glob patterns to discover and load specific data files
- **Time Range Support**: Load data for specific time periods 
- **ObsPy Integration**: Convert DAS data to ObsPy streams for further analysis
- **Example Notebooks**: Comprehensive tutorials and demonstrations

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Denolle-Lab/das-noisepy.git
cd das-noisepy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from noisepy_das.io import DASH5DataStore
from datetime import datetime, timezone
from datetimerange import DateTimeRange

# Initialize the data store
raw_store = DASH5DataStore(
    path="/path/to/your/h5/files/",
    sampling_rate=100,  # Hz
    channel_numbers=[0, 1, 2, 3, 4],
    file_naming="%Y-%m-%d-%H-%M-%S.h5",
    array_name="DAS"
)

# Discover available files
h5_files = raw_store.discover_files("*.h5")
print(f"Found {len(h5_files)} H5 files")

# Get available time spans and channels
timespans = raw_store.get_timespans()
channels = raw_store.get_channels(timespans[0])

# Read data for a specific channel and time
data = raw_store.read_data(timespans[0], channels[0])
print(f"Data shape: {data.data.shape}")
print(f"ObsPy stream: {data.stream}")
```

### File Discovery with Glob Patterns

```python
# Find all H5 files
all_files = raw_store.discover_files("*.h5")

# Find files from a specific date
daily_files = raw_store.discover_files("2023-01-15-*.h5")

# Find files from specific hours
hourly_files = raw_store.discover_files("*-14-*.h5")

# Find files from first 5 minutes of any hour
first5min_files = raw_store.discover_files("*-0[0-4]-*.h5")
```

## Repository Structure

```
das-noisepy/
├── src/noisepy_das/
│   ├── __init__.py
│   └── io/
│       ├── __init__.py
│       ├── datatypes.py      # Core data types (Channel, Station, etc.)
│       ├── stores.py         # Base store classes
│       ├── utils.py          # Utility functions
│       └── h5store.py        # DASH5DataStore implementation
├── notebooks/
│   └── dash5_datastore_demo.ipynb  # Comprehensive tutorial
├── examples/
│   └── sample_data/          # Generated sample data files
├── tests/
├── requirements.txt
├── create_sample_data.py     # Script to generate sample data
├── test_implementation.py    # Test script
└── README.md
```

## Data Format

The DASH5DataStore expects HDF5 files with the following structure:

```
filename: YYYY-MM-DD-HH-MM-SS.h5 (configurable pattern)
/Acquisition/Raw[0]/
├── RawData        # Dataset (samples × channels)
└── RawDataTime    # Dataset (timestamps in microseconds)
```

Each file typically contains 1 minute of data with all channels.

## Examples and Tutorials

### 1. Interactive Notebook
See `notebooks/dash5_datastore_demo.ipynb` for a comprehensive tutorial covering:
- Data store initialization
- File discovery with glob patterns
- Reading multi-channel data
- Working with date ranges
- ObsPy integration
- Basic plotting

### 2. Generate Sample Data
Create synthetic DAS data for testing:
```bash
python create_sample_data.py
```

This generates 60 sample H5 files (1 hour of 1-minute files) with synthetic DAS data.

### 3. Run Tests
Verify the implementation:
```bash
python test_implementation.py
```

## Advanced Features

### Date Range Filtering
```python
# Load only data within a specific time range
from datetimerange import DateTimeRange

date_range = DateTimeRange(
    datetime(2023, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
)

limited_store = DASH5DataStore(
    path="/path/to/data/",
    sampling_rate=100,
    channel_numbers=[0, 1, 2],
    date_range=date_range
)
```

### Custom Storage Options
```python
# For cloud storage or custom filesystems
storage_options = {
    's3': {'anon': False, 'key': 'your-key', 'secret': 'your-secret'}
}

raw_store = DASH5DataStore(
    path="s3://bucket/path/to/data/",
    sampling_rate=100,
    channel_numbers=[0, 1, 2],
    storage_options=storage_options
)
```

## Integration with NoisePy

This implementation is designed to be compatible with the broader NoisePy ecosystem for ambient noise seismology. The DASH5DataStore implements the same interface as other NoisePy data stores, making it easy to integrate into existing workflows.

## Dependencies

- h5py ≥ 3.8.0
- obspy ≥ 1.4.0  
- numpy ≥ 1.20.0
- fsspec ≥ 2023.1.0
- datetimerange ≥ 1.2.0
- tqdm ≥ 4.64.0
- psutil ≥ 5.9.0

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project builds upon the noisepy-io project and follows similar licensing terms. See individual source files for specific license information.

## Acknowledgments

- Based on the DASH5DataStore implementation from [noisepy-io](https://github.com/noisepy/noisepy-io)
- Part of the broader [NoisePy](https://github.com/noisepy) ecosystem for ambient noise seismology
