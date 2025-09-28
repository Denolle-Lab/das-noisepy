#!/usr/bin/env python3
"""
Generate sample HDF5 files for testing the DASH5DataStore.
This creates synthetic DAS data in the expected format.
"""

import os
import h5py
import numpy as np
from datetime import datetime, timedelta


def create_sample_h5_file(filename, start_time, sampling_rate=100, n_channels=10, duration_minutes=1):
    """
    Create a sample H5 file with synthetic DAS data.
    
    Parameters:
        filename: output filename
        start_time: start time as datetime object
        sampling_rate: sampling rate in Hz
        n_channels: number of channels
        duration_minutes: duration in minutes
    """
    
    # Calculate number of samples
    n_samples = int(sampling_rate * duration_minutes * 60)
    
    # Generate synthetic data
    time_array = np.arange(n_samples) / sampling_rate
    
    # Create synthetic DAS data with some spatial correlation
    # Each channel has a base frequency plus some noise
    data = np.zeros((n_samples, n_channels))
    
    for ch in range(n_channels):
        # Base signal with channel-dependent frequency
        base_freq = 1.0 + 0.1 * ch  # 1-2 Hz range
        signal = np.sin(2 * np.pi * base_freq * time_array)
        
        # Add some noise
        noise = 0.2 * np.random.randn(n_samples)
        
        # Add some spatial coherence
        if ch > 0:
            spatial_coupling = 0.3 * data[:, ch-1]
            signal += spatial_coupling
            
        data[:, ch] = signal + noise
    
    # Create H5 file
    with h5py.File(filename, 'w') as f:
        # Create the expected group structure
        acq_group = f.create_group('Acquisition')
        raw_group = acq_group.create_group('Raw[0]')
        
        # Store the raw data (samples x channels)
        raw_group.create_dataset('RawData', data=data)
        
        # Store timestamps (in microseconds since epoch)
        start_timestamp_us = int(start_time.timestamp() * 1e6)
        timestamps = start_timestamp_us + (time_array * 1e6).astype(int)
        raw_group.create_dataset('RawDataTime', data=timestamps)
        
        # Add some metadata
        raw_group.attrs['sampling_rate'] = sampling_rate
        raw_group.attrs['n_channels'] = n_channels
        raw_group.attrs['start_time'] = start_time.isoformat()
        
    print(f"Created {filename} with shape {data.shape}")


def main():
    """Generate a set of sample H5 files"""
    
    output_dir = "examples/sample_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate files for one hour with 1-minute intervals
    start_time = datetime(2023, 1, 15, 14, 0, 0)  # 2023-01-15 14:00:00
    
    print("Generating sample H5 files...")
    
    for minute in range(60):  # 1 hour = 60 minutes
        current_time = start_time + timedelta(minutes=minute)
        filename = os.path.join(output_dir, current_time.strftime("%Y-%m-%d-%H-%M-%S.h5"))
        
        create_sample_h5_file(
            filename=filename,
            start_time=current_time,
            sampling_rate=100,
            n_channels=10,
            duration_minutes=1
        )
        
        # Print progress every 10 files
        if (minute + 1) % 10 == 0:
            print(f"Generated {minute + 1}/60 files")
    
    print(f"\nSuccessfully generated 60 sample H5 files in {output_dir}/")
    print("Files cover the time period: 2023-01-15 14:00:00 to 2023-01-15 14:59:59")
    print("Each file contains 1 minute of 100 Hz data across 10 channels")


if __name__ == "__main__":
    main()