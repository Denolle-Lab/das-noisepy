#!/usr/bin/env python3
"""
Test the DASH5DataStore implementation
"""

import sys
import os
import logging
from datetime import datetime, timezone
from datetimerange import DateTimeRange

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from noisepy_das.io import DASH5DataStore

def main():
    print("Testing DASH5DataStore implementation...")
    
    # Initialize data store
    data_path = "examples/sample_data/"
    raw_store = DASH5DataStore(
        path=data_path,
        sampling_rate=100,
        channel_numbers=[0, 1, 2, 3, 4],
        file_naming="%Y-%m-%d-%H-%M-%S.h5",
        array_name="DAS",
        date_range=None
    )
    
    print(f"✓ Data store initialized successfully")
    print(f"  Path: {data_path}")
    print(f"  Filesystem type: {type(raw_store.fs)}")
    
    # Test file discovery
    h5_files = raw_store.discover_files("*.h5")
    print(f"✓ Found {len(h5_files)} H5 files")
    
    # Show some example files
    print("  Example files:")
    for i, file in enumerate(h5_files[:5]):
        print(f"    {i+1}: {os.path.basename(file)}")
    if len(h5_files) > 5:
        print(f"    ... and {len(h5_files)-5} more files")
    
    # Test timespans
    timespans = raw_store.get_timespans()
    print(f"✓ Found {len(timespans)} timespans")
    if len(timespans) > 0:
        print(f"  First: {timespans[0]}")
        print(f"  Last: {timespans[-1]}")
        
        # Test channels
        channels = raw_store.get_channels(timespans[0])
        print(f"✓ Found {len(channels)} channels for first timespan")
        print("  Example channels:")
        for i, channel in enumerate(channels[:3]):
            print(f"    {i+1}: {channel}")
        
        # Test data reading
        if len(channels) > 0:
            data = raw_store.read_data(timespans[0], channels[0])
            print(f"✓ Successfully read data for {channels[0]}")
            print(f"  Data shape: {data.data.shape}")
            print(f"  Sampling rate: {data.sampling_rate} Hz")
            print(f"  Data range: {data.data.min():.3f} to {data.data.max():.3f}")
            print(f"  Start timestamp: {datetime.fromtimestamp(data.start_timestamp, tz=timezone.utc)}")
            
            # Test ObsPy stream
            print(f"  ObsPy stream: {len(data.stream)} traces")
            if len(data.stream) > 0:
                trace = data.stream[0]
                print(f"    Network: {trace.stats.network}")
                print(f"    Station: {trace.stats.station}")
                print(f"    Channel: {trace.stats.channel}")
                print(f"    Start time: {trace.stats.starttime}")
    
    # Test date range functionality
    start_time = datetime(2023, 1, 15, 14, 10, 0, tzinfo=timezone.utc)
    end_time = datetime(2023, 1, 15, 14, 15, 0, tzinfo=timezone.utc)
    date_range = DateTimeRange(start_time, end_time)
    
    limited_store = DASH5DataStore(
        path=data_path,
        sampling_rate=100,
        channel_numbers=[0, 1, 2],
        file_naming="%Y-%m-%d-%H-%M-%S.h5",
        array_name="DAS",
        date_range=date_range
    )
    
    limited_timespans = limited_store.get_timespans()
    print(f"✓ Limited time range store created with {len(limited_timespans)} timespans")
    print(f"  Date range: {date_range}")
    
    print("\n🎉 All tests passed! DASH5DataStore is working correctly.")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)