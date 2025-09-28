#!/usr/bin/env python3
"""
Quick demonstration of glob-based file discovery with DASH5DataStore
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from noisepy_das.io import DASH5DataStore

def main():
    print("🔍 DASH5DataStore Glob Pattern Demo")
    print("=" * 50)
    
    # Initialize data store
    data_path = "examples/sample_data/"
    if not os.path.exists(data_path):
        print(f"❌ Sample data path '{data_path}' not found.")
        print("💡 Run 'python create_sample_data.py' first to generate sample data.")
        return
    
    raw_store = DASH5DataStore(
        path=data_path,
        sampling_rate=100,
        channel_numbers=[0, 1, 2],
        file_naming="%Y-%m-%d-%H-%M-%S.h5",
        array_name="DAS"
    )
    
    # Demonstrate different glob patterns
    patterns = [
        ("All H5 files", "*.h5"),
        ("Files from 14:1x hours", "*-14-1*.h5"),
        ("Files from first 5 minutes of any hour", "*-0[0-4]-*.h5"),
        ("Files ending in :00 seconds", "*-00.h5"),
        ("Files from 14:30-14:39", "*-14-3*.h5")
    ]
    
    for description, pattern in patterns:
        files = raw_store.discover_files(pattern)
        print(f"\n📁 {description} ('{pattern}'):")
        print(f"   Found {len(files)} files")
        
        # Show first few examples
        for i, file in enumerate(files[:3]):
            filename = os.path.basename(file)
            print(f"   {i+1:2d}: {filename}")
        
        if len(files) > 3:
            print(f"       ... and {len(files)-3} more")
    
    print(f"\n✅ Glob pattern demonstration completed!")
    print(f"📊 Total files in directory: {len(raw_store.discover_files('*.h5'))}")

if __name__ == "__main__":
    main()