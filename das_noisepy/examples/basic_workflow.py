"""
Example script demonstrating basic DAS cross-correlation workflow
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import DAS-NoisePy components
from das_noisepy import (
    DASProcessor,
    ChannelPairSelector, 
    CrossCorrelator,
    get_default_config,
    create_synthetic_das_data
)


def main():
    """Main workflow example"""
    
    print("DAS-NoisePy Basic Workflow Example")
    print("=" * 40)
    
    # 1. Create configuration
    print("\n1. Setting up configuration...")
    config = get_default_config()
    
    # Modify for this example
    config['das']['channel_spacing'] = 2.0  # 2m spacing
    config['preprocessing']['freq_min'] = 1.0  # 1-20 Hz band
    config['preprocessing']['freq_max'] = 20.0
    config['cross_correlation']['max_lag_time'] = 5.0  # 5 second max lag
    
    print(f"   Configured for {config['das']['channel_spacing']}m channel spacing")
    print(f"   Frequency band: {config['preprocessing']['freq_min']}-{config['preprocessing']['freq_max']} Hz")
    
    # 2. Create synthetic data (normally would load real data)
    print("\n2. Loading DAS data...")
    data = create_synthetic_das_data(
        n_time=20000,      # 20 seconds at 1000 Hz
        n_channels=500,    # 1 km array
        sampling_rate=1000.0,
        channel_spacing=config['das']['channel_spacing'],
        signal_velocity=3000.0
    )
    
    print(f"   Data shape: {data.strain.shape} (time, channel)")
    print(f"   Array length: {len(data.channel) * config['das']['channel_spacing'] / 1000:.1f} km")
    
    # 3. Preprocess data
    print("\n3. Preprocessing data...")
    processor = DASProcessor(config['preprocessing'])
    processor.data = data
    processed_data = processor.preprocess()
    
    info = processor.get_channel_info()
    print(f"   Good channels: {info['good_channels']}/{info['n_channels']}")
    
    # 4. Select channel pairs
    print("\n4. Selecting channel pairs...")
    pair_selector = ChannelPairSelector(
        channel_spacing=config['das']['channel_spacing']
    )
    
    # Use adaptive selection for good coverage
    pairs = pair_selector.select_pairs_adaptive(
        n_channels=len(processed_data.channel),
        snr=processed_data.attrs['snr'],
        target_pairs_per_distance=10,
        distance_bins=5
    )
    
    print(f"   Selected {len(pairs)} channel pairs")
    
    # 5. Compute cross-correlations
    print("\n5. Computing cross-correlations...")
    correlator = CrossCorrelator(config['cross_correlation'])
    
    cc_results = correlator.compute_correlations(
        data=processed_data,
        pairs=pairs,
        time_chunked=False  # Use full time series for this example
    )
    
    print(f"   Computed {len(cc_results['correlations'])} cross-correlations")
    
    # 6. Analyze results
    print("\n6. Analyzing results...")
    correlations = cc_results['correlations']
    lag_times = cc_results['lag_times']
    
    # Extract basic statistics
    separations = [correlations[pid]['pair'].separation for pid in correlations.keys()]
    qualities = [correlations[pid]['quality'] for pid in correlations.keys()]
    
    print(f"   Separation range: {np.min(separations):.1f} - {np.max(separations):.1f} m")
    print(f"   Quality range: {np.min(qualities):.4f} - {np.max(qualities):.4f}")
    
    # 7. Create basic visualization
    print("\n7. Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Raw data sample
    axes[0, 0].plot(data.time[:5000], data.strain[:5000, 100], 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Strain')
    axes[0, 0].set_title('Raw DAS Data (Channel 100)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Channel quality
    axes[0, 1].plot(processed_data.distance, 10*np.log10(processed_data.attrs['snr']))
    axes[0, 1].axhline(10*np.log10(config['preprocessing']['snr_threshold']), 
                       color='red', linestyle='--', label='Threshold')
    axes[0, 1].set_xlabel('Distance (m)')
    axes[0, 1].set_ylabel('SNR (dB)')
    axes[0, 1].set_title('Channel Quality')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Example cross-correlations
    example_pairs = list(correlations.keys())[:5]
    colors = plt.cm.tab10(np.linspace(0, 1, len(example_pairs)))
    
    for i, pid in enumerate(example_pairs):
        cc = correlations[pid]['correlation']
        sep = correlations[pid]['pair'].separation
        axes[1, 0].plot(lag_times, cc, color=colors[i], alpha=0.7, 
                       label=f'{sep:.0f}m')
    
    axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Lag Time (s)')
    axes[1, 0].set_ylabel('Correlation Coefficient')
    axes[1, 0].set_title('Example Cross-Correlations')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quality vs separation
    axes[1, 1].scatter(separations, qualities, alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Separation Distance (m)')
    axes[1, 1].set_ylabel('Correlation Quality')
    axes[1, 1].set_title('Quality vs Separation')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)
    plot_file = output_dir / 'basic_example_results.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n   Plot saved to: {plot_file}")
    
    # 8. Save results
    print("\n8. Saving results...")
    
    # Save cross-correlations
    results_file = output_dir / 'example_correlations.h5'
    correlator.save_results(cc_results, results_file)
    
    # Save channel pairs
    pair_file = output_dir / 'example_pairs.csv'
    pair_selector.save_pairs(str(pair_file))
    
    print(f"   Cross-correlations saved to: {results_file}")
    print(f"   Channel pairs saved to: {pair_file}")
    
    print("\n✓ Example workflow completed successfully!")
    print("\nNext steps:")
    print("- Try different channel pair selection strategies")
    print("- Experiment with preprocessing parameters")
    print("- Use real DAS data files")
    print("- Explore the Mt. Rainier notebook for advanced analysis")


if __name__ == '__main__':
    main()