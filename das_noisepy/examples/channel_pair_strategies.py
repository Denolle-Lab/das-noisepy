"""
Example demonstrating different channel pair selection strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from das_noisepy import (
    ChannelPairSelector,
    create_synthetic_das_data,
    get_default_config
)


def main():
    """Compare different channel pair selection strategies"""
    
    print("Channel Pair Selection Strategy Comparison")
    print("=" * 50)
    
    # Create test data
    config = get_default_config()
    data = create_synthetic_das_data(
        n_time=10000,
        n_channels=1000,
        channel_spacing=config['das']['channel_spacing']
    )
    
    n_channels = len(data.channel)
    
    # Simulate channel quality (SNR)
    # Create realistic SNR pattern with some bad channels
    np.random.seed(42)  # For reproducible results
    base_snr = 5.0
    snr_variation = np.random.normal(0, 1, n_channels)
    distance_effect = 0.5 * np.sin(2 * np.pi * np.arange(n_channels) / 200)
    
    # Add some bad channel regions
    bad_region1 = slice(200, 250)  # Bad region 1
    bad_region2 = slice(700, 750)  # Bad region 2
    
    snr = base_snr + snr_variation + distance_effect
    snr[bad_region1] = 0.5  # Low SNR region
    snr[bad_region2] = 0.8  # Another low SNR region
    
    print(f"Test setup: {n_channels} channels, SNR range: {np.min(snr):.1f} - {np.max(snr):.1f}")
    
    # Initialize selector
    pair_selector = ChannelPairSelector(
        channel_spacing=config['das']['channel_spacing']
    )
    
    # Strategy 1: Distance-based selection
    print("\n1. Distance-based selection:")
    pairs_distance = pair_selector.select_pairs_by_distance(
        n_channels=n_channels,
        min_separation=50.0,
        max_separation=500.0,
        step_size=25.0,
        max_pairs_per_separation=30
    )
    print(f"   Selected {len(pairs_distance)} pairs")
    
    # Strategy 2: Quality-based selection
    print("\n2. Quality-based selection:")
    pairs_quality = pair_selector.select_pairs_by_quality(
        n_channels=n_channels,
        snr=snr,
        min_snr=3.0,
        min_separation=50.0,
        max_separation=500.0
    )
    print(f"   Selected {len(pairs_quality)} pairs")
    
    # Strategy 3: Adaptive selection
    print("\n3. Adaptive selection:")
    pairs_adaptive = pair_selector.select_pairs_adaptive(
        n_channels=n_channels,
        snr=snr,
        target_pairs_per_distance=20,
        distance_bins=15
    )
    print(f"   Selected {len(pairs_adaptive)} pairs")
    
    # Strategy 4: Regular grid
    print("\n4. Regular grid selection:")
    pairs_regular = pair_selector.select_pairs_regular_grid(
        n_channels=n_channels,
        separation_distances=[50, 100, 200, 300, 400],
        spacing_between_pairs=20
    )
    print(f"   Selected {len(pairs_regular)} pairs")
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    strategies = [
        ('Distance', pairs_distance),
        ('Quality', pairs_quality), 
        ('Adaptive', pairs_adaptive),
        ('Regular Grid', pairs_regular)
    ]
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Row 1: Channel quality and pair locations
    distances = np.arange(n_channels) * config['das']['channel_spacing']
    
    for i, (name, pairs) in enumerate(strategies):
        ax = axes[0, i]
        
        # Plot SNR
        ax.plot(distances, snr, 'k-', alpha=0.5, label='SNR')
        ax.axhline(3.0, color='red', linestyle='--', alpha=0.7, label='Threshold')
        
        # Highlight bad regions
        ax.axvspan(distances[bad_region1][0], distances[bad_region1][-1], 
                   alpha=0.2, color='red')
        ax.axvspan(distances[bad_region2][0], distances[bad_region2][-1], 
                   alpha=0.2, color='red')
        
        # Mark selected channels
        selected_channels = set()
        for pair in pairs:
            selected_channels.add(pair.ch1)
            selected_channels.add(pair.ch2)
        
        selected_distances = [distances[ch] for ch in selected_channels]
        selected_snr = [snr[ch] for ch in selected_channels]
        
        ax.scatter(selected_distances, selected_snr, c=colors[i], 
                  s=10, alpha=0.7, label='Selected')
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('SNR')
        ax.set_title(f'{name} Strategy\n{len(pairs)} pairs, {len(selected_channels)} channels')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 2: Pair separation distributions
    for i, (name, pairs) in enumerate(strategies):
        ax = axes[1, i]
        
        separations = [pair.separation for pair in pairs]
        ax.hist(separations, bins=20, alpha=0.7, color=colors[i], edgecolor='black')
        ax.set_xlabel('Separation Distance (m)')
        ax.set_ylabel('Number of Pairs')
        ax.set_title(f'{name}: Separation Distribution')
        ax.grid(True, alpha=0.3)
    
    # Row 3: Pair connectivity matrices (subsampled for visibility)
    max_display_channels = 200  # Display subset for clarity
    
    for i, (name, pairs) in enumerate(strategies):
        ax = axes[2, i]
        
        connectivity = np.zeros((max_display_channels, max_display_channels))
        
        for pair in pairs:
            if pair.ch1 < max_display_channels and pair.ch2 < max_display_channels:
                connectivity[pair.ch1, pair.ch2] = 1
                connectivity[pair.ch2, pair.ch1] = 1
        
        im = ax.imshow(connectivity, cmap='Blues', aspect='auto')
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Channel Index')
        ax.set_title(f'{name}: Connectivity Matrix\\n(First {max_display_channels} channels)')
    
    plt.tight_layout()
    
    # Save comparison plot
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)
    plot_file = output_dir / 'pair_selection_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison plot saved to: {plot_file}")
    
    # Print detailed comparison
    print("\nDetailed Strategy Comparison:")
    print("-" * 50)
    
    for name, pairs in strategies:
        separations = [pair.separation for pair in pairs]
        distances_used = [pair.distance for pair in pairs]
        
        # Calculate quality metrics for selected pairs
        pair_qualities = []
        for pair in pairs:
            avg_snr = (snr[pair.ch1] + snr[pair.ch2]) / 2
            pair_qualities.append(avg_snr)
        
        print(f"\n{name} Strategy:")
        print(f"  Total pairs: {len(pairs)}")
        print(f"  Separation range: {np.min(separations):.1f} - {np.max(separations):.1f} m")
        print(f"  Mean separation: {np.mean(separations):.1f} ± {np.std(separations):.1f} m")
        print(f"  Position coverage: {np.min(distances_used):.1f} - {np.max(distances_used):.1f} m")
        print(f"  Average pair quality (SNR): {np.mean(pair_qualities):.2f} ± {np.std(pair_qualities):.2f}")
        
        # Count pairs in bad regions
        bad_pairs = 0
        for pair in pairs:
            if ((bad_region1.start <= pair.ch1 <= bad_region1.stop) or 
                (bad_region1.start <= pair.ch2 <= bad_region1.stop) or
                (bad_region2.start <= pair.ch1 <= bad_region2.stop) or 
                (bad_region2.start <= pair.ch2 <= bad_region2.stop)):
                bad_pairs += 1
        
        print(f"  Pairs involving bad channels: {bad_pairs} ({100*bad_pairs/len(pairs):.1f}%)")
    
    print("\nRecommendations:")
    print("- Distance-based: Good for uniform sampling, but includes bad channels")
    print("- Quality-based: Best for signal quality, but may have spatial gaps") 
    print("- Adaptive: Balanced approach, good spatial coverage with quality control")
    print("- Regular grid: Systematic sampling, useful for specific applications")
    
    print("\n✓ Channel pair selection comparison completed!")


if __name__ == '__main__':
    main()