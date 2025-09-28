"""
Command-line interface for DAS-NoisePy
"""

import click
import logging
from pathlib import Path
from typing import Optional

from . import (
    DASProcessor,
    ChannelPairSelector,
    CrossCorrelator,
    load_config,
    save_config,
    get_default_config,
    load_das_data,
    save_results
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """DAS-NoisePy: Cross-correlation analysis for DAS data"""
    setup_logging(verbose)


@cli.command()
@click.option('--output', '-o', default='config.yaml', 
              help='Output configuration file')
def create_config(output):
    """Create a default configuration file"""
    config = get_default_config()
    save_config(config, output)
    click.echo(f"Created default configuration: {output}")


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file')
@click.option('--output-dir', '-o', default='./output', 
              help='Output directory')
@click.option('--format', default='auto', 
              help='Input data format (auto, hdf5, segy, tdms, npy)')
@click.option('--pair-strategy', default='adaptive',
              type=click.Choice(['distance', 'quality', 'adaptive', 'regular']),
              help='Channel pair selection strategy')
def process(data_file, config, output_dir, format, pair_strategy):
    """Process DAS data for cross-correlation analysis"""
    
    # Load configuration
    if config:
        cfg = load_config(config)
        click.echo(f"Loaded configuration from {config}")
    else:
        cfg = get_default_config()
        click.echo("Using default configuration")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load DAS data
        click.echo(f"Loading DAS data from {data_file}")
        data = load_das_data(data_file, format=format)
        click.echo(f"Loaded data: {data.strain.shape} (time, channel)")
        
        # Initialize processor
        click.echo("Preprocessing DAS data...")
        processor = DASProcessor(cfg['preprocessing'])
        processor.data = data
        
        # Preprocess data
        processed_data = processor.preprocess()
        info = processor.get_channel_info()
        click.echo(f"Preprocessing completed. Good channels: {info.get('good_channels', 'N/A')}")
        
        # Channel pair selection
        click.echo(f"Selecting channel pairs using '{pair_strategy}' strategy...")
        pair_selector = ChannelPairSelector(
            channel_spacing=cfg['das']['channel_spacing']
        )
        
        n_channels = len(processed_data.channel)
        snr = processed_data.attrs.get('snr', None)
        
        if pair_strategy == 'distance':
            pairs = pair_selector.select_pairs_by_distance(
                n_channels=n_channels,
                min_separation=cfg['channel_pairs']['min_separation'],
                max_separation=cfg['channel_pairs']['max_separation'],
                step_size=cfg['channel_pairs'].get('step_size'),
                max_pairs_per_separation=cfg['channel_pairs'].get('max_pairs_per_separation')
            )
        elif pair_strategy == 'quality' and snr is not None:
            pairs = pair_selector.select_pairs_by_quality(
                n_channels=n_channels,
                snr=snr,
                min_snr=cfg['preprocessing']['snr_threshold'],
                min_separation=cfg['channel_pairs']['min_separation'],
                max_separation=cfg['channel_pairs']['max_separation']
            )
        elif pair_strategy == 'adaptive' and snr is not None:
            pairs = pair_selector.select_pairs_adaptive(
                n_channels=n_channels,
                snr=snr,
                target_pairs_per_distance=cfg['channel_pairs']['target_pairs_per_distance'],
                distance_bins=cfg['channel_pairs']['distance_bins']
            )
        elif pair_strategy == 'regular':
            pairs = pair_selector.select_pairs_regular_grid(
                n_channels=n_channels,
                separation_distances=[cfg['channel_pairs']['min_separation'], 
                                    cfg['channel_pairs']['max_separation']],
                spacing_between_pairs=10
            )
        else:
            raise ValueError(f"Strategy '{pair_strategy}' not available or requires SNR data")
        
        click.echo(f"Selected {len(pairs)} channel pairs")
        
        # Save pair information
        pair_file = output_path / 'channel_pairs.csv'
        pair_selector.save_pairs(str(pair_file))
        click.echo(f"Saved channel pairs to {pair_file}")
        
        # Cross-correlation
        click.echo("Computing cross-correlations...")
        correlator = CrossCorrelator(cfg['cross_correlation'])
        
        cc_results = correlator.compute_correlations(
            data=processed_data,
            pairs=pairs,
            time_chunked=True
        )
        
        click.echo(f"Cross-correlation completed for {len(cc_results['correlations'])} pairs")
        
        # Stack correlations
        click.echo("Stacking cross-correlations...")
        stacked_results = correlator.stack_correlations(cc_results)
        
        # Save results
        click.echo("Saving results...")
        results_file = output_path / 'cross_correlations.h5'
        correlator.save_results(stacked_results, results_file)
        
        # Save summary information
        save_results(stacked_results, output_path, format='csv', include_metadata=True)
        
        # Save configuration used
        config_used_file = output_path / 'config_used.yaml'
        save_config(cfg, config_used_file)
        
        click.echo(f"Processing completed successfully!")
        click.echo(f"Results saved to: {output_path}")
        click.echo(f"- Cross-correlations: {results_file}")
        click.echo(f"- Channel pairs: {pair_file}")
        click.echo(f"- Configuration: {config_used_file}")
        
    except Exception as e:
        click.echo(f"Error during processing: {e}", err=True)
        raise click.ClickException(str(e))


@cli.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./plots', 
              help='Output directory for plots')
@click.option('--format', default='png', 
              type=click.Choice(['png', 'pdf', 'svg']),
              help='Output image format')
def plot(results_file, output_dir, format):
    """Generate plots from cross-correlation results"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load results
        click.echo(f"Loading results from {results_file}")
        correlator = CrossCorrelator()
        results = correlator.load_results(results_file)
        
        # Generate plots
        import matplotlib.pyplot as plt
        import numpy as np
        
        click.echo("Generating plots...")
        
        # Extract data
        correlations = results['correlations']
        lag_times = results['lag_times']
        
        pair_ids = list(correlations.keys())
        separations = [correlations[pid]['pair'].separation for pid in pair_ids]
        cc_data = np.array([correlations[pid]['correlation'] for pid in pair_ids])
        qualities = [correlations[pid]['quality'] for pid in pair_ids]
        
        # Sort by separation
        sort_idx = np.argsort(separations)
        separations_sorted = np.array(separations)[sort_idx]
        cc_data_sorted = cc_data[sort_idx, :]
        qualities_sorted = np.array(qualities)[sort_idx]
        
        # Plot 1: Cross-correlation matrix
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(cc_data_sorted, aspect='auto', cmap='seismic',
                      extent=[lag_times[0], lag_times[-1], 
                             len(separations_sorted)-0.5, -0.5],
                      vmin=-np.percentile(np.abs(cc_data_sorted), 95),
                      vmax=np.percentile(np.abs(cc_data_sorted), 95))
        ax.set_xlabel('Lag Time (s)')
        ax.set_ylabel('Pair Index (sorted by separation)')
        ax.set_title('Cross-Correlation Matrix')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        
        plot_file = output_path / f'cross_correlation_matrix.{format}'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        click.echo(f"Saved: {plot_file}")
        
        # Plot 2: Quality analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].scatter(separations_sorted, qualities_sorted, alpha=0.6, s=20)
        axes[0].set_xlabel('Separation Distance (m)')
        axes[0].set_ylabel('Correlation Quality')
        axes[0].set_title('Quality vs Separation Distance')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(qualities_sorted, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(qualities_sorted), color='red', linestyle='--',
                       label=f'Mean: {np.mean(qualities_sorted):.4f}')
        axes[1].set_xlabel('Correlation Quality')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Distribution of Correlation Qualities')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plot_file = output_path / f'quality_analysis.{format}'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        click.echo(f"Saved: {plot_file}")
        
        # Plot 3: Example correlations
        fig, ax = plt.subplots(figsize=(12, 6))
        
        example_separations = [50, 100, 200, 300, 400]
        colors = plt.cm.plasma(np.linspace(0, 1, len(example_separations)))
        
        for i, target_sep in enumerate(example_separations):
            if target_sep <= np.max(separations_sorted):
                sep_diff = np.abs(separations_sorted - target_sep)
                closest_idx = np.argmin(sep_diff)
                actual_sep = separations_sorted[closest_idx]
                
                cc = cc_data_sorted[closest_idx, :]
                ax.plot(lag_times, cc, color=colors[i], alpha=0.8,
                       label=f'{actual_sep:.0f} m')
        
        ax.set_xlabel('Lag Time (s)')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Example Cross-Correlations')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Separation')
        
        plot_file = output_path / f'example_correlations.{format}'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        click.echo(f"Saved: {plot_file}")
        
        click.echo(f"Plotting completed! Plots saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error during plotting: {e}", err=True)
        raise click.ClickException(str(e))


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration file')
def validate(data_file, config):
    """Validate DAS data and configuration"""
    
    try:
        # Load configuration
        if config:
            cfg = load_config(config)
            click.echo(f"Configuration loaded from {config}")
            
            # Validate configuration
            from .utils.config import validate_config, print_config_summary
            validated_cfg = validate_config(cfg)
            click.echo("✓ Configuration is valid")
            
            print_config_summary(validated_cfg)
        
        # Load and validate data
        click.echo(f"\nValidating DAS data: {data_file}")
        data = load_das_data(data_file)
        
        click.echo("✓ Data loaded successfully")
        click.echo(f"  Shape: {data.strain.shape} (time, channel)")
        click.echo(f"  Duration: {len(data.time) / data.attrs['sampling_rate']:.1f} seconds")
        click.echo(f"  Channels: {len(data.channel)}")
        click.echo(f"  Array length: {len(data.channel) * data.attrs['channel_spacing'] / 1000:.2f} km")
        click.echo(f"  Sampling rate: {data.attrs['sampling_rate']} Hz")
        
        # Basic quality checks
        strain_data = data.strain.values
        
        # Check for NaN/Inf values
        n_nan = np.sum(np.isnan(strain_data))
        n_inf = np.sum(np.isinf(strain_data))
        
        if n_nan > 0:
            click.echo(f"⚠ Warning: {n_nan} NaN values found ({100*n_nan/strain_data.size:.2f}%)")
        if n_inf > 0:
            click.echo(f"⚠ Warning: {n_inf} infinite values found ({100*n_inf/strain_data.size:.2f}%)")
        
        if n_nan == 0 and n_inf == 0:
            click.echo("✓ No NaN or infinite values detected")
        
        # Check data range
        data_min = np.nanmin(strain_data)
        data_max = np.nanmax(strain_data)
        data_std = np.nanstd(strain_data)
        
        click.echo(f"  Data range: {data_min:.2e} to {data_max:.2e}")
        click.echo(f"  Standard deviation: {data_std:.2e}")
        
        # Check for channels with zero variance (dead channels)
        channel_vars = np.var(strain_data, axis=0)
        dead_channels = np.sum(channel_vars == 0)
        if dead_channels > 0:
            click.echo(f"⚠ Warning: {dead_channels} channels with zero variance detected")
        else:
            click.echo("✓ All channels have non-zero variance")
        
        click.echo("\n✓ Validation completed successfully!")
        
    except Exception as e:
        click.echo(f"✗ Validation failed: {e}", err=True)
        raise click.ClickException(str(e))


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == '__main__':
    main()