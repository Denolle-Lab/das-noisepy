# das-noisepy

A comprehensive collection of scripts and notebooks that demonstrate using noisepy for processing Distributed Acoustic Sensing (DAS) data.

## Overview

This repository provides tools and examples for:
- Creating DAS datastores from file structures
- Performing cross-correlation analysis using noisepy
- Visualizing results with matplotlib

## Setup

### Using Conda (Recommended)

Create a conda environment with all required dependencies:

```bash
conda env create -f environment.yml
conda activate das-noisepy
```

### Using pip

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

The `notebooks/` directory contains example notebooks demonstrating:

1. **DAS Datastore Creation**: How to create a DASdatastore from file structures using glob
2. **Cross-correlation Analysis**: Using noisepy for cross-correlation with H5DASdatastore and basic plotting

## Dependencies

- obspy: Seismic data processing
- matplotlib: Plotting and visualization
- noisepy: Ambient noise seismology processing
- noisepy-io: I/O utilities for noisepy
- h5py: HDF5 file handling

## Getting Started

1. Clone this repository
2. Set up the environment (see Setup section above)
3. Navigate to the `notebooks/` directory
4. Open and run the example notebooks
