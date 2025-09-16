# SAR Data Exploration

This repository demonstrates a workflow for downloading, extracting, and exploring **Sentinel-1 Synthetic Aperture Radar (SAR)** data. It uses [ASF Search](https://github.com/asfadmin/Discovery-asf_search) to query and download data, [pyroSAR](https://pyrosar.readthedocs.io/) to read metadata, and [spatialist](https://spatialist.readthedocs.io/) to handle raster files.

---

## Features

- Query ASF for Sentinel-1 GRD scenes within a given **date range** and **bounding polygon**  
- Download and extract `.zip` scenes  
- Identify and process **VV-polarization TIFF bands**  
- Compute basic statistics (shape, min, max, mean, std)  
- Save:
  - **Histograms** of backscatter values  
  - **Quick preview images** in grayscale  

All results are stored in `./eda_results`.

---

## Installation

Install the required libraries:

```bash
pip install asf_search pyroSAR spatialist matplotlib numpy


## Next Steps

Potential extensions for SAR exploration:

Process other polarizations (VH, HH, HV) for comparison

Convert amplitude to sigma0 (dB) for calibrated backscatter

Apply speckle filtering to reduce noise

Build time series of backscatter for change detection

Overlay with geospatial layers (e.g., land cover, water bodies)

Explore interferometric SAR (InSAR) for deformation and topography analysis
