# WRF Postprocessing Applications

This directory contains Python scripts converted from Jupyter notebooks for interactive visualization of WRF model output data.

## Available Applications

### `make_plots_hlev.py`
Interactive plotting of WRF data interpolated to height levels.

**Features:**
- 2D and 3D variable visualization
- Interactive time controls
- Customizable colormaps and plot dimensions
- Dask cluster integration for performance
- Web-based interface with SSH tunneling support

**Usage:**
```bash
# Create application and show instructions (no web interface)
python make_plots_hlev.py path/to/wrf_data.zarr --n-workers 4

# Start web interface (graphical environments only)
python make_plots_hlev.py path/to/wrf_data.zarr --serve --web-port 8860
```

**Web Interface Setup:**

*For graphical environments:*
- Run with `--serve` flag and the web interface will open automatically

*For headless/remote servers:*
1. Run the script normally to create the application
2. Set up SSH tunneling from your local machine:
```bash
ssh -N -f -L localhost:8860:localhost:8860 user@server
```
3. Start the web server manually:
```python
import panel as pn
from apps.make_plots_hlev import main
import sys

# Set up arguments
sys.argv = ['make_plots_hlev.py', 'path/to/data.zarr', '--n-workers=1']

# Create the app
main()

# Start server
pn.serve(app, port=8860, show=False)
```
4. Open http://localhost:8860 in your local browser

### `make_plots_plev.py`
Interactive plotting of WRF data interpolated to pressure levels.

**Features:**
- Pressure level visualization
- Interactive time and pressure controls
- Multiple colormap options
- Optimized for meteorological analysis

**Usage:**
```bash
python make_plots_plev.py path/to/wrf_plev_data.zarr --n-workers 4
```

### `make_plots_hlev_advanced.py`
Advanced interactive plotting for WRF data at height levels.

**Status:** ✅ **Enhanced with Advanced Features**
Comprehensive visualization suite with multiple analysis modes.

**Features:**
- **5 Visualization Modes:**
  - 2D Maps: Surface/near-surface variables
  - 3D Maps: Height-interpolated variables
  - Cross-Sections: Vertical profiles through domain
  - Time Series: Temporal evolution at specific points
  - Comparisons: Side-by-side variable comparison
- **Enhanced Controls:**
  - 12+ colormap options
  - Adjustable plot dimensions
  - Interactive spatial selection
  - Time animation controls
- **Advanced Analysis:**
  - Vertical cross-section plotting
  - Point time series extraction
  - Multi-variable comparisons
  - Custom styling options

**Usage:**
```bash
python make_plots_hlev_advanced.py ../data/wrf_hlevs.zarr --serve --web-port 8880
```

## Requirements

- xarray
- holoviews
- geoviews
- panel
- dask
- numpy
- cartopy

## Data Format

All scripts expect Zarr-formatted WRF data as produced by the `preprocess.py` script.

## Web Interface Setup

The plotting applications use Panel for web-based interfaces. For remote servers:

1. **Run the application with `--serve`:**
   ```bash
   python make_plots_hlev.py data.zarr --serve --web-port 8860
   ```

2. **Set up SSH tunneling from your local machine:**
   ```bash
   ssh -N -f -L localhost:8860:localhost:8860 user@server
   ```

3. **Open in your browser:**
   ```
   http://localhost:8860
   ```

## Development

These scripts were converted from Jupyter notebooks using automated extraction of code cells. Some manual cleanup may be needed for production deployment.</content>
<parameter name="filePath">/scratch365/swang18/Workspace/Projects/FATIMA_Darko/fatima_wrf/apps/README.md