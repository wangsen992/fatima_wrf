# FATIMA WRF Postprocessing Tools

A collection of Python tools for preprocessing and analyzing Weather Research and Forecasting (WRF) model outputs for the FATIMA project.

## Features

- **WRF Data Preprocessing**: Convert raw WRF output files to analysis-ready datasets
- **Vertical Interpolation**: Regrid 3D variables to custom height or pressure levels
- **Variable Computation**: Automatically calculate derived meteorological variables (wind speed, direction, temperature, humidity)
- **Zarr Output**: Efficient storage in Zarr format for fast loading and analysis
- **Interactive Visualization**: Web-based plotting interface for data exploration
- **Configurable Processing**: Command-line interface with sensible defaults
- **Remote Server Support**: SSH tunneling support for headless environments

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fatima_wrf
```

2. Create conda environment:
```bash
conda env create -f requirements.txt
conda activate wrf_postproc
```

## Usage

### Basic Preprocessing

```bash
python preprocess.py
```

### Advanced Usage

```bash
python preprocess.py \
    --wrf-run /path/to/wrf/run \
    --case-name your_case_name \
    --file-prefix wrfout_d01 \
    --proc-dir ./output \
    --levs "np.arange(100, 5000, 100)" \
    --interp_var geopotential_height \
    --logfile preprocessing.log
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--wrf-run` | `/scratch365/swang18/Workspace/apps/wrf/run/` | Path to WRF run directory |
| `--case-name` | `fatima-20220720_20220723` | Name of the WRF case |
| `--file-prefix` | `wrfout_d01` | Prefix for WRF output files |
| `--proc-dir` | `./data` | Processing output directory |
| `--levs` | `np.arange(100, 2000, 100)` | Height levels expression |
| `--interp_var` | `geopotential_height` | Interpolation variable (geopotential_height or air_pressure) |
| `--logfile` | `./app.log` | Log file path |

### Output

The preprocessor creates a Zarr dataset at:
```
{proc_dir}/{case_name}/{file_prefix}_hlevs.zarr
```

## Processing Pipeline

The preprocessor performs the following steps:

1. **Data Loading**: Reads multiple WRF output files and concatenates them along the time dimension
2. **Postprocessing**: Applies WRF-specific corrections and destaggering
3. **Variable Computation**: Calculates derived meteorological variables
4. **Vertical Interpolation**: Regrids 3D variables to specified height or pressure levels
5. **Metadata Addition**: Embeds preprocessing parameters and timestamp
6. **Zarr Export**: Saves the processed dataset in efficient Zarr format

## Output Dataset

The processed dataset includes:
- **3D variables** interpolated to specified height/pressure levels
- **2D variables** (surface/near-surface fields)
- **Computed variables**: wind speed/direction, temperature, relative humidity
- **Metadata**: preprocessing parameters and timestamp

### Global Attributes
- `PREPROCESS_TIMESTAMP`: ISO timestamp of preprocessing
- `PREPROCESS_WRFRUN`: WRF run directory path
- `PREPROCESS_CASE_NAME`: Case name
- `PREPROCESS_FILE_PREFIX`: File prefix used
- `PREPROCESS_LEVS`: Height levels expression
- `PREPROCESS_INTERP_VAR`: Interpolation variable used

## Interactive Visualization

The `apps/` directory contains three interactive plotting applications:

### Height Level Plotting
```bash
python apps/make_plots_hlev.py ./data/wrf_hlevs.zarr --serve --web-port=8860
```

### Pressure Level Plotting
```bash
python apps/make_plots_plev.py ./data/wrf_hlevs.zarr --serve --web-port=8870
```

### Advanced Height Level Plotting
```bash
python apps/make_plots_hlev_advanced.py ./data/wrf_hlevs.zarr --serve --web-port=8880
```

### Remote Server Access
For headless/remote servers, use SSH tunneling:
```bash
# On your local machine
ssh -N -f -L localhost:8860:localhost:8860 user@server
# Then open http://localhost:8860 in your browser
```

All applications support:
- Interactive 2D/3D variable visualization
- Time animation controls
- Customizable colormaps and plot dimensions
- SSH tunneling for remote access
- Headless environment compatibility

## Data Variables

### 3D Variables (Height/Pressure Levels)
- `wind_east`, `wind_north`: Wind components (U, V)
- `air_potential_temperature`: Potential temperature (T)
- `QVAPOR`: Water vapor mixing ratio
- `air_pressure`: Pressure (P)
- `wind_speed`: Computed wind speed
- `wind_direction`: Computed wind direction
- `Ta`: Absolute temperature
- `RH`: Relative humidity

### 2D Variables (Surface)
- `T2`: 2m temperature
- `U10`, `V10`: 10m wind components
- `wind_speed_10`: 10m wind speed
- `wind_direction_10`: 10m wind direction

## Project Structure

```
fatima_wrf/
├── preprocess.py          # Main preprocessing script
├── tests/
│   ├── load_dataset.py    # Basic data loading test
│   └── __init__.py
├── AGENTS.md             # Development guidelines
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── __init__.py
└── .gitignore
```

## Testing

Run the basic test to verify data loading:

```bash
python tests/load_dataset.py
```

See [AGENTS.md](AGENTS.md) for detailed testing and development information.

## Contributing

1. Follow the code style guidelines in [AGENTS.md](AGENTS.md)
2. Add tests for new functionality
3. Update documentation as needed
4. Use meaningful commit messages

## License

[Add license information here]
