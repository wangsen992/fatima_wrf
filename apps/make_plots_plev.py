#!/usr/bin/env python3
"""
Interactive plotting script for WRF data at pressure levels.

This script creates interactive visualizations of WRF model output data
interpolated to specified pressure levels.

Usage:
    python make_plots_plev.py <zarr_file_path>

Example:
    python make_plots_plev.py ./data/fatima-20220720_20220723/wrfout_d01_plevs.zarr
"""

import argparse
import pathlib
import sys
import numpy as np
import xarray as xr
import holoviews as hv
import geoviews as gv
import geoviews.feature as gf
import panel as pn
from cartopy import crs
from dask import config as dask_config

# Configure dask for synchronous execution (simpler for debugging)
dask_config.set(scheduler="synchronous")

# Initialize plotting extensions
gv.extension("bokeh")
pn.extension()

# Import additional libraries
import xwrf
from distributed import LocalCluster, Client


def get_crs(ds):
    """Extract coordinate reference system from WRF dataset."""
    # wrf_projection is a scalar, so no indexing needed
    wrf_crs = crs.CRS(ds.wrf_projection.compute().item())
    wrf_crs_dict = wrf_crs.to_dict()
    new_crs = crs.LambertConformal(
        central_longitude=wrf_crs_dict["lon_0"],
        central_latitude=wrf_crs_dict["lat_0"],
        standard_parallels=(wrf_crs_dict["lat_1"], wrf_crs_dict["lat_2"])
    )
    return new_crs


def setup_dask_cluster(n_workers=4):
    """Set up a local dask cluster for parallel processing."""
    print(f"Using synchronous scheduler (n_workers={n_workers} ignored)")
    # For now, just return None since we're using synchronous scheduler
    return None


def load_wrf_data(zarr_path):
    """Load WRF data from Zarr archive."""
    print(f"Loading WRF data from: {zarr_path}")
    ds = xr.open_zarr(zarr_path).persist()
    return ds


def create_pressure_level_plots(ds, wrf_crs):
    """Create interactive plotting interface for pressure level data."""
    print("Setting up interactive plotting interface...")
    # Set up plotting defaults
    gv.opts.defaults(gv.opts.Image(cmap='jet', width=400, colorbar=True))
    print("Plotting defaults set")

    # Check if we have pressure-level data or height-level data
    if 'pressure' in ds.dims:
        # True pressure-level data
        dims_3d = [v for v in ds.dims if 'pressure' in ds.dims]
        vars_3d = [v for v in ds.data_vars if 'pressure' in ds[v].dims]
        p_levels = ds.pressure.values
        level_name = "pressure"
        level_unit = "hPa"
    elif 'air_pressure' in ds.dims:
        # Pressure-level data with air_pressure as coordinate
        dims_3d = [v for v in ds.dims if 'air_pressure' in ds.dims]
        vars_3d = [v for v in ds.data_vars if 'air_pressure' in ds[v].dims]
        p_levels = ds.air_pressure.values / 100  # Convert Pa to hPa
        level_name = "pressure"
        level_unit = "hPa"
        print(f"Using pressure-level data: {len(p_levels)} levels")
    elif 'geopotential_height' in ds.dims and 'air_pressure' in ds.data_vars:
        # Height-level data with pressure variable - use pressure values as levels
        dims_3d = [v for v in ds.dims if 'geopotential_height' in ds.dims]
        vars_3d = [v for v in ds.data_vars if 'geopotential_height' in ds[v].dims]
        # Use mean pressure values across domain as level labels
        p_levels = ds.air_pressure.mean(dim=['x', 'y']).isel(Time=0).values / 100  # Convert to hPa
        level_name = "pressure (approx)"
        level_unit = "hPa"
        print(f"Using height-level data with approximate pressure levels: {len(p_levels)} levels")
    else:
        raise ValueError("Dataset must have either 'pressure' coordinate, 'air_pressure' coordinate, or 'air_pressure' variable")

    # Get time list
    t_list = ds.Time.values

    # Define plotting functions
    def get_image_3d(var_name, time, plev, **opts):
        if 'pressure' in ds.dims:
            return gv.Image(ds[var_name].sel(Time=time, pressure=plev),
                           ['x','y'], crs=wrf_crs).opts(**opts)
        elif 'air_pressure' in ds.dims:
            # For pressure-level data with air_pressure coordinate
            # Convert plev back to Pa for selection
            plev_pa = plev * 100  # Convert hPa back to Pa
            return gv.Image(ds[var_name].sel(Time=time, air_pressure=plev_pa),
                           ['x','y'], crs=wrf_crs).opts(**opts)
        else:
            # For height-level data, find the closest height level to the selected pressure
            # Use sel instead of isel for time selection
            pressure_values = ds.air_pressure.mean(dim=['x', 'y']).sel(Time=time).values / 100  # hPa
            height_idx = abs(pressure_values - plev).argmin()
            height_value = ds.geopotential_height.values[height_idx]
            return gv.Image(ds[var_name].sel(Time=time, geopotential_height=height_value),
                           ['x','y'], crs=wrf_crs).opts(**opts)

    # Create interactive widgets
    cmap_list = ['jet', 'nipy_spectral', 'gnuplot', "gist_rainbow", 'turbo', 'gist_ncar']
    cmap_widget = pn.widgets.Select(name='cmap', options=cmap_list, value='jet', width=200)
    time_widget = pn.widgets.DiscretePlayer(name="time", options=list(t_list), value=t_list[0])
    width_widget = pn.widgets.IntInput(name='width', value=450, step=20, width=200)
    height_widget = pn.widgets.IntInput(name='height', value=300, step=20, width=200)

    # Pressure/height level plotting interface
    level_options = [f"{p:.0f} {level_unit}" for p in p_levels]
    level_values = list(p_levels)
    default_level = level_values[len(level_values)//2]  # Middle level

    level_widget = pn.widgets.Select(name=level_name.title(), value=default_level,
                                    options=level_values, width=100)
    var3d_select = pn.widgets.Select(description="var3d", options=vars_3d,
                                   value=vars_3d[0], width=100)

    plot_3d = pn.bind(get_image_3d, var3d_select, time_widget, level_widget,
                     width=width_widget, height=height_widget, cmap=cmap_widget)

    app = pn.Column(f"## 3D Variables ({level_name})", plot_3d,
                   pn.Row(time_widget, pn.Column(var3d_select, level_widget)),
                   pn.Row(width_widget, height_widget, cmap_widget))

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive WRF plotting at pressure levels")
    parser.add_argument("zarr_file", help="Path to Zarr file containing WRF data")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of dask workers")
    parser.add_argument("--serve", action="store_true", help="Serve the web interface")
    parser.add_argument("--web-port", type=int, default=5007, help="Port for web interface")
    parser.add_argument("--force-serve", action="store_true", help="Force serving even in headless environments (may hang)")

    args = parser.parse_args()

    # Setup dask cluster
    client = setup_dask_cluster(args.n_workers)

    try:
        print(f"Starting plev plotting script with file: {args.zarr_file}")
        # Load data
        print("Loading WRF data...")
        ds = load_wrf_data(args.zarr_file)
        print("Data loaded successfully")

        # Extract coordinate system
        print("Extracting coordinate system...")
        wrf_crs = get_crs(ds)
        print("CRS extracted successfully")

        # Create interactive plots
        print("Creating interactive plotting interface...")
        app = create_pressure_level_plots(ds, wrf_crs)
        print("Interactive plotting interface created successfully")

        # Handle web interface
        if args.serve:
            print(f"Web interface will be served on port {args.web_port}")
            print(f"Open your browser to: http://localhost:{args.web_port}")
            print()
            print("If running on a remote server, set up SSH tunneling first:")
            print(f"ssh -N -f -L localhost:{args.web_port}:localhost:{args.web_port} user@server")
            print(f"Then open: http://localhost:{args.web_port} in your local browser")
            print()
            print("Starting server... (press Ctrl+C to stop)")

            # Serve the web interface
            import os
            is_headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

            if is_headless and not args.force_serve:
                print("Detected headless environment.")
                print("Application created successfully!")
                print("To start the web server manually, run these commands in Python:")
                print("```python")
                print("import panel as pn")
                print("from apps.make_plots_plev import main")
                print("import sys")
                print()
                print("# Set up arguments")
                print(f"sys.argv = ['make_plots_plev.py', '{args.zarr_file}', '--n-workers=1']")
                print()
                print("# Create the app (this will set the global 'app' variable)")
                print("main()")
                print()
                print("# Start the server")
                print(f"pn.serve(app, port={args.web_port}, show=False)")
                print("```")
                print()
                print("Or use SSH tunneling to access an existing server.")
                return
            elif is_headless and args.force_serve:
                print("Detected headless environment but --force-serve specified.")
                print("Attempting to serve automatically...")

            # Handle web serving
            print(f"Environment check: is_headless={is_headless}, force_serve={args.force_serve}")
            if is_headless and not args.force_serve:
                print("Detected headless environment.")
                print("Application created successfully!")
                print("To start the web server manually, run these commands in Python:")
                print("```python")
                print("import panel as pn")
                print("from apps.make_plots_plev import main")
                print("import sys")
                print()
                print("# Set up arguments")
                print(f"sys.argv = ['make_plots_plev.py', '{args.zarr_file}', '--n-workers=1']")
                print()
                print("# Create the app (this will set the global 'app' variable)")
                print("main()")
                print()
                print("# Start the server")
                print(f"pn.serve(app, port={args.web_port}, show=False)")
                print("```")
                print()
                print("For SSH tunneling, run on your local machine:")
                print(f"ssh -N -f -L localhost:{args.web_port}:localhost:{args.web_port} user@server")
                print(f"Then open: http://localhost:{args.web_port}")
                print()
                return
            elif is_headless and args.force_serve:
                print("Detected headless environment but --force-serve specified.")
                print("Attempting to serve automatically...")

            # Serve the application
            try:
                show_browser = not is_headless
                print(f"Starting Panel server on port {args.web_port} (show_browser={show_browser})...")
                pn.serve(app, port=args.web_port, show=show_browser)
                print("Server started successfully!")
            except KeyboardInterrupt:
                print("Server stopped by user")
            except Exception as e:
                print(f"Error starting web server: {e}")
                if is_headless:
                    print("In headless environment, try the manual approach shown above.")
                else:
                    print("Try the manual approach shown above.")
        else:
            print("Application created successfully!")
            print("To start the web interface, run with --serve flag")
            print(f"Example: python {sys.argv[0]} {args.zarr_file} --serve --web-port={args.web_port}")
            print()
            print("For manual server control, the 'app' variable is available in this scope.")
            print("You can run: pn.serve(app, port=5007, show=False)")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client is not None:
            print("Closing dask client...")
            client.close()
        print("Done")


if __name__ == "__main__":
    main()
