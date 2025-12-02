#!/usr/bin/env python3
"""
Advanced interactive plotting script for WRF data at height levels.

This script provides comprehensive visualization capabilities including:
- 2D and 3D spatial maps with enhanced styling
- Vertical cross-sections through the domain
- Time series analysis at specific locations
- Multi-variable comparison plots
- Custom colormaps, sizing, and interactive controls

Features:
- 5 different visualization modes in tabbed interface
- Interactive time controls and spatial selection
- Multiple colormap options (12+ choices)
- Adjustable plot dimensions
- SSH tunneling support for remote access

Usage:
    python make_plots_hlev_advanced.py <zarr_file_path>

Examples:
    python make_plots_hlev_advanced.py ./data/wrf_hlevs.zarr --serve --web-port=8880
    python make_plots_hlev_advanced.py ./data/wrf_hlevs.zarr --serve --force-serve
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
import cartopy.crs as ccrs
from dask import config as dask_config

# Configure dask for synchronous execution
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
    return None


def load_wrf_data(zarr_path):
    """Load WRF data from Zarr archive."""
    print(f"Loading WRF data from: {zarr_path}")
    ds = xr.open_zarr(zarr_path).sel(geopotential_height=slice(None, 300))
    print(f"Data loaded: {ds.dims}")
    return ds


def create_advanced_plots(ds, wrf_crs):
    """Create advanced interactive plotting interface with cross-sections and time series."""
    print("Setting up advanced plotting interface...")

    # Set up plotting defaults with enhanced styling
    gv.opts.defaults(
        gv.opts.Image(cmap='jet', width=500, height=400, colorbar=True, tools=['hover']),
        gv.opts.Curve(width=600, height=300, tools=['hover'], line_width=2),
        gv.opts.Scatter(width=600, height=300, tools=['hover'], size=6)
    )

    # Get data dimensions and variables
    vars_2d = [v for v in ds.data_vars if 'geopotential_height' not in ds[v].dims]
    vars_3d = [v for v in ds.data_vars if 'geopotential_height' in ds[v].dims]

    # Get time list and spatial coordinates
    t_list = ds.Time.values
    x_coords = ds.x.values
    y_coords = ds.y.values
    z_coords = ds.geopotential_height.values

    print(f"Found {len(t_list)} time steps, {len(vars_2d)} 2D variables, {len(vars_3d)} 3D variables")
    print(f"Spatial domain: {len(x_coords)}x{len(y_coords)} points, {len(z_coords)} height levels")

    # Define plotting functions
    def get_image_2d(var_name, time, **opts):
        ds_t = ds.sel(Time=time)
        # Use model coordinates with CRS - ensure aspect is always equal
        plot = gv.Image(ds_t[var_name], ['x','y'], crs=wrf_crs).opts(
        **opts,
        aspect='equal',
        xlim=(ds.x.min().values, ds.x.max().values),
        ylim=(ds.y.min().values, ds.y.max().values)
    )
        return plot

    def get_image_3d(var_name, time, hlev, **opts):
        ds_selected = ds[var_name].sel(Time=time, geopotential_height=hlev)
        # Use model coordinates with CRS - ensure aspect is always equal
        plot = gv.Image(ds_selected, ['x','y'], crs=wrf_crs).opts(
        **opts,
        aspect='equal',
        xlim=(ds.x.min().values, ds.x.max().values),
        ylim=(ds.y.min().values, ds.y.max().values)
    )
        return plot

    # Advanced plotting functions
    def get_vertical_cross_section(var_name, time, x_idx, orientation='NS', **opts):
        """Create vertical cross-section along x or y axis."""
        ds_t = ds.sel(Time=time)
        if orientation == 'NS':  # North-South cross-section
            cross_section = ds_t[var_name].isel(x=x_idx)
            coords = ['y', 'geopotential_height']
            title = f'NS Cross-section at x={x_coords[x_idx]:.0f}'
        else:  # East-West cross-section
            cross_section = ds_t[var_name].isel(y=x_idx)  # Using x_idx for consistency
            coords = ['x', 'geopotential_height']
            title = f'EW Cross-section at y={y_coords[x_idx]:.0f}'

        plot = gv.Image(cross_section, coords, crs=wrf_crs).opts(title=title, **opts)
        return plot

    def get_time_series(var_name, x_idx, y_idx, hlev_idx=None, **opts):
        """Create time series at a specific location."""
        if hlev_idx is not None:
            # 3D variable time series
            data = ds[var_name].isel(x=x_idx, y=y_idx, geopotential_height=hlev_idx)
            title = f'Time Series: {var_name} at (x={x_coords[x_idx]:.0f}, y={y_coords[y_idx]:.0f}, z={z_coords[hlev_idx]:.0f}m)'
        else:
            # 2D variable time series
            data = ds[var_name].isel(x=x_idx, y=y_idx)
            title = f'Time Series: {var_name} at (x={x_coords[x_idx]:.0f}, y={y_coords[y_idx]:.0f})'

        plot = hv.Curve((t_list, data.values), 'Time', var_name, label=title).opts(**opts)
        return plot

    def get_multi_variable_comparison(var1, var2, time, hlev, **opts):
        """Compare two variables side by side."""
        plot1 = get_image_3d(var1, time, hlev, **opts).opts(title=f'{var1}', aspect='equal')
        plot2 = get_image_3d(var2, time, hlev, **opts).opts(title=f'{var2}', aspect='equal')
        return (plot1 + plot2).opts(shared_axes=False)

    # Create interactive widgets with enhanced options
    cmap_list = ['jet', 'nipy_spectral', 'gnuplot', 'gist_rainbow', 'turbo', 'gist_ncar',
                'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm']
    cmap_widget = pn.widgets.Select(name='Colormap', options=cmap_list, value='jet', width=150)
    time_widget = pn.widgets.DiscretePlayer(name="Time", options=list(t_list), value=t_list[0], width=300)
    width_widget = pn.widgets.IntInput(name='Width', value=500, step=50, start=200, end=1200, width=100)
    height_widget = pn.widgets.IntInput(name='Height', value=400, step=50, start=200, end=800, width=100)

    # Additional widgets for advanced features
    x_cross_widget = pn.widgets.IntSlider(name='X Cross-section', start=0, end=len(x_coords)-1,
                                         value=len(x_coords)//2, width=200)
    y_cross_widget = pn.widgets.IntSlider(name='Y Cross-section', start=0, end=len(y_coords)-1,
                                         value=len(y_coords)//2, width=200)
    orientation_widget = pn.widgets.Select(name='Orientation', options=['NS', 'EW'], value='NS', width=80)

    # Time series widgets
    ts_x_widget = pn.widgets.IntSlider(name='TS X', start=0, end=len(x_coords)-1,
                                      value=len(x_coords)//2, width=200)
    ts_y_widget = pn.widgets.IntSlider(name='TS Y', start=0, end=len(y_coords)-1,
                                      value=len(y_coords)//2, width=200)

    # ===== BASIC PLOTTING INTERFACES =====

    # 3D plotting interface
    z_widget = pn.widgets.Select(name='Height Level', value=100,
                                options=list(ds.geopotential_height.values), width=120)
    var3d_select = pn.widgets.Select(name="3D Variable", options=vars_3d,
                                   value='wind_speed', width=150)

    plot_3d = pn.bind(get_image_3d, var3d_select, time_widget, z_widget,
                     width=width_widget, height=height_widget, cmap=cmap_widget)

    app_3d = pn.Column(
        pn.pane.Markdown("## 3D Variables (Height Levels)"),
        plot_3d,
        pn.Row(
            pn.Column(pn.pane.Markdown("### Controls"), time_widget, var3d_select, z_widget),
            pn.Column(pn.pane.Markdown("### Style"), width_widget, height_widget, cmap_widget)
        )
    )

    # 2D plotting interface
    var2d_select = pn.widgets.Select(name="2D Variable", options=vars_2d,
                                   value=vars_2d[0], width=150)

    plot_2d = pn.bind(get_image_2d, var2d_select, time_widget,
                     width=width_widget, height=height_widget, cmap=cmap_widget)

    app_2d = pn.Column(
        pn.pane.Markdown("## 2D Variables (Surface)"),
        plot_2d,
        pn.Row(
            pn.Column(pn.pane.Markdown("### Controls"), time_widget, var2d_select),
            pn.Column(pn.pane.Markdown("### Style"), width_widget, height_widget, cmap_widget)
        )
    )

    # ===== ADVANCED FEATURES =====

    # Cross-section plotting (simplified - using fixed NS orientation for now)
    var_cross_select = pn.widgets.Select(name="Cross-section Variable", options=vars_3d,
                                        value='wind_speed', width=150)

    plot_cross = pn.bind(get_vertical_cross_section, var_cross_select, time_widget,
                        x_cross_widget, orientation_widget,
                        width=width_widget, height=height_widget, cmap=cmap_widget)

    app_cross = pn.Column(
        pn.pane.Markdown("## Vertical Cross-Sections"),
        pn.pane.Markdown("*North-South cross-sections through the domain*"),
        plot_cross,
        pn.Row(
            pn.Column(pn.pane.Markdown("### Controls"),
                     time_widget, var_cross_select, x_cross_widget),
            pn.Column(pn.pane.Markdown("### Style"), width_widget, height_widget, cmap_widget)
        )
    )

    # Time series analysis
    ts_var_select = pn.widgets.Select(name="Time Series Variable", options=vars_3d + vars_2d,
                                     value='wind_speed', width=150)
    ts_z_widget = pn.widgets.Select(name='Height Level (3D vars only)', value=100,
                                   options=list(ds.geopotential_height.values), width=140)

    # Create dynamic height index based on whether variable is 2D or 3D
    def get_height_index(var_name, hlev):
        if var_name in vars_3d:
            return list(ds.geopotential_height.values).index(hlev)
        return None

    plot_ts = pn.bind(get_time_series, ts_var_select, ts_x_widget, ts_y_widget,
                     pn.bind(get_height_index, ts_var_select, ts_z_widget),
                     width=width_widget, height=height_widget)

    app_ts = pn.Column(
        pn.pane.Markdown("## Time Series Analysis"),
        pn.pane.Markdown("*Click on the map or use sliders to select a location*"),
        plot_ts,
        pn.Row(
            pn.Column(pn.pane.Markdown("### Location"), ts_x_widget, ts_y_widget, ts_z_widget),
            pn.Column(pn.pane.Markdown("### Variable"), ts_var_select, time_widget),
            pn.Column(pn.pane.Markdown("### Style"), width_widget, height_widget)
        )
    )

    # Multi-variable comparison
    var1_select = pn.widgets.Select(name="Variable 1", options=vars_3d, value='wind_speed', width=120)
    var2_select = pn.widgets.Select(name="Variable 2", options=vars_3d, value='Ta', width=120)

    plot_compare = pn.bind(get_multi_variable_comparison, var1_select, var2_select,
                          time_widget, z_widget, width=width_widget, height=height_widget, cmap=cmap_widget)

    app_compare = pn.Column(
        pn.pane.Markdown("## Multi-Variable Comparison"),
        pn.pane.Markdown("*Compare two 3D variables side by side*"),
        plot_compare,
        pn.Row(
            pn.Column(pn.pane.Markdown("### Variables"), var1_select, var2_select),
            pn.Column(pn.pane.Markdown("### Controls"), time_widget, z_widget),
            pn.Column(pn.pane.Markdown("### Style"), width_widget, height_widget, cmap_widget)
        )
    )

    # Create main tabs
    print(f"Creating advanced interface with {len(vars_2d)} 2D and {len(vars_3d)} 3D variables...")
    tabs = pn.Tabs(
        ("2D Maps", app_2d),
        ("3D Maps", app_3d),
        ("Cross-Sections", app_cross),
        ("Time Series", app_ts),
        ("Comparisons", app_compare)
    )
    print("Advanced plotting interface created successfully!")

    return tabs


def main():
    parser = argparse.ArgumentParser(description="Advanced interactive WRF plotting at height levels")
    parser.add_argument("zarr_file", help="Path to Zarr file containing WRF data")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of dask workers")
    parser.add_argument("--serve", action="store_true", help="Serve the web interface")
    parser.add_argument("--web-port", type=int, default=5008, help="Port for web interface")
    parser.add_argument("--force-serve", action="store_true", help="Force serving even in headless environments (may hang)")

    args = parser.parse_args()

    # Setup dask cluster
    print(f"Setting up dask cluster with {args.n_workers} workers...")
    client = setup_dask_cluster(args.n_workers)
    print("Dask cluster ready")

    try:
        # Load data
        print("Loading WRF data...")
        ds = load_wrf_data(args.zarr_file)
        print("Data loaded successfully")

        # Extract coordinate system
        print("Extracting coordinate system...")
        wrf_crs = get_crs(ds)
        print("CRS extracted successfully")

        # Create advanced plotting interface
        print("Creating advanced plotting interface...")
        app = create_advanced_plots(ds, wrf_crs)

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
                print("from apps.make_plots_hlev_advanced import main")
                print("import sys")
                print()
                print("# Set up arguments")
                print(f"sys.argv = ['make_plots_hlev_advanced.py', '{args.zarr_file}', '--n-workers=1']")
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
            print("You can run: pn.serve(app, port=5008, show=False)")

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