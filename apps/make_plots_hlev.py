#!/usr/bin/env python3
"""
Interactive plotting script for WRF data at height levels.

This script creates interactive visualizations of WRF model output data
interpolated to specified height levels.

Usage:
    python make_plots_hlev.py <zarr_file_path>

Example:
    python make_plots_hlev.py ./data/fatima-20220720_20220723/wrfout_d01_hlevs.zarr
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
        standard_parallels=(wrf_crs_dict["lat_1"], wrf_crs_dict["lat_2"]),
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
    ds = xr.open_zarr(zarr_path)
    print(f"Data loaded: {ds.dims}")
    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Interactive WRF plotting at height levels"
    )
    parser.add_argument("zarr_file", help="Path to Zarr file containing WRF data")
    parser.add_argument(
        "--n-workers", type=int, default=4, help="Number of dask workers"
    )
    parser.add_argument("--port", type=int, default=8787, help="Dask dashboard port")
    parser.add_argument("--serve", action="store_true", help="Serve the web interface")
    parser.add_argument(
        "--web-port", type=int, default=5006, help="Port for web interface"
    )
    parser.add_argument(
        "--force-serve",
        action="store_true",
        help="Force serving even in headless environments (may hang)",
    )

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
        print(f"CRS type: {type(wrf_crs)}")

        # Get data dimensions and variables
        print("Categorizing variables...")
        vars_2d = [v for v in ds.data_vars if "geopotential_height" not in ds[v].dims]
        vars_3d = [v for v in ds.data_vars if "geopotential_height" in ds[v].dims]
        print(f"Found {len(vars_2d)} 2D and {len(vars_3d)} 3D variables")

        # Create interactive plotting interface
        print("Creating interactive plotting interface...")
        app = create_interactive_plots(ds, wrf_crs, vars_2d, vars_3d)
        print("Interactive plotting interface created successfully")

        # Serve web interface if requested
        if args.serve:
            print(f"Web interface will be served on port {args.web_port}")
            print(f"Open your browser to: http://localhost:{args.web_port}")
            print()
            print("If running on a remote server, set up SSH tunneling first:")
            print(
                f"ssh -N -f -L localhost:{args.web_port}:localhost:{args.web_port} user@server"
            )
            print(f"Then open: http://localhost:{args.web_port} in your local browser")
            print()
            print("Starting server... (press Ctrl+C to stop)")

            # Serve the web interface
            import os

            is_headless = not (
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            )

            if is_headless and not args.force_serve:
                print("Detected headless environment.")
                print("Application created successfully!")
                print("To start the web server manually, run these commands in Python:")
                print("```python")
                print("import panel as pn")
                print("from apps.make_plots_hlev import main")
                print("import sys")
                print()
                print("# Set up arguments")
                print(
                    f"sys.argv = ['make_plots_hlev.py', '{args.zarr_file}', '--n-workers=1']"
                )
                print()
                print("# Create the app (this will set the global 'app' variable)")
                print("main()")
                print()
                print("# Start the server")
                print(f"pn.serve(app, port={args.web_port}, show=False)")
                print("```")
                print()
                print("For SSH tunneling, run on your local machine:")
                print(
                    f"ssh -N -f -L localhost:{args.web_port}:localhost:{args.web_port} user@server"
                )
                print(f"Then open: http://localhost:{args.web_port}")
                print()
                print("Or use --force-serve to attempt automatic serving.")
                return
            elif is_headless and args.force_serve:
                print("Detected headless environment but --force-serve specified.")
                print("Attempting to serve automatically...")
                print("Note: This may hang in headless environments.")
                print("Use SSH tunneling instead for better reliability.")

            # Serve the application
            try:
                show_browser = not is_headless
                print(
                    f"Starting Panel server on port {args.web_port} (show_browser={show_browser})..."
                )
                pn.serve(app, port=args.web_port, show=show_browser)
                print("Server started successfully!")
            except KeyboardInterrupt:
                print("Server stopped by user")
            except Exception as e:
                print(f"Error starting web server: {e}")
                if is_headless:
                    print(
                        "In headless environment, try the manual approach shown above."
                    )
                else:
                    print("Try the manual approach shown above.")
                print("Try manual serving as shown above.")
        else:
            print("Application created successfully!")
            print("To start the web interface, run with --serve flag")
            print(
                f"Example: python {sys.argv[0]} {args.zarr_file} --serve --web-port={args.web_port}"
            )
            print()
            print(
                "For manual server control, the 'app' variable is available in this scope."
            )
            print("You can run: pn.serve(app, port=8860, show=False)")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if client is not None:
            print("Closing dask client...")
            client.close()
        print("Done")


def create_interactive_plots(ds, wrf_crs, vars_2d, vars_3d):
    """Create interactive plotting interface."""
    print("Setting up interactive plotting interface...")
    # Set up plotting defaults
    gv.opts.defaults(gv.opts.Image(cmap="jet", width=400, colorbar=True))
    print("Plotting defaults set")

    # Get time list
    t_list = ds.Time.values
    print(f"Found {len(t_list)} time steps")
    print(f"Sample time: {t_list[0]}")

    # Define plotting functions
    def get_image_2d(var_name, time, **opts):
        ds_t = ds.sel(Time=time)
        plot = gv.Image(ds_t[var_name], ["x", "y"], crs=wrf_crs).opts(**opts)
        return plot

    def get_image_3d(var_name, time, hlev, **opts):
        return gv.Image(
            ds[var_name].sel(Time=time, geopotential_height=hlev),
            ["x", "y"],
            crs=wrf_crs,
        ).opts(**opts)

    # Create interactive widgets
    cmap_list = [
        "jet",
        "nipy_spectral",
        "gnuplot",
        "gist_rainbow",
        "turbo",
        "gist_ncar",
    ]
    cmap_widget = pn.widgets.Select(
        name="cmap", options=cmap_list, value="jet", width=200
    )
    time_widget = pn.widgets.DiscretePlayer(
        name="time", options=list(t_list), value=t_list[0]
    )
    width_widget = pn.widgets.IntInput(name="width", value=450, step=20, width=200)
    height_widget = pn.widgets.IntInput(name="height", value=300, step=20, width=200)

    # 3D plotting interface
    z_widget = pn.widgets.Select(
        name="Z", value=100, options=list(ds.geopotential_height.values), width=100
    )
    var3d_select = pn.widgets.Select(
        description="var3d", options=vars_3d, value="wind_speed", width=100
    )

    plot_3d = pn.bind(
        get_image_3d,
        var3d_select,
        time_widget,
        z_widget,
        width=width_widget,
        height=height_widget,
        cmap=cmap_widget,
    )

    app_3d = pn.Column(
        "## 3D Variables",
        plot_3d,
        pn.Row(time_widget, pn.Column(var3d_select, z_widget)),
        pn.Row(width_widget, height_widget, cmap_widget),
    )

    # 2D plotting interface
    var2d_select = pn.widgets.Select(
        description="var2d", options=vars_2d, value=vars_2d[0], width=100
    )

    plot_2d = pn.bind(
        get_image_2d,
        var2d_select,
        time_widget,
        width=width_widget,
        height=height_widget,
        cmap=cmap_widget,
    )

    app_2d = pn.Column(
        "## 2D Variables",
        plot_2d,
        pn.Row(time_widget, var2d_select),
        pn.Row(width_widget, height_widget, cmap_widget),
    )

    # Create tabs for 2D and 3D plotting
    print(f"Creating tabs with {len(vars_2d)} 2D and {len(vars_3d)} 3D variables...")
    tabs = pn.Tabs(("2D Variables", app_2d), ("3D Variables", app_3d))
    print("Tabs created successfully")

    # Return the application
    print("Interactive plotting application created successfully!")
    return tabs

    # Define plotting functions
    def get_image_2d(var_name, time, **opts):
        ds_t = ds.sel(Time=time)
        plot = gv.Image(ds_t[var_name], ["x", "y"], crs=wrf_crs).opts(**opts)
        return plot

    def get_image_3d(var_name, time, hlev, **opts):
        return gv.Image(
            ds[var_name].sel(Time=time, geopotential_height=hlev),
            ["x", "y"],
            crs=wrf_crs,
        ).opts(**opts)

    # Create interactive widgets
    cmap_list = [
        "jet",
        "nipy_spectral",
        "gnuplot",
        "gist_rainbow",
        "turbo",
        "gist_ncar",
    ]
    cmap_widget = pn.widgets.Select(
        name="cmap", options=cmap_list, value="jet", width=200
    )
    time_widget = pn.widgets.DiscretePlayer(
        name="time", options=list(t_list), value=t_list[0]
    )
    width_widget = pn.widgets.IntInput(name="width", value=450, step=20, width=200)
    height_widget = pn.widgets.IntInput(name="height", value=300, step=20, width=200)

    # 3D plotting interface
    z_widget = pn.widgets.Select(
        name="Z", value=100, options=list(ds.geopotential_height.values), width=100
    )
    var3d_select = pn.widgets.Select(
        description="var3d", options=vars_3d, value="wind_speed", width=100
    )

    plot_3d = pn.bind(
        get_image_3d,
        var3d_select,
        time_widget,
        z_widget,
        width=width_widget,
        height=height_widget,
        cmap=cmap_widget,
    )

    app_3d = pn.Column(
        plot_3d,
        pn.Row(time_widget, pn.Column(var3d_select, z_widget)),
        pn.Row(width_widget, height_widget, cmap_widget),
    )

    # 2D plotting interface
    var2d_select = pn.widgets.Select(
        description="var2d", options=vars_2d, value=vars_2d[0], width=100
    )

    plot_2d = pn.bind(
        get_image_2d,
        var2d_select,
        time_widget,
        width=width_widget,
        height=height_widget,
        cmap=cmap_widget,
    )

    app_2d = pn.Column(
        plot_2d,
        pn.Row(time_widget, var2d_select),
        pn.Row(width_widget, height_widget, cmap_widget),
    )

    # Create tabs for 2D and 3D plotting
    print(f"Creating tabs with {len(vars_2d)} 2D and {len(vars_3d)} 3D variables...")
    tabs = pn.Tabs(("2D Variables", app_2d), ("3D Variables", app_3d))
    print("Tabs created successfully")

    # Serve the application
    print("Starting interactive plotting application...")
    print("Open your browser to the displayed URL")
    print(
        "Note: If running on a headless server, use --show=False and access via SSH tunnel"
    )

    # Check if we have a display
    import os

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        pn.serve(tabs, port=5006, show=True)
    else:
        print("No display detected. Starting server in background...")
        print("Access the application at: http://localhost:5006")
        server = pn.serve(tabs, port=5006, show=False, threaded=True)
        print("Server started. Press Ctrl+C to stop.")
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping server...")
            server.stop()


if __name__ == "__main__":
    main()
