# AGENTS.md - Development Guidelines for FATIMA WRF Postprocessing

## Build/Lint/Test Commands

### Testing
- **Run all tests**: `python -m pytest tests/` (pytest not currently installed)
- **Run single test**: `python tests/load_dataset.py` (manual test execution)
- **Environment setup**: Use conda environment from `requirements.txt`

### Linting & Code Quality
- No linting tools currently configured (flake8, ruff, black not available)
- Recommended: Install ruff for fast Python linting and formatting

### Build Commands
- No build system configured (no setup.py, pyproject.toml, or Makefile)
- Environment managed via conda: `conda env create -f requirements.txt`

### Running the Preprocessor
- Basic usage: `python preprocess.py`
- With custom paths: `python preprocess.py --wrf-run /path/to/wrf/run --case-name my_case`
- Custom height levels: `python preprocess.py --hlevs "np.arange(50, 500, 25)"`
- Parallel processing: `python preprocess.py --parallel`
- Custom logging: `python preprocess.py --logfile my_log.txt`

## Code Style Guidelines

### Imports
- Standard library imports first, then third-party imports
- Use explicit imports: `from pathlib import Path` instead of `import pathlib`
- Group imports logically with blank lines between groups
- Use type imports from typing module: `from typing import Union, List`

### Naming Conventions
- **Functions**: snake_case (e.g., `read_wrfout`, `to_hlevs`)
- **Variables**: snake_case (e.g., `ds_raw`, `file_prefix`)
- **Classes**: PascalCase (none currently defined)
- **Constants**: UPPER_CASE (none currently defined)
- **Files**: snake_case with .py extension

### Formatting & Style
- Use 4 spaces for indentation (Python standard)
- Line length: No explicit limit, but keep readable
- Use f-strings for string formatting: `f"Reading wrfout files: {paths}"`
- Use pathlib.Path for file/directory operations
- Add docstrings to all functions with triple quotes

### Type Hints
- Use type hints for function parameters and return values
- Use Union types for multiple possible types: `Union[list, np.ndarray]`
- Use List type hints for collections: `list[Path]`

### Error Handling
- Use logging extensively for debugging and monitoring
- Configure logging with timestamps and function names
- Avoid bare except clauses; catch specific exceptions
- Use logging levels appropriately (DEBUG, INFO, WARNING, ERROR)

### Code Structure
- Keep functions focused on single responsibilities
- Use descriptive variable names
- Avoid code duplication (refactor repeated computations)
- Use list/dict comprehensions for concise data transformations
- Prefer functional programming patterns where appropriate

### Testing
- Place tests in `tests/` directory
- Use relative imports for testing local modules
- Test files should be executable as standalone scripts
- Include logging configuration in test files

### Configuration
- Use argparse for all configuration with sensible defaults
- No external config files required
- All parameters can be overridden via command-line arguments</content>
<parameter name="filePath">/scratch365/swang18/Workspace/Projects/FATIMA_Darko/fatima_wrf/AGENTS.md