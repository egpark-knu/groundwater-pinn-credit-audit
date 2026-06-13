# Groundwater PINN Physical-Credit Audit

Code and analysis scripts for the groundwater forecasting physical-credit audit manuscript.

## Repository Contents

- `src/groundwater_research/`: reusable model, audit, baseline, data-quality, and virtual-aquifer modules.
- `scripts/`: command-line analysis and figure-generation scripts used for the manuscript experiments.
- `tests/`: pytest checks for core analysis paths.

Large result folders, raw NGMS/GIMS data, manuscript packages, PDFs, DOCX files, and local cache files are intentionally not committed.

## Data

Daily groundwater-level records are available from the Korean National Groundwater Monitoring System through the National Groundwater Information Center / Groundwater Information Management Service:

https://www.gims.go.kr

This repository does not redistribute the raw NGMS/GIMS datasets. Download the data from GIMS, then point the scripts at local data paths with environment variables.

## Path Configuration

The original analysis scripts use local data and result directories. Public copies are configured through environment variables:

```bash
export PINN_PROJECT_ROOT=/path/to/working/project
export NGMS_GROUNDWATER_ROOT=/path/to/gims/groundwater
export KOREA_GEODATA_ROOT=/path/to/korea/geodata
export MF6_EXE_CANDIDATES=/path/to/mf6
```

`MF6_EXE_CANDIDATES` may contain multiple paths separated by the platform path separator.

## Minimal Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some scripts require optional geospatial or MODFLOW dependencies and local data files.

## Example Entry Points

```bash
python scripts/select_50_wells.py
python scripts/run_50well_falsification.py
python scripts/run_50well_architecture.py
python scripts/analyze_nuisance_collapse.py
python scripts/generate_publication_figures.py
```

## Notes

This is an analysis-code release prepared ahead of manuscript submission. Results and manuscript artifacts are kept out of the public repository to avoid redistributing data or pushing large generated files.
