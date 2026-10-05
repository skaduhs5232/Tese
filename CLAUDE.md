# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a master's thesis (Tese, in Portuguese) research pipeline that tests whether the population of a drainage basin correlates with the microplastic found at its river mouth. It cross-references three independent geospatial datasets over North America down to the Panama Canal:

- **GEBCO 2024** — global relief/bathymetry raster → used to derive drainage basins.
- **GPW v4** — global population-count raster (2020) → used to derive urban blobs and per-basin population.
- **NOAA Marine Microplastics** (`dados/noaa_microplastics.csv`, Nurdle Patrol only, unit `pieces/10 mins`) → microplastic per basin.

The project is microplastic-only and US/North-America-only: the Brazil analysis and the macroplastic (WAW / MDMAP) pipeline were removed.

Read `README.md` first — it documents, in Portuguese, the reasoning behind every methodological decision (coastal sinks instead of `pysheds`, the crop and GEBCO coarsening, CCA parameters calibrated by Zipf's law, splitting urban blobs across basins, nearest-sampling-site-per-mouth with a 1 km radius sweep, basin-level allometry). `tasks.txt` is the terse, authoritative statement of the methodology. Treat it as the spec; `README.md` is the explanation of it.

## Running the notebooks

**The notebooks only run on Google Colab** — they assume GDAL and other heavy geospatial deps are already present in the Colab runtime (numba included), install a few extra packages inline (`rioxarray`, `powerlaw`), download data with `gdown`, and `git clone` fresh copies of IPBA and CCA at runtime. They are not adapted to run locally. When editing notebook cells, validate by reading/checking cell syntax and logic — do not attempt to execute the full pipeline in this local environment (small synthetic tests of individual functions are fine).

Notebooks live in `notebooks/` and run in order:

1. **`01_Zipf_calibracao_CCA.ipynb`** — sweeps `D*` = 10…3000 hab/km² (step 10) × `ℓ` = 100…5000 m (step 100) over the analysis crop, builds the matrix of Zipf exponents α (Hill estimator on the top N = 50/100/200 blobs), picks the (`D*`, `ℓ`) with `ℓ ≥ L_MIN` (1400 m, above the GPW pixel diagonal) minimizing Δ = max_N |α_N − 1|, cross-checks with a CSN fit, verifies it against `cca_grid`, and writes `results/cca_calibracao.json`. The sweep is a numba union-find that activates pixels by decreasing density and snapshots clusters at each `D*`; it must stay equivalent to `cca_grid` (density `> D*`, haversine distance `< ℓ`).
2. **`02_Micro_Plastic.ipynb`** — the main pipeline:
   1. Crop `(-125, 7, -65, 50)`; South America (lon > −79, lat < 12.6) masked as ocean / zero population. GEBCO coarsened 8× (2′) because IPBA is pure Python and stores sink labels as `uint16` (< 65,535 sinks). Ocean = elevation ≤ 0 connected to the crop edge.
   2. Coastal-sink IPBA basins; keep basins covering 90% of the area (`P = 0.90`).
   3. CCA urban blobs with `D*`/`ℓ` read from `results/cca_calibracao.json` (falls back to the U.S. Census 386 hab/km², 3 km with a warning).
   4. Per-basin population; blobs spanning several basins are split, each piece counted in its own basin (`pedacos`).
   5. Each basin gets the nearest Nurdle Patrol site to its mouth pixel (concentration, and concentration × basin area); acceptance radius swept 1…50 km in 1 km steps; `R_MAIORIA` = smallest radius containing ≥ 50% of sites.
   6. Basin-level allometry (log-log scaling fit, bootstrap CI, permutation p) for R = 1 km ("guaranteed measurement") and `R_MAIORIA`; CSVs and figures to `results/`.

The crop constants (`BOX`, `CANAL_LON`, `CANAL_LAT`) are duplicated in both notebooks and must stay identical. `D*`/`ℓ` must come from the Zipf calibration (or another citable criterion), never be tuned to make the population–plastic correlation look right.

## IPBA (vendored dependency)

`IPBA/` is a local copy of https://github.com/erneson/IPBA (its own git repo, nested inside this one — not a subtree/submodule of this repo). Each notebook run clones its own fresh copy of IPBA into the Colab runtime; the local `IPBA/` folder here is a reference copy for offline reading, not what the notebooks execute against.

IPBA implements the *Invasion Percolation Based Algorithm* for delineating drainage basins from a relief raster, given a set of seed "sinks":

- `main.py` — CLI entry point: `python -B main.py <relief_tif_basename> <sinks_tif_basename> <seed> <T|F periodic-boundary>`. Reads the relief GeoTIFF and a sink-label GeoTIFF from `datasets/`, runs the percolation, writes basin labels to `results/<relief>.tif`.
- `lattice/` — grid abstraction over the raster (`Lattice`, `Site` per pixel: height, sigma, label, parent, status).
- `heap/` — binary min-heap used as the percolation frontier, ordered by site height.
- `drainage_basins/drainage_basins.py` — `SetDrainageBasins` loops `InvasionPercolation` over every lattice site.
- `drainage_basins/invasion_percolation/invasion_percolation.py` — the core algorithm: floods outward through the heap, stopping when it hits a labeled site or a site with `sigma >= 2` (a basin boundary), assigning `parent` pointers that trace each pixel back to the sink whose basin it belongs to.
- `geotiff/` — thin GDAL-backed read/write helpers (`ReadGeoTiff`, `WriteGeoTiff`); labels are written as `uint16`.

## Data layout

- `notebooks/` — the two analysis notebooks above.
- `dados/` — `gpw_data/` (GPW v4 raster), `noaa_microplastics.csv` (full NOAA dump; notebooks filter to one method), `archives/` (downloaded zips), `pop_gpw_crop.tif` (old Texas crop). The notebooks' `local()` helper looks in `.`, `..`, `dados/` and `../dados/` before downloading.
- `results/` — notebook outputs; `results/texas_legado/` holds outputs of the previous Texas-crop / 50 km-radius version (still read by `artigo/gerar_graficos_artigo.py`).
- `artigo/` — LaTeX article (US Zipf) and Overleaf package.
- `apresentacao/`, `artigos_relacionados/` — slides and reference literature.
- `requirements.txt` — Python deps for local/offline work (not what Colab uses).
- `venv/` — local virtualenv; not part of the project's source.

## Working conventions

- The user communicates in Portuguese (pt-BR) for this project; match that unless asked otherwise.
- There is no test suite, linter, or build step in this repo — it is a research pipeline of notebooks plus one vendored CLI algorithm (IPBA), validated by re-reading the changed cells/code and cross-checking against `tasks.txt`, not by an automated CI.
