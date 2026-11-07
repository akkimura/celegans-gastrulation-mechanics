# Cortical Recoil Analysis

This directory contains Python code for analyzing cortical recoil following laser ablation during *C. elegans* gastrulation. The scripts calculate velocity fields from time-lapse NMY-2 fluorescence images, determine mean recoil velocity relative to the ablation site, and quantify recoil direction.

## Files

- `01_compute_piv/compute_piv.py` — Calculates PIV velocity fields from pairs of TIFF images and exports CSV and MAT files.
- `02_compute_mean_recoil/compute_mean_recoil.py` — Calculates mean recoil velocity by reflecting velocity components relative to the ablation site.
- `03_analyze_recoil_direction/analyze_recoil_direction.py` — Calculates recoil direction angles and exports time-series data and plots.

## Requirements

- Python 3.10 or later is recommended.
- NumPy, SciPy, ImageIO, tifffile, OpenPIV, and Matplotlib.
- A Python installation with Tkinter support is required for `compute_piv.py`.

Install the Python packages using:

```bash
python3 -m pip install numpy scipy imageio tifffile openpiv matplotlib
```

## Input data

The analysis uses a time series of two-dimensional grayscale TIFF images of NMY-2 fluorescence, with one image per frame. Images must be stored in a `raw_images` directory within the project directory.

File names must end with a frame number followed by `.tif` (e.g., `image0000.tif`, `image0001.tif`). Images in each analyzed pair must have identical dimensions.

The coordinates of the laser ablation site are specified in pixels when calculating mean recoil velocity.

## Usage

Run the scripts sequentially from the directory containing this README. Replace `/path/to/project` and the ablation coordinates with the appropriate values.

**1. Calculate PIV velocity fields:**

```bash
python3 01_compute_piv/compute_piv.py \
  --project-dir "/path/to/project" \
  --k 1 --w 16 --sn 1.025
```

Here, `--k` specifies the frame interval between compared images, `--w` the interrogation window size in pixels, and `--sn` the signal-to-noise threshold. Acquisition settings, including `FRAME_INTERVAL_S`, should be checked near the beginning of the script before execution.

**2. Calculate mean recoil velocity:**

```bash
python3 02_compute_mean_recoil/compute_mean_recoil.py \
  --project-dir "/path/to/project" \
  --origin-x 256 --origin-y 256
```

The `--origin-x` and `--origin-y` arguments specify the laser ablation site in pixel coordinates.

**3. Calculate recoil direction:**

```bash
python3 03_analyze_recoil_direction/analyze_recoil_direction.py \
  --project-dir "/path/to/project" \
  --dt 1.0 --time-offset 0.0
```

The `--dt` argument specifies the acquisition interval in seconds per frame and should match `FRAME_INTERVAL_S` used in the PIV analysis.

## Quantitative analysis

Three analysis steps are performed:

1. **PIV analysis:** Displacement vectors are calculated from consecutive or selected pairs of images. Vectors below the specified signal-to-noise threshold are excluded.

2. **Mean recoil velocity:** Each valid displacement vector is averaged. The magnitude is calculated from the mean vector components, rather than by averaging individual vector magnitudes.

3. **Recoil direction:** The angle between the mean velocity vector and the positive x-axis is calculated using the arccosine of its normalized x-component.


## Output data

- `piv_results_k*_w*_snr*/` — PIV velocity fields for each image pair, saved as CSV and MAT files.
- `mean_recoil/mean_recoil_velocity.csv` — Mean recoil velocity components and magnitudes for each image pair and analysis condition.
- `recoil_direction/<condition>/recoil_direction_time_series.csv` — Time-series measurements of mean recoil velocity and direction.
- `recoil_direction/<condition>/recoil_angle_to_x_vs_time.png` — Angle relative to the positive x-axis over time.
- `recoil_direction/<condition>/recoil_direction_vs_time.png` — Signed recoil direction angle over time.

