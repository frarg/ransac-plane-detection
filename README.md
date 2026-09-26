# RANSAC Plane Detection from Point Clouds

Implementation of multi-plane RANSAC detection from 3D point clouds with RMS error filtering, based on the approach described in Schnabel et al. (2007).

## Features

- Iterative RANSAC plane detection: finds the best plane, removes inliers, repeats
- RMS error filtering to reject curved surfaces misclassified as planes
- Point reassignment refinement to clean up boundary misclassifications
- Configurable distance threshold, inlier count, and RMS error limits
- Gaussian noise evaluation for robustness testing

## Usage

### Requirements

```bash
pip install numpy matplotlib
```

`geomproc/` is vendored, so there is nothing else to install.

### Command line

Run from the repository root. Datasets are resolved relative to the working
directory (`DATA_DIR = "plane-detection"` in `project.py`).

```bash
# process every dataset in plane-detection/
python project.py

# process a single dataset
python project.py w_shape_w6_x_12_x_500
```

Each run takes about a second per dataset and writes
`output/<dataset>_ransac_planes.ply` the input geometry with one colour per
detected plane, and unassigned points left grey. Re-running overwrites the
previous result.

Passing an unknown dataset name prints the list of available ones.

### Tuning

Detection parameters are module-level constants at the top of `project.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `DISTANCE_THRESHOLD` | `0.001` | Inlier cutoff, in the units of the point cloud. Raising it merges nearby faces into a single plane. |
| `MAX_RMS_ERROR` | `0.0005` | Rejects candidates whose points deviate too much from the fitted plane. Lower it to reject curved surfaces. |
| `MIN_INLIERS` | `100` | A candidate plane must capture at least this many points to be kept. |
| `ITERATIONS` | `5000` | RANSAC samples per plane search. |
| `NOISE_SIGMA` | `0.0` | Standard deviation of Gaussian noise added to the input before detection. Set above `0.0` to test robustness. |

`DISTANCE_THRESHOLD` and `MAX_RMS_ERROR` interact: if the threshold admits a
candidate whose RMS error exceeds `MAX_RMS_ERROR`, detection stops at that
point rather than skipping the candidate and continuing.

A few details worth knowing:

- `detect_planes_iteratively` returns a list of `(PlaneModel, inlier_indices)`
  tuples, not bare planes. Each `PlaneModel` stores a unit `normal` and an
  offset `d`, and exposes `.distance(points)` for unsigned point-to-plane
  distances.
- A label of `-1` means the point was not assigned to any plane; it is written
  out grey.
- `labels_to_pcloud` returns a `geomproc.pcloud`; it does not write to disk.
  Saving needs a `write_options` with `write_point_colors` enabled, otherwise
  the `.ply` is written without RGB values.

## Project Structure

| File | Purpose |
|------|---------|
| `project.py` | Main RANSAC implementation: plane estimation, iterative detection, labeling |
| `geomproc/` | Geometry processing library for loading/saving point clouds and meshes |
| `plane-detection/` | Sample point cloud data (.ply) with ground truth labels (.npy) |

## Sample Data

The `plane-detection/` directory contains some sample model point clouds including various beam, flange, pin, and bracket shapes.
