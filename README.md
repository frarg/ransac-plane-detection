# RANSAC Plane Detection from Point Clouds

Implementation of multi-plane RANSAC detection from 3D point clouds with RMS error filtering, based on the approach described in Schnabel et al. (2007).

## Features

- Iterative RANSAC plane detection: finds the best plane, removes inliers, repeats
- RMS error filtering to reject curved surfaces misclassified as planes
- Point reassignment refinement to clean up boundary misclassifications
- Configurable distance threshold, inlier count, and RMS error limits
- Gaussian noise evaluation for robustness testing

## Usage

```python
from geomproc import pcloud
import project

# Load a point cloud
pc = pcloud.load("plane-detection/w_shape_w6_x_12_x_500_points.ply")

# Run multi-plane detection
planes = project.detect_planes_iteratively(
    pc,
    num_iterations=5000,
    distance_threshold=0.005,
    min_inliers=50,
    max_rms_error=0.0005
)

# Visualize with coloured labels
labels = project.label_points(pc, planes)
project.labels_to_pcloud(pc, labels, "output.ply")
```

## Project Structure

| File | Purpose |
|------|---------|
| `project.py` | Main RANSAC implementation: plane estimation, iterative detection, labeling |
| `geomproc/` | Geometry processing library for loading/saving point clouds and meshes |
| `plane-detection/` | Sample point cloud data (.ply) with ground truth labels (.npy) |

## Sample Data

The `plane-detection/` directory contains industrial CAD model point clouds including various beam, flange, pin, and bracket shapes.
