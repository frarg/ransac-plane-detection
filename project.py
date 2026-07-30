import numpy as np
import os
import geomproc
import matplotlib.pyplot as plt


DISTANCE_THRESHOLD = 0.001   # inlier distance cutoff for ransac plane fitting
MAX_RMS_ERROR = 0.0005       # max rms error allowed for an accepted plane
MIN_INLIERS = 100            # minimum number of inliers to keep a plane
ITERATIONS = 5000            # ransac iterations per plane search
NOISE_SIGMA = 0.0            # standard deviation of added gaussian noise (0.0 = no noise)


class PlaneModel:
    # store plane in normalized form n * x + d = 0
    def __init__(self, normal, d):
        # keep everything in float64 to maintain precision
        self.normal = np.asarray(normal, dtype=np.float64)
        self.d = float(d)

    def distance(self, pts):
        # compute unsigned point to plane distance for (N,3) array
        pts = np.asarray(pts, dtype=np.float64)
        n_norm = np.linalg.norm(self.normal)   # recomputed here for simplicity
        if n_norm == 0:
            # degenerate normal, just return zeros instead of crashing
            return np.zeros(pts.shape[0], dtype=np.float64)
        vals = pts @ self.normal + self.d
        dists = np.abs(vals) / n_norm
        return dists


def plane_from_three_points(p1, p2, p3):
    # build plane parameters (n, d) from 3 points
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    v1 = p2 - p1
    v2 = p3 - p1

    # normal is cross product of the two edges
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        # points almost collinear, not a valid plane
        raise ValueError("Degenerate triple of points.")

    n = n / n_norm
    d = -np.dot(n, p1)
    return n.copy(), float(d)

def ransac_plane(points,
                 distance_threshold=DISTANCE_THRESHOLD,
                 iterations=ITERATIONS,
                 min_inliers=MIN_INLIERS):
    # run ransac once to find the best plane in a noisy point cloud
    # returns (PlaneModel or None, inlier_mask or None)
    pts = np.asarray(points, dtype=np.float64)
    N = pts.shape[0]

    best_plane = None
    best_inliers = 0
    best_mask = None

    for _ in range(iterations):
        # pick 3 distinct points as a minimal sample
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = pts[idx]

        try:
            n, d = plane_from_three_points(p1, p2, p3)
        except ValueError:
            # skip degenerate triples
            continue

        plane = PlaneModel(n, d)

        # label points as inliers/outliers using the distance threshold
        dists = plane.distance(pts)
        mask = dists < distance_threshold
        num_inliers = int(np.count_nonzero(mask))

        # update best plane only if it beats the current one and is large enough
        if num_inliers > best_inliers and num_inliers >= min_inliers:
            best_plane = plane
            best_inliers = num_inliers
            best_mask = mask.copy()

    return best_plane, best_mask


def detect_planes_iteratively(points,
                              distance_threshold=DISTANCE_THRESHOLD,
                              iterations=ITERATIONS,
                              min_inliers=MIN_INLIERS,
                              max_rms_error=MAX_RMS_ERROR):
    # repeatedly run ransac, removing inliers to get multiple planes
    pts_all = np.asarray(points, dtype=np.float64)
    all_planes = []

    # keep track of remaining points using a separate array + index list
    remaining_points = pts_all.copy()
    remaining_indices = np.arange(pts_all.shape[0])

    iter_id = 0
    while True:
        # stop if too few points are left to form a useful plane
        if remaining_points.shape[0] < min_inliers:
            break

        iter_id += 1
        print("RANSAC iteration", iter_id, ": running on", remaining_points.shape[0], "points")

        # detect a single plane in the current remaining set
        plane, mask = ransac_plane(
            remaining_points,
            distance_threshold=distance_threshold,
            iterations=iterations,
            min_inliers=min_inliers,
        )

        # ransac may fail if no good plane is left
        if plane is None or mask is None:
            print("  No valid plane found, stopping.")
            break

        # recompute inliers explicitly
        dists_all = plane.distance(remaining_points)
        inlier_mask = dists_all < distance_threshold
        num_inliers = int(np.count_nonzero(inlier_mask))

        if num_inliers < min_inliers:
            print("  Candidate plane has only", num_inliers,
                  "inliers (min required", min_inliers, "); stopping.")
            break

        # map inliers back to original indices
        inlier_indices_global = remaining_indices[inlier_mask]
        pts_inliers = pts_all[inlier_indices_global]
        dists_inliers = plane.distance(pts_inliers)

        # rms error on this plane using squared distances
        squared_dists = dists_inliers ** 2
        rms = float(np.sqrt(np.mean(squared_dists)))

        print("  Candidate plane:", inlier_indices_global.size,
              "inliers, RMS error = %.5f" % rms)

        # reject planes that are too noisy
        if rms > max_rms_error:
            print("  RMS error %.5f exceeds max_rms_error=%.5f; stopping." %
                  (rms, max_rms_error))
            break

        # store accepted plane and its inlier indices
        all_planes.append((plane, inlier_indices_global.copy()))
        print("  Accepted plane", len(all_planes) - 1, ".")

        # drop inliers from the remaining pool and continue with leftovers
        remaining_points = remaining_points[~inlier_mask]
        remaining_indices = remaining_indices[~inlier_mask]

    return all_planes


def compute_plane_errors(points, planes):
    # compute per-plane rms error on their stored inlier indices
    pts_all = np.asarray(points, dtype=np.float64)
    errors = []

    for plane, idxs in planes:
        idxs = np.asarray(idxs, dtype=np.int64)
        pts = pts_all[idxs]
        if pts.size == 0:
            # handle empty inlier lists gracefully
            errors.append(0.0)
            continue

        dists = plane.distance(pts)
        squared = dists ** 2
        rms = float(np.sqrt(np.mean(squared)))
        errors.append(rms)

    return errors


def filter_planes_by_error(points, planes, max_rms_error=MAX_RMS_ERROR):
    # keep only planes whose rms error is below the given threshold
    if not planes:
        return [], []

    errors = compute_plane_errors(points, planes)
    kept_planes = []
    kept_errors = []

    for pl, err in zip(planes, errors):
        if err <= max_rms_error:
            kept_planes.append(pl)
            kept_errors.append(err)

    return kept_planes, kept_errors


def labels_to_pcloud(points, labels):
    # build a geomproc pcloud with per-point colour from integer labels
    pts = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    pc = geomproc.pcloud()
    pc.point = pts

    unique_labels = np.unique(labels)
    colours = np.zeros((pts.shape[0], 3), dtype=np.float32)

    # use a matplotlib colour map
    cmap = plt.get_cmap("tab20")

    for k in unique_labels:
        if k < 0:
            # negative label = unassigned / outlier
            continue

        colour = np.array(cmap(k % 20)[:3], dtype=np.float32)
        colours[labels == k] = colour

    # make unlabeled points gray
    colours[labels < 0] = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    pc.colour = colours
    return pc


def make_synthetic_plane_cloud(n_plane=500, n_outliers=200, noise=0.01):
    # sample points near a horizontal plane plus some random outliers
    xy = np.random.uniform(-1.0, 1.0, size=(n_plane, 2))
    z = 0.5 + noise * np.random.randn(n_plane)
    plane_pts = np.column_stack([xy, z])

    # uniform junk points in a larger box
    outliers = np.random.uniform(-1.5, 1.5, size=(n_outliers, 3))

    pts = np.vstack([plane_pts, outliers])
    return pts.copy()


DATA_DIR = "plane-detection"

def build_path(base_name, data_dir=DATA_DIR):
    # construct filenames for the points of a dataset
    points_path = os.path.join(data_dir, f"{base_name}_points.ply")
    return points_path


def run(base_name,
        noise_sigma=NOISE_SIGMA,
        distance_threshold=DISTANCE_THRESHOLD,
        max_rms_error=MAX_RMS_ERROR,
        min_inliers=MIN_INLIERS,
        iterations=ITERATIONS):
    # load one dataset, run plane detection, and save a coloured result
    points_path = build_path(base_name)

    print("Loading points from", points_path)
    points = np.asarray(geomproc.load(points_path).point, dtype=np.float64)
    points = points.reshape(-1, 3)
    print("  loaded", points.shape[0], "points")

    # if enabled, add noise
    if noise_sigma > 0.0:
        print("  adding Gaussian noise with std =", noise_sigma)
        points = points + noise_sigma * np.random.randn(*points.shape)

    theta = distance_threshold

    # detect planes using iterative ransac
    planes = detect_planes_iteratively(
        points,
        distance_threshold=theta,
        iterations=iterations,
        min_inliers=min_inliers,
        max_rms_error=max_rms_error,
    )

    # secondary cleanup for high-rms error planes just to be safe
    kept_planes, kept_errors = filter_planes_by_error(
        points, planes, max_rms_error=max_rms_error
    )

    print("Detected", len(planes), "candidate planes;",
          "kept", len(kept_planes), "with RMS <=", "%.5f" % max_rms_error)

    # initialize every point as unassigned (-1)
    labels = -np.ones(points.shape[0], dtype=int)
    all_planes = []
    # label each point with an integer corresponding to its plane
    for pid, ((plane, idxs), rms) in enumerate(zip(kept_planes, kept_errors)):
        labels[idxs] = pid
        all_planes.append(plane)
        print("  Plane", pid, ":", idxs.size, "inliers,",
              "rms = %.5f" % rms, ", normal =", plane.normal)

    # reassign each point to the closest plane if it is close enough for a cleaner visual
    if all_planes:
        dists_mat = np.stack([m.distance(points) for m in all_planes], axis=1)
        min_dists = dists_mat.min(axis=1)
        closest = dists_mat.argmin(axis=1)

        # keep a copy of the original labelling so we can compare distances
        orig_labels = labels.copy()
        new_labels = labels.copy()

        for i in range(points.shape[0]):
            dmin = min_dists[i]
            if dmin >= theta:
                # point is too far from every plane, stay unlabeled
                continue

            new_pid = int(closest[i])
            old_pid = int(orig_labels[i])

            if old_pid < 0:
                # previously unlabeled, so just assign to the best plane
                new_labels[i] = new_pid
            else:
                # point was already assigned, only switch if the new plane is much closer
                old_dist = dists_mat[i, old_pid]
                if dmin < 0.5 * old_dist:
                    new_labels[i] = new_pid

        labels = new_labels

    # build coloured point cloud and save as ply
    pc_out = labels_to_pcloud(points, labels)
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, f"{base_name}_ransac_planes.ply")
    print("Saving coloured point cloud to", out_name)
    wo = geomproc.write_options()
    wo.write_point_colours = True
    pc_out.save(out_name, wo)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="all",
                        help="Dataset name or 'all'")

    args = parser.parse_args()

    base_names = []

    # collect all dataset base names from DATA_DIR
    if os.path.isdir(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            if fname.endswith("_points.ply"):
                base = fname[:-len("_points.ply")]
                base_names.append(base)

    base_names = sorted(base_names)

    # allow running on a single pointcloud name instead of "all"
    if args.dataset != "all":
        if args.dataset in base_names:
            base_names = [args.dataset]
        else:
            print(f"Requested dataset '{args.dataset}' not found in {DATA_DIR}.")
            print("Available datasets:")
            for b in base_names:
                print("  ", b)
            return

    if not base_names:
        print(f"No *_points.ply files found in {DATA_DIR}. Nothing to do.")
        return

    print("Global parameters:")
    print("  Noise standard deviation:", NOISE_SIGMA)
    print("  RANSAC distance threshold:", DISTANCE_THRESHOLD)
    print("  RMS error threshold:", MAX_RMS_ERROR)
    print("  Min inliers:", MIN_INLIERS)
    print("  Iterations:", ITERATIONS)

    print("Datasets to process:")
    for b in base_names:
        print("  ", b)

    # run the pipeline on each selected dataset
    for b in base_names:
        print("-----------------------------------------")
        print("Running dataset:", b)
        run(b)


if __name__ == "__main__":
    main()
