"""
Crack Matching & Evaluation Script (Batch Version)
----------------------------------------------------

This script evaluates crack measurements against ground truth data.

It supports two modes:

1) SINGLE MODE
   data_path points directly to a folder containing
       GroundTruth.json
       Evaluation_example.json

2) BATCH MODE
   data_path points to a root folder (e.g. ".../CrackQuant/") that contains
   several test series:

       CrackQuant/
         R01/
           R1_001/  -> GroundTruth.json, Evaluation_example.json
           R1_002/  -> GroundTruth.json, Evaluation_example.json
         R02/
           R2_001/  -> Evaluation_example.json
           ...

   The script searches data_path RECURSIVELY for every folder that contains
   an 'Evaluation_example.json'. For each such folder, a matching
   'GroundTruth.json' is looked up: first in the same folder, otherwise in
   the parent folders up to data_path (e.g. if the ground truth applies to a
   whole test series).

   For every experiment found, exactly the same metric/matching logic as in
   single mode is applied. At the end, the script prints:
     - a detail table for every matched crack (all experiments)
     - a summary per experiment
     - a summary per test series
     - an overall summary across all experiments

Usage:

    python Evaluation_batch.py --data_path /path/to/a/single/experiment --plot
    python Evaluation_batch.py --data_path /path/to/root --plot
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

FIGSIZE = (12, 8)


# =========================
# JSON Loading
# =========================

def load_json(path: Path):
    """Load a UTF-8 JSON file safely."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# =========================
# Data Helpers
# =========================

def to_array(coords):
    """
    Convert coordinate lists to Nx3 NumPy arrays.
    Ensures that a single [x,y,z] becomes [[x,y,z]].
    """
    arr = np.asarray(coords or [], dtype=float)
    if arr.ndim == 1 and arr.size == 3:
        arr = arr.reshape(1, 3)
    return arr


def collect_cracks(data):
    """
    Flatten nested ground truth / measurement structure.

    Input format:
        { folder : { crack_name : {...} } }

    Output:
        { "folder/id" : { ... } }
    """
    cracks = {}
    for folder, items in data.items():
        for name, info in items.items():
            uid = f"{folder}/{name}"
            cracks[uid] = {
                "folder": folder,
                "name": name,
                "coords": to_array(info.get("coordinates", [])),
                "width_mm_median": info.get("width_mm_median"),
                "length_m": info.get("length_m"),
            }
    return cracks


# =========================
# Geometry Utilities
# =========================

def point_to_polyline_distances(P, polyline, chunk_segments=1024):
    """
    Compute minimal distance between each point in P and a polyline.

    Returns:
        Array of shape (N,) with per-point minimal distances.
    """
    P = np.asarray(P, dtype=float)
    C = np.asarray(polyline, dtype=float)

    if P.size == 0 or C.shape[0] < 2:
        return np.array([np.inf], dtype=float)

    A_all = C[:-1]
    B_all = C[1:]
    M = A_all.shape[0]

    min_d2 = np.full((P.shape[0],), np.inf, dtype=float)

    for s in range(0, M, chunk_segments):
        A = A_all[s:s + chunk_segments]
        B = B_all[s:s + chunk_segments]

        AB = B - A
        AB2 = np.sum(AB * AB, axis=1)

        PA = P[:, None, :] - A[None, :, :]
        dot = np.sum(PA * AB[None, :, :], axis=2)

        denom = np.where(AB2 > 0, AB2, 1.0)
        t = np.clip(dot / denom[None, :], 0.0, 1.0)

        Q = A[None, :, :] + t[:, :, None] * AB[None, :, :]
        d2 = np.sum((P[:, None, :] - Q) ** 2, axis=2)

        min_d2 = np.minimum(min_d2, np.min(d2, axis=1))

    return np.sqrt(min_d2)


def polyline_distance(A_points, B_points, statistic="p95", chunk_segments=1024):
    """
    One-direction polyline distance.
    Converts point-to-polyline distances into a chosen statistic.
    """
    d = point_to_polyline_distances(A_points, B_points, chunk_segments)
    if statistic == "mean":
        return float(np.mean(d))
    if statistic == "median":
        return float(np.median(d))
    if statistic == "max":
        return float(np.max(d))
    if statistic == "p95":
        return float(np.percentile(d, 95))
    if statistic == "p99":
        return float(np.percentile(d, 99))
    raise ValueError("Unknown statistic")


def symmetric_polyline_distance(A_points, B_points, statistic="max", chunk_segments=1024):
    """
    Compute symmetric polyline distance:
        d1 = A -> B
        d2 = B -> A
    Returns (d1, d2).
    """
    d1 = polyline_distance(A_points, B_points, statistic, chunk_segments)
    d2 = polyline_distance(B_points, A_points, statistic, chunk_segments)
    return d1, d2


# =========================
# Crack Matching
# =========================

def match_measurements_to_gt(gt_cracks, meas_cracks):
    """
    For every measured crack, find the best matching GT crack.
    Matching is based on symmetric polyline distance.

    A match is accepted only if the minimum directional distance < 0.05 m.
    """
    results = {}
    gt_items = list(gt_cracks.items())

    for meas_uid, meas in meas_cracks.items():
        A = meas["coords"]
        best_uid = None
        best_d = float("inf")
        best_d_min = None

        for gt_uid, gt in gt_items:
            d1, d2 = symmetric_polyline_distance(A, gt["coords"],
                                                 statistic="mean",
                                                 chunk_segments=1024)
            d = max(d1, d2)

            if d < best_d:
                best_d = d
                best_d_min = min(d1, d2)
                best_uid = gt_uid

        # Apply acceptance criterion
        if best_d_min is not None and best_d_min < 0.05:
            results[meas_uid] = {
                "best_gt": best_uid,
                "distance": best_d,
                "width_mm_mes": meas["width_mm_median"],
                "width_mm_gt": gt_cracks[best_uid]["width_mm_median"],
                "length_m_mes": meas["length_m"],
                "length_m_gt": gt_cracks[best_uid]["length_m"],
            }

    return results


# =========================
# Visualization
# =========================

def _set_axes_equal(ax):
    """Ensure equal aspect ratio in 3D plots."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    span = max(x_limits[1] - x_limits[0],
               y_limits[1] - y_limits[0],
               z_limits[1] - z_limits[0]) / 2

    centers = [
        (x_limits[0] + x_limits[1]) / 2,
        (y_limits[0] + y_limits[1]) / 2,
        (z_limits[0] + z_limits[1]) / 2,
    ]

    ax.set_xlim3d([centers[0] - span, centers[0] + span])
    ax.set_ylim3d([centers[1] - span, centers[1] + span])
    ax.set_zlim3d([centers[2] - span, centers[2] + span])


def plot_matches(gt_cracks, meas_cracks, matches, title_suffix=""):
    """Visualize GT (solid) vs measurement (dashed) in 3D."""
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    gt_uids = list(gt_cracks.keys())
    cmap = plt.get_cmap("tab20")
    color_map = {uid: cmap(i % 20) for i, uid in enumerate(gt_uids)}

    # Ground truth (solid lines)
    for uid, g in gt_cracks.items():
        C = g["coords"]
        if C.size == 0:
            continue
        ax.plot(*C.T, linewidth=2.5, color=color_map[uid], label=f"GT {uid}")

    # Measurements (dashed lines)
    for uid, m in meas_cracks.items():
        C = m["coords"]
        if C.size == 0:
            continue

        if uid in matches:
            best = matches[uid]["best_gt"]
            distance = matches[uid]["distance"]
            col = color_map.get(best, (0.5, 0.5, 0.5, 1.0))

            ax.plot(*C.T, linestyle="--", linewidth=1.5, color=col,
                    label=f"Meas {uid} -> {best} ({distance:.4f} m)")

            centroid = np.mean(C, axis=0)
            ax.text(*centroid, f"{uid}\n{distance:.3f} m",
                    fontsize=8, color=col)

        else:
            ax.plot(*C.T, linestyle="--", linewidth=1.5,
                    color=(0.5, 0.5, 0.5, 1.0),
                    label=f"Meas {uid} (no match)")

    ax.set_title(f"Ground Truth (solid) and Measurements (dashed){title_suffix}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax)
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.show()


# =========================
# Single-experiment evaluation
# =========================

def evaluate_single_experiment(gt_path: Path, meas_path: Path):
    """
    Runs the full matching pipeline for exactly one experiment
    (one GroundTruth.json + one Evaluation_example.json).

    Returns (gt_cracks, meas_cracks, matches).
    """
    gt_data = load_json(gt_path)
    meas_data = load_json(meas_path)

    # Keep only folders that exist in both files
    shared_folders = sorted(set(gt_data.keys()) & set(meas_data.keys()))
    if not shared_folders:
        raise ValueError(
            "No common top-level keys found between GT and measurements.\n"
            f"GT keys: {list(gt_data.keys())}\n"
            f"Meas keys: {list(meas_data.keys())}"
        )

    gt_filtered = {k: gt_data[k] for k in shared_folders}
    meas_filtered = {k: meas_data[k] for k in shared_folders}

    gt_cracks = collect_cracks(gt_filtered)
    meas_cracks = collect_cracks(meas_filtered)

    matches = match_measurements_to_gt(gt_cracks, meas_cracks)

    return gt_cracks, meas_cracks, matches


def print_single_result(matches, meas_cracks):
    """Prints the same textual comparison table as in the original script."""
    print("\n=== Comparison: Ground Truth <-> Measurements ===")
    for meas_uid, info in matches.items():
        w_gt = info["width_mm_gt"]
        w_meas = info["width_mm_mes"]
        l_gt = info["length_m_gt"]
        l_meas = info["length_m_mes"]

        print(
            f"Measurement: {meas_uid}\n"
            f"  Matched Ground Truth: {info['best_gt']}\n"
            f"  Geometric distance: {info['distance']:.6f} m\n"
            f"  Width:      GT = {w_gt:.3f} mm   |   Measured = {w_meas:.3f} mm   "
            f"(Delta = {abs(w_gt - w_meas):.3f} mm)\n"
            f"  Length:     GT = {l_gt:.3f} m    |   Measured = {l_meas:.3f} m    "
            f"(Delta = {abs(l_gt - l_meas):.3f} m)\n"
        )


# =========================
# Batch discovery
# =========================

def find_ground_truth(start_dir: Path, root_path: Path):
    """
    Looks for 'GroundTruth.json' first in start_dir, then in the parent
    folders upwards, up to (and including) root_path.
    Returns None if nothing was found.
    """
    current = start_dir.resolve()
    root = root_path.resolve()

    while True:
        candidate = current / "GroundTruth.json"
        if candidate.exists():
            return candidate
        if current == root or current.parent == current:
            break
        current = current.parent
    return None


def find_experiments(root_path: Path):
    """
    Recursively searches root_path for every folder that contains an
    'Evaluation_example.json', and assigns a matching 'GroundTruth.json'
    to it if possible.

    Returns:
        List of dicts with:
            test_series, experiment, path, meas_path, gt_path
    """
    experiments = []

    for meas_path in sorted(root_path.rglob("Evaluation_example.json")):
        exp_dir = meas_path.parent
        gt_path = find_ground_truth(exp_dir, root_path)

        try:
            rel_parts = exp_dir.relative_to(root_path).parts
        except ValueError:
            rel_parts = (exp_dir.name,)

        if len(rel_parts) >= 2:
            test_series, experiment = rel_parts[0], rel_parts[-1]
        elif len(rel_parts) == 1:
            test_series, experiment = rel_parts[0], rel_parts[0]
        else:
            test_series, experiment = exp_dir.parent.name, exp_dir.name

        experiments.append({
            "test_series": test_series,
            "experiment": experiment,
            "path": exp_dir,
            "meas_path": meas_path,
            "gt_path": gt_path,
        })

    return experiments


# =========================
# Aggregation
# =========================

def matches_to_rows(matches, test_series, experiment):
    rows = []
    for meas_uid, info in matches.items():
        w_gt, w_meas = info["width_mm_gt"], info["width_mm_mes"]
        l_gt, l_meas = info["length_m_gt"], info["length_m_mes"]
        rows.append({
            "test_series": test_series,
            "experiment": experiment,
            "measurement": meas_uid,
            "matched_gt": info["best_gt"],
            "distance_m": info["distance"],
            "width_gt_mm": w_gt,
            "width_meas_mm": w_meas,
            "width_abs_error_mm": abs(w_meas - w_gt),
            "length_gt_m": l_gt,
            "length_meas_m": l_meas,
            "length_abs_error_m": abs(l_meas - l_gt),
        })
    return rows


def summarize(df: pd.DataFrame, group_cols):
    """
    Aggregate metrics, grouped by group_cols (list of column names).

    For every metric (distance, width error, length error) both the mean
    and the median are reported, using the consistent naming scheme
    '{metric}_mean_{unit}' / '{metric}_median_{unit}'.
    """
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(group_cols, dropna=False).agg(
        n_matches=("measurement", "count"),
        distance_mean_m=("distance_m", "mean"),
        distance_median_m=("distance_m", "median"),
        width_abs_error_mean_mm=("width_abs_error_mm", "mean"),
        width_abs_error_median_mm=("width_abs_error_mm", "median"),
        length_abs_error_mean_m=("length_abs_error_m", "mean"),
        length_abs_error_median_m=("length_abs_error_m", "median"),
    ).reset_index()
    return agg


def overall_summary(df: pd.DataFrame):
    """Overall summary without grouping (all experiments combined)."""
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "n_matches": len(df),
        "distance_mean_m": df["distance_m"].mean(),
        "distance_median_m": df["distance_m"].median(),
        "width_abs_error_mean_mm": df["width_abs_error_mm"].mean(),
        "width_abs_error_median_mm": df["width_abs_error_mm"].median(),
        "length_abs_error_mean_m": df["length_abs_error_m"].mean(),
        "length_abs_error_median_m": df["length_abs_error_m"].median(),
    }])


# =========================
# Batch workflow
# =========================

def run_batch_evaluation(root_path: Path, plot=False):
    experiments = find_experiments(root_path)

    if not experiments:
        raise FileNotFoundError(
            f"No 'Evaluation_example.json' found below {root_path}."
        )

    print(f"\nFound {len(experiments)} experiment(s) with Evaluation_example.json.\n")

    all_rows = []
    failed = []

    for exp in experiments:
        test_series, experiment = exp["test_series"], exp["experiment"]
        meas_path, gt_path = exp["meas_path"], exp["gt_path"]

        print(f"--- Processing {test_series}/{experiment} ---")

        if gt_path is None:
            print("  -> No GroundTruth.json found (skipped).")
            failed.append({"test_series": test_series, "experiment": experiment,
                            "reason": "GroundTruth.json missing"})
            continue

        try:
            gt_cracks, meas_cracks, matches = evaluate_single_experiment(gt_path, meas_path)
        except Exception as e:
            print(f"  -> Error during evaluation: {e}")
            failed.append({"test_series": test_series, "experiment": experiment, "reason": str(e)})
            continue

        rows = matches_to_rows(matches, test_series, experiment)
        all_rows.extend(rows)
        print(f"  -> {len(rows)} of {len(meas_cracks)} measurements matched "
              f"(GT: {gt_path.relative_to(root_path)}).")

        if plot:
            plot_matches(gt_cracks, meas_cracks, matches,
                         title_suffix=f" — {test_series}/{experiment}")

    df = pd.DataFrame(all_rows)

    print("\n=== Detailed results per matched crack ===")
    if not df.empty:
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(df.to_string(index=False))
    else:
        print("No matches found.")

    print("\n=== Summary per experiment ===")
    per_experiment = summarize(df, ["test_series", "experiment"])
    print(per_experiment.to_string(index=False) if not per_experiment.empty else "No data.")

    print("\n=== Summary per test series ===")
    per_series = summarize(df, ["test_series"])
    print(per_series.to_string(index=False) if not per_series.empty else "No data.")

    print("\n=== Overall summary (all experiments) ===")
    overall = overall_summary(df)
    print(overall.to_string(index=False) if not overall.empty else "No data.")

    if failed:
        print("\n=== Skipped / failed experiments ===")
        for f in failed:
            print(f"  {f['test_series']}/{f['experiment']}: {f['reason']}")

    return df, per_experiment, per_series, overall


# =========================
# Main Workflow
# =========================

def main(data_path: Path, plot: bool):
    gt_path = data_path / "GroundTruth.json"
    meas_path = data_path / "Evaluation_example.json"

    if gt_path.exists() and meas_path.exists():
        # -------- Single mode (same as the original script) --------
        gt_cracks, meas_cracks, matches = evaluate_single_experiment(gt_path, meas_path)
        print_single_result(matches, meas_cracks)
        plot_matches(gt_cracks, meas_cracks, matches)
    else:
        # -------- Batch mode: data_path is the root folder --------
        run_batch_evaluation(data_path, plot=plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crack Matching Evaluation (single or batch mode)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to a single experiment folder OR to a root folder "
                             "containing multiple test series/experiments")
    parser.add_argument("--plot", action="store_true",
                        help="Show 3D plots (in batch mode: one per experiment, off by default)")
    args = parser.parse_args()

    main(Path(args.data_path), plot=args.plot)