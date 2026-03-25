"""
Crack Matching & Evaluation Script
----------------------------------

This script loads two datasets:
 - Ground truth crack data
 - Measured crack data

It:
 1) Parses and flattens the crack structures
 2) Computes symmetric polyline distances between measured and GT cracks
 3) Matches each measurement to the best GT crack (based on geometric similarity)
 4) Prints comparison metric tables
 5) Visualizes GT vs measured cracks in 3D

Usage:
    python Evaluation_example.py --data_path /path/to/dataset

Expected dataset structure inside data_path:
    GroundTruth.json
    Evaluation_example.json
"""

from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

FIGSIZE = (12, 8)


# =========================
# JSON Loading
# =========================

def load_json(path: Path):
    """Load a UTF‑8 JSON file safely."""
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
        Array of shape (N,) with per‑point minimal distances.
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
    One‑direction polyline distance.
    Converts point‑to‑polyline distances into a chosen statistic.
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
        if best_d_min < 0.05:
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


def plot_matches(gt_cracks, meas_cracks, matches):
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
                    label=f"Meas {uid} → {best} ({distance:.4f} m)")

            centroid = np.mean(C, axis=0)
            ax.text(*centroid, f"{uid}\n{distance:.3f} m",
                    fontsize=8, color=col)

        else:
            ax.plot(*C.T, linestyle="--", linewidth=1.5,
                    color=(0.5, 0.5, 0.5, 1.0),
                    label=f"Meas {uid} (no match)")

    ax.set_title("Ground Truth (solid) and Measurements (dashed) — Visual Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax)
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.show()


# =========================
# Main Workflow
# =========================

def main(data_path: Path):
    """Main evaluation pipeline."""
    gt_path = data_path / "GroundTruth.json"
    meas_path = data_path / "Evaluation_example.json"

    gt_data = load_json(gt_path)
    meas_data = load_json(meas_path)

    # Keep only folders that exist in both files
    shared_folders = sorted(set(gt_data.keys()) & set(meas_data.keys()))
    if not shared_folders:
        raise ValueError(
            "No common top‑level keys found between GT and measurements.\n"
            f"GT keys: {list(gt_data.keys())}\n"
            f"Meas keys: {list(meas_data.keys())}"
        )

    gt_filtered = {k: gt_data[k] for k in shared_folders}
    meas_filtered = {k: meas_data[k] for k in shared_folders}

    gt_cracks = collect_cracks(gt_filtered)
    meas_cracks = collect_cracks(meas_filtered)

    matches = match_measurements_to_gt(gt_cracks, meas_cracks)

    print("\n=== Comparison: Ground Truth ↔ Measurements ===")
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
            f"(Δ = {abs(w_gt - w_meas):.3f} mm)\n"
            f"  Length:     GT = {l_gt:.3f} m    |   Measured = {l_meas:.3f} m    "
            f"(Δ = {abs(l_gt - l_meas):.3f} m)\n"
        )

    plot_matches(gt_cracks, meas_cracks, matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crack Matching Evaluation")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset directory")
    args = parser.parse_args()

    main(Path(args.data_path))