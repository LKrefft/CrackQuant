"""
Crack-to-Mask Projection Error Evaluation
-----------------------------------------

This script evaluates how far a projected 3D crack polyline deviates
from a binary ground-truth crack mask in the camera image.

It performs:
 1) Projection of 3D crack points into the image plane
 2) Construction of a predicted polyline mask
 3) Distance transform from predicted pixels → ground-truth mask
 4) Calculation of deviation in pixels and millimeters
 5) Visual overlay (blue: predicted line, green: inside mask, red: outside)

Usage:
    python Evaluate_cracks_center_axes.py --data_path /path/to/dataset --mask_folder /path/to/mask_folder

Expected dataset structure inside data_path:
    GroundTruth.json
    Cameras.json
    binary_masks
"""


import matplotlib.pyplot as plt
import json
import cv2 as cv
import numpy as np
from pathlib import Path
import argparse


# ===============================================================
# Helper function for loading JSON
# ===============================================================

def load_json(path: Path):
    """Load UTF‑8 encoded JSON file with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ===============================================================
# Main evaluation function (logic unchanged)
# ===============================================================

def evaluate_projection_error(dataset_path: Path, mask_folder: Path):
    """Evaluate crack projection errors using the original logic."""

    # Load ground truth crack dataset
    gt_data = load_json(dataset_path / "GroundTruth.json")

    # Load camera parameters
    camera_data = load_json(dataset_path / "Cameras.json")
    K = np.array(camera_data["Intrinsics"]["K"], dtype=np.float32)
    dist = np.array(camera_data["Intrinsics"]["dist"], dtype=np.float32)

    # Iterate through ground truth structure
    for group_name, group in gt_data.items():

        for crack_name, crack_info in group.items():

            points_3d = np.asarray(crack_info["coordinates"], dtype=np.float32)
            if points_3d.size == 0:
                continue

            # Process each camera view
            for image_name, extr in camera_data["Extrinsics"].items():

                mask_path = mask_folder / f"{image_name}.png"
                if not mask_path.exists():
                    continue

                mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

                # Undistort mask
                mask_undist = cv.undistort(mask, K, dist)

                # Camera pose
                R = np.array(extr["Rotation"], dtype=np.float32)
                t = np.array(extr["Translation"], dtype=np.float32).reshape(3, 1)

                # -----------------------------------------------------------
                # 3D → Camera coordinates → Projection
                # -----------------------------------------------------------
                cam_pts = R @ (points_3d.T - t)
                z = cam_pts[2, :]
                valid_z = z > 0

                proj = (K @ cam_pts).T
                proj[:, 0] /= proj[:, 2]
                proj[:, 1] /= proj[:, 2]

                pts_2d = proj[:, :2]
                h, w = mask_undist.shape[:2]
                px, py = pts_2d[:, 0], pts_2d[:, 1]

                valid_img = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                valid = valid_z & valid_img

                if not np.any(valid):
                    continue

                pixel_points = pts_2d[valid].astype(np.int32)

                # Mask lookup (inside GT mask = True)
                mask_values = mask_undist[pixel_points[:, 1], pixel_points[:, 0]]
                inside_mask = mask_values > 0

                inside_pts = pixel_points[inside_mask]
                outside_pts = pixel_points[~inside_mask]

                # ============================================================
                # Create predicted polyline mask (same as original)
                # ============================================================
                if len(inside_pts) > 1:

                    line_mask = np.zeros((h, w), dtype=np.uint8)

                    line_points = pixel_points

                    # Remove identical consecutive points
                    if len(line_points) >= 2:
                        dedup = [line_points[0]]
                        for p in line_points[1:]:
                            if not np.array_equal(p, dedup[-1]):
                                dedup.append(p)
                        line_points = np.asarray(dedup, dtype=np.int32)

                    # Draw polyline
                    if len(line_points) >= 2:
                        cv.polylines(
                            line_mask,
                            [line_points.reshape(-1, 1, 2)],
                            isClosed=False,
                            color=255,
                            thickness=1,
                            lineType=cv.LINE_AA
                        )

                    # Binary masks
                    target_mask = (mask_undist > 0).astype(np.uint8)
                    pred_mask = (line_mask > 0)

                    # ============================================================
                    # Distance transform: distance(predicted pixel → GT mask)
                    # ============================================================
                    dist_map = cv.distanceTransform(
                        (1 - target_mask).astype(np.uint8),
                        cv.DIST_L2,
                        5
                    )

                    d_pred = dist_map[pred_mask]

                    mean_dist_px = d_pred.mean() if d_pred.size > 0 else np.nan


                    # Convert to metric scale
                    pts3d_valid = points_3d[valid]
                    pts2d_valid = pts_2d[valid]

                    d_world = np.linalg.norm(pts3d_valid[1:] - pts3d_valid[:-1], axis=1)
                    d_pixel = np.linalg.norm(pts2d_valid[1:] - pts2d_valid[:-1], axis=1)

                    valid_scale = d_pixel > 0
                    scale_vals = d_world[valid_scale] / d_pixel[valid_scale]
                    mean_scale = scale_vals.mean() if scale_vals.size > 0 else np.nan

                    mean_dist_mm = mean_dist_px * mean_scale * 1000 if mean_scale is not np.nan else np.nan

                    # ============================================================
                    # Output (clear and descriptive)
                    # ============================================================
                    print("\n=== Crack Projection Error ===")
                    print(f"Crack ID: {crack_name}")
                    print(f"Image: {image_name}")
                    print(f"Mean deviation (pixel): {mean_dist_px:.3f} px")
                    print(f"Mean deviation (metric): {mean_dist_mm:.3f} mm")

                    # ============================================================
                    # Visualization (same logic)
                    # ============================================================
                    vis = cv.cvtColor(mask_undist, cv.COLOR_GRAY2BGR)

                    # Predicted polyline → blue
                    vis[line_mask > 0] = (255, 0, 0)

                    # Outside GT mask → red
                    for pt in outside_pts:
                        cv.drawMarker(vis, tuple(pt), (0, 0, 255),
                                      markerType=cv.MARKER_CROSS,
                                      markerSize=10, thickness=2)

                    # Inside GT mask → green
                    for pt in inside_pts:
                        cv.drawMarker(vis, tuple(pt), (0, 255, 0),
                                      markerType=cv.MARKER_CROSS,
                                      markerSize=10, thickness=2)

                    vis_rgb = cv.cvtColor(vis, cv.COLOR_BGR2RGB)

                    plt.figure(figsize=(12, 8))
                    plt.imshow(vis_rgb)
                    plt.title(
                        f"{image_name}\n"
                        f"Projected: {len(pixel_points)} | "
                        f"Inside GT: {np.sum(inside_mask)} | "
                        f"Outside GT: {np.sum(~inside_mask)}"
                    )
                    plt.axis("off")
                    plt.tight_layout()
                    plt.show()


# ===============================================================
# Argument Parser
# ===============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate projection error between 3D crack lines and 2D GT masks."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset folder containing GroundTruth.json and Cameras.json",
    )

    parser.add_argument(
        "--mask_folder",
        type=str,
        required=True,
        help="Path to folder containing binary crack masks (.png files).",
    )

    args = parser.parse_args()

    evaluate_projection_error(Path(args.data_path), Path(args.mask_folder))