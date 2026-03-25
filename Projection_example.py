"""
Point Cloud Projection & Camera Visualization Demo
---------------------------------------------------
This script demonstrates:
 - Loading & downsampling a point cloud (Open3D)
 - Projecting 3D points into camera images using intrinsics/extrinsics
 - Visualizing points in the image and in 3D
 - Clean, portable, GitHub-ready code structure

Usage:
    python Projection_example.py --data_path /path/to/dataset

Expected dataset folder structure:
    PointCloud.ply
    Cameras.json
    Images/
"""

import argparse
import os
import json
import copy
import numpy as np
import cv2 as cv
import open3d as o3d
import matplotlib.pyplot as plt


def load_pointcloud(path: str, voxel: float = 0.0005):
    """Load and downsample the point cloud."""
    pcd_path = os.path.join(path, "PointCloud.ply")
    if not os.path.isfile(pcd_path):
        raise FileNotFoundError(f"PointCloud.ply not found: {pcd_path}")

    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = pcd.voxel_down_sample(voxel_size=voxel)
    return pcd


def load_camera_data(path: str):
    """Load Cameras.json."""
    json_path = os.path.join(path, "Cameras.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Cameras.json not found: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)


def project_points(points, R, t, K):
    """Project 3D points into a camera based on R, t, K."""
    # Transform points to camera coordinate system
    pts_cam = R @ (points.T - t)

    # Only keep points in front of the camera
    z_filter = pts_cam[2, :] > 0

    # Project points
    pts_proj = (K @ pts_cam).T
    pts_proj[:, 0] /= pts_proj[:, 2]
    pts_proj[:, 1] /= pts_proj[:, 2]

    return pts_proj[:, :2].astype(np.int32), z_filter


def visualize_image(image, projected_points):
    """Draw 2D projected points onto the image and display it."""
    for pt in projected_points:
        cv.drawMarker(
            image,
            tuple(pt),
            color=[255, 0, 0],
            thickness=2,
            markerType=cv.MARKER_TILTED_CROSS,
            markerSize=4,
            line_type=cv.LINE_AA
        )

    img_small = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    img_small = cv.cvtColor(img_small, cv.COLOR_BGR2RGB)

    plt.imshow(img_small)
    plt.title("Projected Points")
    plt.axis("off")
    plt.show()


def visualize_3d(pcd, mask, R, t):
    """Show the point cloud and the camera frame in Open3D."""
    colors = np.asarray(pcd.colors).copy()
    colors[mask] = [1.0, 0.0, 0.0]  # highlight visible points

    pcd_vis = copy.deepcopy(pcd)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)

    # Build 4x4 camera pose
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R.T
    cam_pose[:3, 3] = t.reshape(3)

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    cam_frame.transform(cam_pose)

    o3d.visualization.draw_geometries([pcd_vis, cam_frame])


def main(data_path: str):
    # Load data
    pcd = load_pointcloud(data_path)
    cams = load_camera_data(data_path)

    image_dir = os.path.join(data_path, "Images")
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"Images folder missing: {image_dir}")

    points = np.asarray(pcd.points, dtype=np.float32)
    K = np.array(cams["Intrinsics"]["K"])
    dist = np.array(cams["Intrinsics"]["dist"])

    for img_name, extr in cams["Extrinsics"].items():

        img_path = os.path.join(image_dir, img_name + ".JPG")
        if not os.path.isfile(img_path):
            print(f"[WARN] Image missing: {img_path}")
            continue

        image = cv.imread(img_path)
        if image is None:
            print(f"[WARN] Failed to load image: {img_path}")
            continue

        # Undistort image
        image = cv.undistort(image, K, dist)

        R = np.array(extr["Rotation"])
        t = np.array(extr["Translation"]).reshape(3, 1)

        proj_pts, z_filter = project_points(points, R, t, K)

        h, w = image.shape[:2]
        valid = (
            (proj_pts[:, 0] >= 0) & (proj_pts[:, 0] < w) &
            (proj_pts[:, 1] >= 0) & (proj_pts[:, 1] < h) &
            z_filter
        )

        visualize_image(image.copy(), proj_pts[valid])
        visualize_3d(pcd, valid, R, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointCloud Projection Demo")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset folder"
    )
    args = parser.parse_args()
    main(args.data_path)