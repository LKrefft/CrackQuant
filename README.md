
<img width="1732" height="599" alt="Cover" src="https://github.com/user-attachments/assets/d672558c-ee57-4130-9f8a-c3a1b9e0a256" />


CrackQuant Dataset Utilities

This repository is part of the CrackQuant dataset.
It contains helper scripts intended to simplify working with the dataset (e.g., projection, evaluation against ground truth, and verification of crack center axes).

Installation

All required Python packages are listed in requirements.txt. Install them using:

pip install -r requirements.txt


Usage

All scripts accept the dataset root directory via --data_path.

1) Projecting a 3D Point Cloud to the Image Plane

Projection_example.py demonstrates how to project a 3D point cloud onto the image plane.

python Projection_example.py --data_path /path/to/dataset

2) Evaluating Estimated Results Against Ground Truth

Evaluation_example.py provides an example workflow for comparing estimated/derived results against the ground truth data.

python Evaluation_example.py --data_path /path/to/dataset

3) Verifying Ground Truth Crack Center Axes Using Binary Masks

Evaluate_cracks_center_axes.py demonstrates how to verify the center axes stored in ground-truth .json files using corresponding binary masks.

python Evaluate_cracks_center_axes.py --data_path /path/to/dataset --mask_folder /path/to/mask_folder

Arguments:
- --data_path: Path to the CrackQuant dataset root directory
- --mask_folder: Path to the folder containing the binary masks (e.g., predicted or processed masks)

Notes

- Replace /path/to/dataset and /path/to/mask_folder with your actual paths.


License / Attribution

Please follow the CrackQuant dataset license and citation requirements.
This repository contains only helper scripts and does not redistribute the dataset itself.
