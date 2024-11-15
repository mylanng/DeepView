# DeepView
# Volumetric to Surface Mesh Estimation

## Problem Statement
This project aims to develop a deep learning model that estimates the surface mesh of a given volumetric ultrasound image. Each volume is a separate scan containing a piece (or connected pieces) of steel pipe, with or without an object inside the pipe, and with or without debris/dirt at the bottom of the pipe. The solution must address the challenge of mapping raw 3D volumetric data to structured surface meshes for applications such as industrial inspection, medical imaging, and 3D modeling.

### Dataset Overview
- **Training Dataset**:
  - **Volumes**: 89 raw volumetric ultrasound images stored as `.raw` files.
  - **Meshes**: 5 reference 3D meshes in `.ply` format corresponding to the volumetric scans `001-005`.
- **Testing Dataset**:
  - **Volumes**: 10 raw volumetric ultrasound images.
  - **Meshes**: 10 reference 3D meshes.

### Raw Volumetric Image Metadata
- **Origin**: (0, 0, 0)
- **Spacing**: (0.49479, 0.49479, 0.3125)
- **Data Type**: Unsigned short integer
- **Volume Dimension**: (768, 768, 1280)

### Visualization
We recommend using [ParaView](https://www.paraview.org) (version 5.9.1) for visualizing volumetric data and meshes. Proper alignment can be achieved using the provided metadata. Some suggestions to visualize the data:
- Data Scalar Type: unsigned short
- Data Byte Order: LittleEndian
- Data Extent: 768 x 768 x 1280

---

## Project Workflow

### 1. Preprocessing
- **Volumes**:
  - Normalize intensity values to the range [0, 1].
  - Resize volumes to a uniform shape (e.g., 128x128x128 or 64x64x64 for optimization).
- **Meshes**:
  - Center vertex coordinates.
  - Normalize mesh dimensions to fit within a unit sphere.
  - Pad or trim vertices to ensure a consistent number (e.g., 10,000 vertices).

### 2. Model Architecture
The `VolumetricToMeshModel` uses a 3D Convolutional Neural Network (3D-CNN) backbone followed by fully connected layers for regression. Key features:
- **3D-CNN Layers**: Extract spatial features from volumetric data.
- **Fully Connected Layers**: Map latent features to 3D vertex predictions.

### 3. Training
- Loss Function: Mean Squared Error (MSE) between predicted and ground truth mesh vertices.
- Optimizer: Adam with a learning rate of 1e-4.
- Training Process:
  - Input: Preprocessed volumetric images.
  - Target: Corresponding 3D mesh vertices.

### 4. Evaluation
- **Metrics**:
  - **Chamfer Distance**: Measures the average distance between predicted and ground truth point clouds.
  - **Hausdorff Distance**: Measures the maximum distance between the two point clouds.

---

## Instructions

### 1. Dataset directory
Make sure that the training folder and testing folder is in the same root directory as README.md and Report.ipynb.

### 2. Training
Run the training script to preprocess data and train the model:
```bash
python harry_model.py
```
This will save the trained model as `volumetric_to_mesh_model.keras` in the `source/` directory.

### 3. Evaluation
Run the evaluation script to compute Chamfer and Hausdorff distances:
```bash
python model_eval.py
```
Ensure the `testing/` directory contains the volumetric and mesh files for testing.

### 4. Report
The project includes a Jupyter notebook report (`report.ipynb`) documenting the methodology, results, and analysis. To view the report:
```bash
jupyter notebook report.ipynb
```

---

## Requirements
- Python 
- TensorFlow 
- NumPy
- SciPy
- trimesh
- ParaView (optional, for visualization)

---

## Acknowledgments
This project is part of a computer vision competition jointly organized by DarkVision and the UBC Data Science Club. Special thanks to both organizations for fostering innovation and providing the resources to tackle challenging problems in 3D modeling and machine learning.
