import os
import numpy as np
import trimesh
from scipy.ndimage import binary_dilation
import random

# Define directories
raw_data_dir = "DeepView_code/training/volumes"
ply_data_dir = "DeepView_code/training/meshes"
output_mask_dir = "DeepView_code/output_masks"

# Dimensions for reshaping volumes
volume_dimensions = (1280, 768, 768)

def load_raw_file(file_path, dimensions=(1280, 768, 768), dtype=np.uint16):
    volume = np.fromfile(file_path, dtype=dtype)
    volume = volume.reshape(dimensions)
    return volume

def generate_mask_from_ply(ply_file, dimensions=(1280, 768, 768)):
    mesh = trimesh.load(ply_file)
    points = mesh.vertices
    mask = np.zeros(dimensions, dtype=np.uint8)
    for x, y, z in points:
        ix, iy, iz = int(z), int(y), int(x)
        if 0 <= ix < dimensions[0] and 0 <= iy < dimensions[1] and 0 <= iz < dimensions[2]:
            mask[ix, iy, iz] = 1
    return mask

def dilate_mask(mask, iterations=2):
    return binary_dilation(mask, iterations=iterations).astype(np.uint8)

def augment_data(volume, mask=None):
    k = random.choice([0, 1, 2, 3])
    axes = random.choice([(0, 1), (0, 2), (1, 2)])
    augmented_volume = np.rot90(volume, k=k, axes=axes)
    augmented_mask = None
    if mask is not None:
        augmented_mask = np.rot90(mask, k=k, axes=axes)
    if random.choice([True, False]):
        axis = random.choice([0, 1, 2])
        augmented_volume = np.flip(augmented_volume, axis=axis)
        if augmented_mask is not None:
            augmented_mask = np.flip(augmented_mask, axis=axis)
    return augmented_volume, augmented_mask

# Process all .raw files in a single loop
for raw_file in sorted(os.listdir(raw_data_dir)):
    if raw_file.endswith(".raw"):
        # Paths to the .raw file and corresponding .ply file
        raw_path = os.path.join(raw_data_dir, raw_file)
        ply_path = os.path.join(ply_data_dir, raw_file.replace(".raw", ".ply"))

        # Load the volume from the .raw file
        volume = load_raw_file(raw_path, dimensions=volume_dimensions)

        # Check if the .ply file exists (to determine if it's labeled or unlabeled)
        if os.path.exists(ply_path):
            # Labeled data: generate mask and apply augmentation
            mask = generate_mask_from_ply(ply_path, dimensions=volume_dimensions)
            mask = dilate_mask(mask)
            augmented_volume, augmented_mask = augment_data(volume, mask)

            # Save the augmented labeled volume and mask
            output_volume_path = os.path.join(output_mask_dir, f"augmented_{raw_file}.npy")
            output_mask_path = os.path.join(output_mask_dir, f"mask_{raw_file}.npy")
            np.save(output_volume_path, augmented_volume)
            np.save(output_mask_path, augmented_mask)

            print(f"Processed labeled file: {raw_file} with mask {os.path.basename(ply_path)}")

        else:
            # Unlabeled data: apply augmentation without mask
            augmented_volume, _ = augment_data(volume)

            # Save the augmented unlabeled volume
            output_volume_path = os.path.join(output_mask_dir, f"augmented_unlabeled_{raw_file}.npy")
            np.save(output_volume_path, augmented_volume)

            print(f"Processed unlabeled file: {raw_file}")
