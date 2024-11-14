import numpy as np
import trimesh
import os

def generate_mask_from_ply(ply_file, dimensions=(1280, 768, 768)):
    mesh = trimesh.load(ply_file)
    points = mesh.vertices
    mask = np.zeros(dimensions, dtype=np.uint8)
    for x, y, z in points:
        ix, iy, iz = int(z), int(y), int(x)
        if 0 <= ix < dimensions[0] and 0 <= iy < dimensions[1] and 0 <= iz < dimensions[2]:
            mask[ix, iy, iz] = 1
    return mask

def create_test_masks(mesh_dir, output_mask_dir, dimensions=(1280, 768, 768)):
    os.makedirs(output_mask_dir, exist_ok=True)
    for i in range(1, 11):  # Assuming 10 test files
        ply_file = os.path.join(mesh_dir, f"scan_{i:03d}.ply")
        mask = generate_mask_from_ply(ply_file, dimensions=dimensions)
        output_file = os.path.join(output_mask_dir, f"mask_scan_{i:03d}.npy")
        np.save(output_file, mask)
        print(f"Generated and saved mask for {ply_file} to {output_file}")

# Define paths for test meshes and output directory for masks
mesh_dir = "DeepView_code/testing/meshes"
output_mask_dir = "DeepView_code/testing/masks"
create_test_masks(mesh_dir, output_mask_dir)
