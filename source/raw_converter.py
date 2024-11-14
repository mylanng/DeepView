import os
import numpy as np

# Define paths
raw_data_dir = "testing/volumes"
output_npy_dir = "testing/volumes_npy"
os.makedirs(output_npy_dir, exist_ok=True)

# Define dimensions of each volume (adjust this based on your actual `.raw` data dimensions)
# Set the expected dimensions
volume_dimensions = (1280, 768, 768)
dtype = np.uint16  # Example data type; adjust based on actual data

# Convert each .raw file to .npy
for raw_file in sorted(os.listdir(raw_data_dir)):
    if raw_file.endswith(".raw"):
        raw_path = os.path.join(raw_data_dir, raw_file)
        npy_path = os.path.join(output_npy_dir, raw_file.replace(".raw", ".npy"))

        # Load and reshape the .raw file
        volume = np.fromfile(raw_path, dtype=dtype)
        volume = volume.reshape(volume_dimensions)
        np.save(npy_path, volume)
        print(f"Converted {raw_file} to {npy_path}")
