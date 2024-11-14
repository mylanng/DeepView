import os
import numpy as np
from data_generator import DataGenerator
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("3d_unet_model.h5", compile=False)

# Define directories
data_dir = "testing/volumes_npy" 
output_dir = "testing/predictions"
os.makedirs(output_dir, exist_ok=True)

# Prepare list of test volume files
test_volume_files = sorted([f for f in os.listdir(data_dir) if f.startswith("scan") and f.endswith(".npy")])

# Generate predictions
print("Generating predictions on test data...")
test_generator = DataGenerator(data_dir, test_volume_files, batch_size=1, input_shape=(128, 128, 128))

# Generate predictions only for files in test_volume_files
for i, test_volume in enumerate(test_generator):
    pred = model.predict(test_volume)
    np.save(os.path.join(output_dir, f"prediction_{i+1:03d}.npy"), pred)
    print(f"Saved prediction for scan_{i+1:03d} to {output_dir}")
