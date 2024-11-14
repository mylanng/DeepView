import tensorflow as tf
from model import unet_3d
from data_generator import DataGenerator
import os
import numpy as np

# Define directories
data_dir = "output_masks"
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
input_shape = (128, 128, 128, 1)
batch_size = 2
epochs = 20

# Prepare file lists for labeled and unlabeled data
volume_files = sorted([f for f in os.listdir(data_dir) if f.startswith("augmented_scan") and f.endswith(".npy")])
mask_files = sorted([f for f in os.listdir(data_dir) if f.startswith("mask_scan") and f.endswith(".npy")])
unlabeled_files = sorted([f for f in os.listdir(data_dir) if f.startswith("augmented_unlabeled_scan") and f.endswith(".npy")])

# Initialize data generators
train_generator = DataGenerator(data_dir, volume_files, mask_files, batch_size=batch_size, input_shape=input_shape[:3])

# Split train_generator for validation if necessary
# Here we can split `volume_files` and `mask_files` into training and validation sets if needed.
val_generator = DataGenerator(data_dir, volume_files, mask_files, batch_size=batch_size, input_shape=input_shape[:3])

# Initialize and compile the model
model = unet_3d(input_shape=input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Starting training...")
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save the model
model.save("3d_unet_model.h5")
print("Model saved as '3d_unet_model.h5'")

# Prediction on unlabeled test data
print("Generating predictions on test data...")
test_generator = DataGenerator(data_dir, unlabeled_files, batch_size=1, input_shape=input_shape[:3])

for i, test_volume in enumerate(test_generator):
    pred = model.predict(test_volume)
    np.save(os.path.join(output_dir, f"prediction_{i}.npy"), pred)
    print(f"Saved prediction {i} to {output_dir}")
