import numpy as np
import os
import tensorflow as tf

# Define Dice coefficient function
def dice_coefficient(y_true, y_pred, smooth=1e-5):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# Directories for ground truth masks and predictions
ground_truth_dir = "testing/masks"
prediction_dir = "testing/predictions"

# Target shape for resizing (adjust based on model's expected output)
target_shape = (128, 128, 128)

# Function to resize 3D volume slice by slice
def resize_3d_volume(volume, target_shape):
    resized_slices = []
    for slice in volume:
        if slice.ndim == 2:  # Check if the slice is 2D
            slice = np.expand_dims(slice, axis=-1)  # Add a channel dimension to make it 3D
        resized_slice = tf.image.resize(slice, target_shape[:2], method='nearest').numpy()
        resized_slice = resized_slice[..., 0]  # Remove the channel dimension after resizing
        resized_slices.append(resized_slice)
    return np.stack(resized_slices, axis=0)

# Calculate Dice coefficient for each test sample
dice_scores = []
for i in range(1, 11):  # Assuming 10 test files
    # Load ground truth mask with leading zeros in the filename
    gt_mask = np.load(os.path.join(ground_truth_dir, f"mask_scan_{i:03d}.npy"))
    pred_mask = np.load(os.path.join(prediction_dir, f"prediction_{i:03d}.npy"))
    
    # Remove batch and channel dimensions from the prediction if they exist
    if pred_mask.ndim == 5:  # Shape is (1, 128, 128, 128, 1)
        pred_mask = np.squeeze(pred_mask)  # Remove dimensions to make it (128, 128, 128)
    
    print(f"Original shapes - GT: {gt_mask.shape}, Prediction: {pred_mask.shape}")
    
    # Resize the ground truth to match the prediction dimensions
    gt_mask_resized = resize_3d_volume(gt_mask, target_shape)
    
    # Crop the ground truth if it has more depth than the prediction
    if gt_mask_resized.shape[0] > target_shape[0]:
        gt_mask_resized = gt_mask_resized[:target_shape[0], :, :]  # Crop depth
    
    print(f"Resized shapes - GT: {gt_mask_resized.shape}, Prediction: {pred_mask.shape}")
    
    # Threshold the predicted mask to create a binary mask
    pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
    
    # Calculate Dice coefficient
    dice_score = dice_coefficient(gt_mask_resized, pred_mask_binary)
    dice_scores.append(dice_score)
    print(f"Dice coefficient for scan_{i:03d}: {dice_score:.4f}")

# Print the average Dice coefficient
average_dice = np.mean(dice_scores)
print(f"Average Dice coefficient on the test set: {average_dice:.4f}")
