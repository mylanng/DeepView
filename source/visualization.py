import numpy as np
import matplotlib.pyplot as plt

# Function to visualize slices
def visualize_mask_slices(mask_path, slice_axis=0):
    """
    Visualizes slices from a 3D mask file.

    Parameters:
    - mask_path (str): Path to the .npy mask file.
    - slice_axis (int): Axis along which to slice the mask (0, 1, or 2).
    """
    # Load the mask and squeeze to remove any singleton dimensions
    mask = np.load(mask_path).squeeze()

    # Calculate the middle slice along the specified axis
    middle_slice_index = mask.shape[slice_axis] // 2

    # Slice along the specified axis
    if slice_axis == 0:
        mask_slice = mask[middle_slice_index, :, :]
    elif slice_axis == 1:
        mask_slice = mask[:, middle_slice_index, :]
    elif slice_axis == 2:
        mask_slice = mask[:, :, middle_slice_index]
    else:
        print("Invalid slice axis. Choose 0, 1, or 2.")
        return

    # Plot the mask slice
    plt.imshow(mask_slice, cmap='gray')
    plt.title(f"Slice along axis {slice_axis}, index {middle_slice_index}")
    plt.axis('off')
    plt.show()

# Paths to the ground truth and prediction masks
gt_mask_path = "testing/masks/mask_scan_001.npy"  # Example path
pred_mask_path = "testing/predictions/prediction_001.npy"  # Example path

# Visualize the middle slice of each mask
print("Visualizing Ground Truth Mask:")
visualize_mask_slices(gt_mask_path, slice_axis=0)

print("Visualizing Predicted Mask:")
visualize_mask_slices(pred_mask_path, slice_axis=0)
