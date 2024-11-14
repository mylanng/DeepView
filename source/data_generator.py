import numpy as np
import os
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from scipy.ndimage import zoom

class DataGenerator(Sequence):
    def __init__(self, data_dir, volume_files, mask_files=None, batch_size=2, input_shape=(128, 128, 128)):
        self.data_dir = data_dir
        self.volume_files = volume_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.input_shape = input_shape
    
    def __len__(self):
        return len(self.volume_files) // self.batch_size
    
    def resize_volume(self, volume):
        # Calculate the zoom factors for each dimension
        zoom_factors = [self.input_shape[i] / volume.shape[i] for i in range(3)]
        return zoom(volume, zoom_factors, order=1)  # Use order=1 for bilinear interpolation

    def __getitem__(self, idx):
        batch_volumes = []
        batch_masks = []

        for i in range(self.batch_size):
            volume_path = os.path.join(self.data_dir, self.volume_files[idx * self.batch_size + i])
            volume = np.load(volume_path)
            volume = self.resize_volume(volume)  # Resize 3D volume
            batch_volumes.append(volume[..., np.newaxis])  # Add channel dimension

            if self.mask_files:
                mask_path = os.path.join(self.data_dir, self.mask_files[idx * self.batch_size + i])
                mask = np.load(mask_path)
                mask = self.resize_volume(mask)  # Resize 3D mask
                batch_masks.append(mask[..., np.newaxis])  # Add channel dimension

        if self.mask_files:
            return np.array(batch_volumes), np.array(batch_masks)
        else:
            return np.array(batch_volumes)
