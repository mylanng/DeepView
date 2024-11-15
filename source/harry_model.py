import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import trimesh
import os
from scipy.ndimage import zoom

class VolumetricToMeshModel(tf.keras.Model):
    def __init__(self, latent_dim=1024, num_vertices=10000):
        super(VolumetricToMeshModel, self).__init__()

        # 3D-CNN Backbone
        self.backbone = models.Sequential([
            layers.Conv3D(16, kernel_size=3, strides=1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling3D(pool_size=2, strides=2),

            layers.Conv3D(32, kernel_size=3, strides=1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling3D(pool_size=2, strides=2),

            layers.Conv3D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling3D(pool_size=2, strides=2),

            layers.Conv3D(128, kernel_size=3, strides=1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling3D()
        ])

        # Fully connected layers for regression
        self.fc = models.Sequential([
            layers.Dense(latent_dim, activation="relu"),
            layers.Dense(num_vertices * 3)  # Predict x, y, z for each vertex
        ])

    def call(self, inputs):
        x = self.backbone(inputs)
        x = self.fc(x)
        return tf.reshape(x, (-1, tf.shape(x)[1] // 3, 3))  # Output shape: (batch_size, num_vertices, 3)

class VolumetricMeshDataset(tf.data.Dataset):
    def __new__(cls, volumes_dir, meshes_dir, input_shape=(128, 128, 128), num_vertices=10000):
        # Get file paths
        volume_files = sorted([os.path.join(volumes_dir, f) for f in os.listdir(volumes_dir) if f.endswith('.raw')])
        mesh_files = sorted([os.path.join(meshes_dir, f) for f in os.listdir(meshes_dir) if f.endswith('.ply')])

        def preprocess(volume_path, mesh_path):
            # Load and normalize volume
            with open(volume_path, 'rb') as f:
                volume = np.frombuffer(f.read(), dtype=np.uint16).reshape((768, 768, 1280))
            volume = volume / np.max(volume)  # Normalize to [0, 1]
            
            # Resize volume to input_shape
            zoom_factors = [input_shape[0] / volume.shape[0], 
                            input_shape[1] / volume.shape[1], 
                            input_shape[2] / volume.shape[2]]
            volume = zoom(volume, zoom_factors, order=1)  # Bilinear interpolation
            volume = tf.expand_dims(volume, axis=-1)  # Add channel dimension

            # Load mesh
            mesh = trimesh.load(mesh_path, process=False)
            vertices = mesh.vertices.astype(np.float32)
            vertices = vertices - np.mean(vertices, axis=0)  # Center vertices
            vertices = vertices / np.max(np.linalg.norm(vertices, axis=1))  # Normalize to unit sphere

            # Pad vertices if fewer than num_vertices
            if vertices.shape[0] < num_vertices:
                padding = np.zeros((num_vertices - vertices.shape[0], 3), dtype=np.float32)
                vertices = np.vstack([vertices, padding])
            else:
                vertices = vertices[:num_vertices]

            return volume, vertices

        def generator():
            for vol_path, mesh_path in zip(volume_files, mesh_files):
                yield preprocess(vol_path, mesh_path)

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(input_shape[0], input_shape[1], input_shape[2], 1), dtype=tf.float32),
                tf.TensorSpec(shape=(num_vertices, 3), dtype=tf.float32)
            )
        )

        return dataset

# Example usage
if __name__ == "__main__": 
    volumes_dir = "training/volumes"
    meshes_dir = "DeepView_code/training/meshes"

    dataset = VolumetricMeshDataset(volumes_dir, meshes_dir)
    train_size = int(0.8 * len(list(dataset)))
    test_size = len(list(dataset)) - train_size

    train_dataset = dataset.take(train_size).batch(2).prefetch(tf.data.AUTOTUNE)
    test_dataset = dataset.skip(train_size).batch(2).prefetch(tf.data.AUTOTUNE)

    # Create model
    model = VolumetricToMeshModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=["mse"])

    # Fit the model
    model.fit(train_dataset, epochs=1, validation_data=test_dataset)
    model.save("DeepView_code/Harry/volumetric_to_mesh_model.keras")


