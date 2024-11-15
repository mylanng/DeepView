import tensorflow as tf
import numpy as np
import trimesh
import os
from scipy.ndimage import zoom

# Chamfer Distance Function
def chamfer_distance(predicted, ground_truth):
    """
    Computes the Chamfer Distance between two point clouds.
    """
    dists_pred_to_gt = tf.reduce_min(
        tf.reduce_sum((tf.expand_dims(predicted, axis=1) - tf.expand_dims(ground_truth, axis=0))**2, axis=-1), axis=1
    )
    dists_gt_to_pred = tf.reduce_min(
        tf.reduce_sum((tf.expand_dims(ground_truth, axis=1) - tf.expand_dims(predicted, axis=0))**2, axis=-1), axis=1
    )
    chamfer = tf.reduce_mean(dists_pred_to_gt) + tf.reduce_mean(dists_gt_to_pred)
    return chamfer.numpy()

# Hausdorff Distance Function
def hausdorff_distance(predicted, ground_truth):
    """
    Computes the Hausdorff Distance between two point clouds.
    """
    dists_pred_to_gt = tf.reduce_min(
        tf.reduce_sum((tf.expand_dims(predicted, axis=1) - tf.expand_dims(ground_truth, axis=0))**2, axis=-1), axis=1
    )
    dists_gt_to_pred = tf.reduce_min(
        tf.reduce_sum((tf.expand_dims(ground_truth, axis=1) - tf.expand_dims(predicted, axis=0))**2, axis=-1), axis=1
    )
    hausdorff = max(tf.reduce_max(dists_pred_to_gt).numpy(), tf.reduce_max(dists_gt_to_pred).numpy())
    return hausdorff

# Ensure the custom model class is defined properly
class VolumetricToMeshModel(tf.keras.Model):
    def __init__(self, latent_dim=1024, num_vertices=10000, **kwargs):
        super(VolumetricToMeshModel, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_vertices = num_vertices

        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Conv3D(16, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D(pool_size=2, strides=2),

            tf.keras.layers.Conv3D(32, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D(pool_size=2, strides=2),

            tf.keras.layers.Conv3D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D(pool_size=2, strides=2),

            tf.keras.layers.Conv3D(128, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling3D()
        ])

        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim, activation="relu"),
            tf.keras.layers.Dense(num_vertices * 3)
        ])

    def call(self, inputs):
        x = self.backbone(inputs)
        x = self.fc(x)
        return tf.reshape(x, (-1, tf.shape(x)[1] // 3, 3))

    def get_config(self):
        config = super(VolumetricToMeshModel, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "num_vertices": self.num_vertices
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load the saved model
model = tf.keras.models.load_model(
    "DeepView_code/Harry/volumetric_to_mesh_model.keras",
    custom_objects={"VolumetricToMeshModel": VolumetricToMeshModel}
)

# Testing data directories
test_volumes_dir = "DeepView_code/testing/volumes"
test_meshes_dir = "DeepView_code/testing/meshes"

# Prepare test dataset
def preprocess(volume_path, mesh_path):
    with open(volume_path, 'rb') as f:
        volume = np.frombuffer(f.read(), dtype=np.uint16).reshape((768, 768, 1280))
    volume = volume / np.max(volume)
    zoom_factors = [128 / volume.shape[0], 128 / volume.shape[1], 128 / volume.shape[2]]
    volume = zoom(volume, zoom_factors, order=1)
    volume = tf.expand_dims(volume, axis=-1)
    mesh = trimesh.load(mesh_path, process=False)
    vertices = mesh.vertices.astype(np.float32)
    vertices = vertices - np.mean(vertices, axis=0)
    vertices = vertices / np.max(np.linalg.norm(vertices, axis=1))
    return volume, vertices

def test_generator():
    volume_files = sorted([os.path.join(test_volumes_dir, f) for f in os.listdir(test_volumes_dir) if f.endswith('.raw')])
    mesh_files = sorted([os.path.join(test_meshes_dir, f) for f in os.listdir(test_meshes_dir) if f.endswith('.ply')])
    for vol_path, mesh_path in zip(volume_files, mesh_files):
        yield preprocess(vol_path, mesh_path)

test_dataset = tf.data.Dataset.from_generator(
    test_generator,
    output_signature=(
        tf.TensorSpec(shape=(128, 128, 128, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
    )
).batch(1)

# Evaluate metrics
chamfer_scores = []
hausdorff_scores = []

for volume_batch, gt_mesh_batch in test_dataset:
    predicted_mesh_batch = model.predict(volume_batch)
    for predicted, ground_truth in zip(predicted_mesh_batch, gt_mesh_batch):
        chamfer_scores.append(chamfer_distance(predicted, ground_truth))
        hausdorff_scores.append(hausdorff_distance(predicted, ground_truth))

# Calculate average scores
avg_chamfer = np.mean(chamfer_scores)
avg_hausdorff = np.mean(hausdorff_scores)

print(f"Average Chamfer Distance: {avg_chamfer}")
print(f"Average Hausdorff Distance: {avg_hausdorff}")
