import trimesh
import matplotlib.pyplot as plt

def visualize_ply_file(ply_path):
    # Load the mesh
    mesh = trimesh.load(ply_path)

    # Display the mesh
    mesh.show()

    # Plot the vertices (optional - for checking structure in 2D)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], s=0.5)
    plt.title("3D Points in .ply file")
    plt.show()

# Path to the .ply file
ply_path = "testing/meshes/scan_001.ply"  # Example path

# Visualize the .ply file
visualize_ply_file(ply_path)
