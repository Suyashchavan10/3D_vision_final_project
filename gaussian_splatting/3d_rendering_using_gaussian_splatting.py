import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math

# Load .obj file and extract points
def load_obj(obj_file_path):
    """
    Load a point cloud from a .obj file and extract points and optionally colors.
    """
    mesh = o3d.io.read_triangle_mesh(obj_file_path)
    if not mesh.is_empty():
        # Convert the mesh to point cloud
        point_cloud = mesh.sample_points_uniformly(number_of_points=10000)  # adjust the number of points as needed
        return np.asarray(point_cloud.points), np.asarray(point_cloud.colors)
    else:
        raise ValueError("The .obj file is empty or not a valid mesh!")

# 3D Gaussian splatting function
def gaussian_splatting(coords, radius=0.01, intensity=1.0):
    """
    Apply Gaussian splatting on 3D points to create a smooth rendering.
    :param coords: (N, 3) array of 3D points.
    :param radius: Radius (standard deviation) of the Gaussian blobs.
    :param intensity: Intensity (height) of the splat.
    :return: Projected 2D points and their intensities.
    """
    # Projecting points to a 2D view (simple orthogonal projection)
    projected_2d_points = coords[:, :2]  # Ignore the Z-axis for the 2D projection
    radii = np.full(len(coords), radius)  # Same radius for all points, could vary per point
    return projected_2d_points, radii

# Render the point cloud with Gaussian splatting
def render_gaussian_splatting(projected_points, radii, intensity=1.0):
    """
    Render the 2D Gaussian splats on a canvas (matplotlib for simplicity).
    """
    fig, ax = plt.subplots()
    for pt, radius in zip(projected_points, radii):
        ax.scatter(pt[0], pt[1], s=radius * 1000, c='b', alpha=0.5)  # Scatter plot with adjusted size for radius
    ax.set_aspect('equal', 'box')
    plt.show()

def main():
    # Load the .obj file and extract points and colors
    obj_file_path = r"C:\Users\suyash\IIITB\3D_vision_final_project\point_cloud_bottle.obj"  # Replace with the path to your .obj file
    try:
        coords, colors = load_obj(obj_file_path)
    except ValueError as e:
        print(e)
        return

    # Apply Gaussian Splatting to project points
    projected_points, radii = gaussian_splatting(coords)

    # Render the point cloud with Gaussian splats
    render_gaussian_splatting(projected_points, radii)

if __name__ == "__main__":
    main()
