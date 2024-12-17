import struct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_colmap_points3D_bin(points3D_file):
    """
    Reads the binary COLMAP points3D.bin file and extracts 3D points and their colors.
    """
    points = []
    colors = []

    with open(points3D_file, "rb") as f:
        while True:
            # Read the binary chunk for a single point
            binary_chunk = f.read(43)  # Expected size of each point entry
            if not binary_chunk:
                break  # End of file reached

            try:
                # Unpack the binary data
                unpacked_data = struct.unpack("QfffBBBf", binary_chunk[:28])
                _, x, y, z, r, g, b, _ = unpacked_data

                # Append data
                points.append([x, y, z])
                colors.append([r / 255.0, g / 255.0, b / 255.0])  # Normalize RGB to [0, 1]

            except struct.error:
                print("Error unpacking data. File may not match expected format.")
                break

    return np.array(points), np.array(colors)


def visualize_point_cloud(points, colors):
    """
    Visualizes the 3D points using matplotlib.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Point Cloud")
    plt.show()


if __name__ == "__main__":
    # Path to the COLMAP points3D.bin file
    points3D_bin_path = r"C:\Users\suyash\IIITB\3D_vision_final_project\colmap_experiments\colmap_output\bottle\0\points3D.bin"

    # Read the points and colors
    points, colors = read_colmap_points3D_bin(points3D_bin_path)

    # Print some details for debugging
    print(f"Loaded {len(points)} 3D points.")
    if len(points) > 0:
        print(f"First point: {points[0]}, Color: {colors[0]}")

    # Visualize the point cloud
    visualize_point_cloud(points, colors)