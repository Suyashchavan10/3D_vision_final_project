import numpy as np
import matplotlib.pyplot as plt

def load_points3D(file_path):
    points = []
    
    # Read the points3D.txt file
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components
            parts = line.strip().split()
            
            # Skip lines that don't have enough data (skip headers or malformed lines)
            if len(parts) < 7:
                continue
            
            try:
                # Extract 3D coordinates (X, Y, Z)
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points.append([x, y, z])
            except ValueError:
                # Skip lines that can't be converted to float (e.g., header lines)
                continue

    return np.array(points)

def plot_point_cloud(points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='jet')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set the title
    ax.set_title('3D Point Cloud')
    
    plt.show()

# File path to the points3D.txt
file_path = r"C:\Users\suyash\IIITB\3D_vision_final_project\colmap_experiments\colmap_output\bottle\txt_model\points3D.txt"

# Load the points from the file
points = load_points3D(file_path)

# Plot the point cloud
plot_point_cloud(points)
