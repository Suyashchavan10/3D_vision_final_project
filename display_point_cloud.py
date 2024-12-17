# import open3d as o3d

# def view_point_cloud(file_path):
#     # Try reading the file as a point cloud
#     pcd = o3d.io.read_point_cloud(file_path)
    
#     # If the point cloud is successfully loaded, visualize it
#     if pcd.is_empty():
#         print("Error: Failed to load the .obj file as a point cloud.")
#     else:
#         print("Point cloud loaded successfully!")
#         o3d.visualization.draw_geometries([pcd])

# # Example usage
# file_path = r"C:\Users\suyash\IIITB\3D_vision_final_project\point_cloud_bottle.obj"  
# view_point_cloud(file_path)


import trimesh

def view_point_cloud(file_path):
    mesh = trimesh.load_mesh(file_path)
    mesh.show()

file_path = r"C:\Users\suyash\IIITB\3D_vision_final_project\point_cloud_bottle.obj"
# file_path = r"C:\Users\suyash\IIITB\3D_vision_final_project\colmap_experiments\colmap_output\bottle\0\points3D.bin"
view_point_cloud(file_path)
