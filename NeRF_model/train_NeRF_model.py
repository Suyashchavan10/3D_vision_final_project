import os
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from nerf import NeRFModel  # Assuming you have a NeRF model in the nerf.py

# Define your dataset class
class NeRFDataset(Dataset):
    def __init__(self, points_file, cameras_file, resolution=(800, 600)):
        # Load points and camera data
        self.points3D = self.load_points3D(points_file)
        self.cameras = self.load_cameras(cameras_file)
        self.resolution = resolution

    def load_points3D(self, points_file):
        # Parse the points3D.txt file (adjust according to the actual file format)
        points = []
        with open(points_file, 'r') as f:
            for line in f.readlines():
                parts = line.split()
                if len(parts) > 7:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])
        return np.array(points)

    def load_cameras(self, cameras_file):
        # Parse the cameras.txt file (adjust according to the actual file format)
        cameras = []
        with open(cameras_file, 'r') as f:
            for line in f.readlines():
                parts = line.split()
                if len(parts) > 3:
                    # Extract camera parameters (location, orientation, etc.)
                    # Here, assuming the first three values are the camera position
                    camera_position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    cameras.append(camera_position)
        return np.array(cameras)

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        # Return camera pose and corresponding image data
        camera = self.cameras[idx]
        # You would need to fetch the corresponding 2D projection of the points (image)
        # For simplicity, assume a placeholder image is generated here
        image = np.random.rand(*self.resolution, 3)  # Placeholder random image
        return torch.tensor(camera, dtype=torch.float32), torch.tensor(image, dtype=torch.float32)

# Initialize dataset and dataloader
points_file = r"C:\Users\suyash\IIITB\3D_vision_final_project\colmap_experiments\colmap_output\bottle\txt_model\points3D.txt"
cameras_file = r"C:\Users\suyash\IIITB\3D_vision_final_project\colmap_experiments\colmap_output\bottle\txt_model\cameras.txt"
dataset = NeRFDataset(points_file, cameras_file)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize the NeRF model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeRFModel().to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = torch.nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for camera_poses, images in dataloader:
        camera_poses = camera_poses.to(device)
        images = images.to(device)

        optimizer.zero_grad()

        # Forward pass (You would need to implement the NeRF forward pass)
        outputs = model(camera_poses)  # Placeholder for the actual NeRF forward pass

        # Compute loss (MSE loss between generated image and actual image)
        loss = criterion(outputs, images)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), r"C:\Users\suyash\IIITB\3D_vision_final_project\NeRF_model\nerf_model.pth")

# Inference - Render a new view
with torch.no_grad():
    test_camera_pose = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device)  # Example camera pose
    rendered_image = model(test_camera_pose)
    plt.imshow(rendered_image.cpu().numpy())
    plt.show()
