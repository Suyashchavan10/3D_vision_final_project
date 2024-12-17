import torch
import torch.nn as nn

class Voxel(nn.Module):
    def __init__(self, nb_voxels=100, scale=1, device='cpu'):
        super(Voxel, self).__init__()
        self.nb_voxels = nb_voxels
        self.scale = scale
        self.device = device

        self.voxels = torch.nn.Parameter(torch.rand((nb_voxels, nb_voxels, nb_voxels, 4), device=self.device))

    def forward(self, xyz, d):

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        cond = (x.abs() < (self.scale / 2)) & (y.abs() < (self.scale / 2)) & (z.abs() < (self.scale / 2))
        
        indx = (x[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        indy = (y[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        indz = (z[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)

        
        
        colors_and_densities = torch.zeros((xyz.shape[0], 4), device=xyz.device)
        colors_and_densities[cond, :3] = self.voxels[indx, indy, indz, :3]
        colors_and_densities[cond, -1] = self.voxels[indx, indy, indz, -1]

        return torch.sigmoid(colors_and_densities[:, :3]), torch.relu(colors_and_densities[:, -1:])

    def intersect(self, x, d):
        return self.forward(x, d)


class NeRF(nn.Module):
    def __init__(self, Lpos=10, Ldir=4, hidden_dim=256, device='cpu'):
        super(NeRF, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(Lpos*6 + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + Lpos*6+3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )

        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + Ldir * 6 + 3, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3), nn.Sigmoid()
        )

        self.Lpos = Lpos
        self.Ldir = Ldir

    def positional_encoding(self, x, L):
        out = [x]
        for i in range(L):
            out.append(torch.sin(2 ** i * x))
            out.append(torch.cos(2 ** i * x))
        return torch.cat(out, dim=1)


    def forward(self, xyz, d):
        x_emb = self.positional_encoding(xyz, self.Lpos)
        d_emb = self.positional_encoding(d, self.Ldir) 

        h = self.block1(x_emb)
        h = self.block2(torch.cat((h, x_emb), dim=1))
        sigma = h[:, -1]
        h = h[:, :-1]
        c = self.rgb_head(torch.cat((h, d_emb), dim=1))

        return c, torch.relu(sigma)

    def intersect(self, x, d):
        return self.forward(x, d)