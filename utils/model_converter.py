import numpy as np
import torch

# Load the .npz file
path = '../pretrained_pth/R50+ViT-B_16.npz'
npz_data = np.load(path)

# Example: Convert to PyTorch format
torch_data = {key: torch.tensor(npz_data[key]) for key in npz_data.files}

# Save as a PyTorch model
torch.save(torch_data, '../pretrained_pth/R50+ViT-B_16.pth')
