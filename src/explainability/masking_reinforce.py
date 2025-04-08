import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class MaskPolicy(nn.Module):
    def __init__(self, batch_size, shape, num_cells, device, dtype):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_cells), requires_grad=True)
        self.num_cells = num_cells
        self.batch_size = batch_size
        self.dtype = dtype
        self.shape = shape
        self.device = device
        self.pad_size = (num_cells // 2, num_cells // 2) # Padding size for the grid, half of the number of cells.

    def forward(self):
        probabilities = torch.sigmoid(self.logits) # Apply sigmoid to the logits to get probabilities
        batch_distribution = Bernoulli(probabilities.unsqueeze(0).expand(self.batch_size, -1)) # Bernoulli distribution with a batch of probabilities
        grid = batch_distribution.sample() # Sample a batch of mask from the Bernoulli distribution
        log_probs = batch_distribution.log_prob(grid).sum(dim=1) # Log probability of the masks
        grid = grid.unsqueeze(1) # Convert grid from (batch_size, num_cells) to (batch_size, 1, num_cells)
        grid_up = F.interpolate(grid, size=self.shape[-1], mode="linear", align_corners=False)
        # Pad the grid with reflection and sample a shift in the x and y directions
        if self.device == 'mps':
            grid_up = F.pad(grid_up, self.pad_size, mode='reflect')
            shift_x = torch.randint(0, self.num_cells, (self.batch_size,))
            masks = torch.empty((self.batch_size, *self.shape), dtype = self.dtype)
        else:
            grid_up = F.pad(grid_up, self.pad_size, mode='reflect').to(self.device) # Pad the grid with reflection, meaning the values at the edges are reflected
            shift_x = torch.randint(0, self.num_cells, (self.batch_size,), device=self.device) # Randomly sample a shift in the x-direction
            masks = torch.empty((self.batch_size, *self.shape), device=self.device, dtype = self.dtype) # Initialize the masks tensor
        # Generate the masks
        for mask_i in range(self.batch_size):
            # Extract the mask from the grid with the correct shift and length
            masks[mask_i] = grid_up[mask_i, :, shift_x[mask_i]:shift_x[mask_i] + self.shape[-1]]
        return masks, log_probs # Return the mask and the log probability of the mask