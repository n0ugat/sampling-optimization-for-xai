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
        self.pad_size = (num_cells // 2, num_cells // 2)

    def forward(self):
        """
        Generate a batch of masks by sampling Bernoulli random variables (probability_of_drop) in a lower dimensional grid (num_cells)
        and upsamples the discrete masks using linear interpolation to obtain smooth continuous mask in (0, 1).
        """
        probabilities = torch.sigmoid(self.logits) # Apply sigmoid to the logits to get probabilities
        batch_distribution = Bernoulli(probabilities.unsqueeze(0).expand(self.batch_size, -1)) # Bernoulli distribution with a batch of probabilities
        grid = batch_distribution.sample() # Sample a batch of mask from the Bernoulli distribution
        log_probs = batch_distribution.log_prob(grid).sum(dim=1).to(self.device) # Log probability of the masks
        
        grid = grid.unsqueeze(1) 
        grid_up = F.interpolate(grid, size=self.shape[-1], mode="linear", align_corners=False)

        return grid_up, log_probs # Return the mask and the log probability of the mask