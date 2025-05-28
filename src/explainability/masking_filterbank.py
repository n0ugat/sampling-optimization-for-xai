import torch
import torch.nn as nn
from torch.distributions import Bernoulli

class FilterbankMaskPolicy(nn.Module):
    def __init__(self, batch_size, shape, num_banks, device, dtype=torch.float32):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_banks), requires_grad=True)
        self.num_banks = num_banks
        self.batch_size = batch_size
        self.dtype = dtype
        self.shape = shape
        self.device = device

        self.pad_size = (num_banks // 2, num_banks // 2) # Padding size for the grid, half of the number of filterbanks.
        self.mask_shape = shape[:-1] + (num_banks,)
        self.length = self.mask_shape[-1]

    def forward(self):
        probabilities = torch.sigmoid(self.logits) # Apply sigmoid to the logits to get probabilities
        batch_distribution = Bernoulli(probabilities.unsqueeze(0).expand(self.batch_size, -1)) # Bernoulli distribution with a batch of probabilities
        grid = batch_distribution.sample() # Sample a batch of mask from the Bernoulli distribution
        log_probs = batch_distribution.log_prob(grid).sum(dim=1).to(self.device) # Log probability of the masks
        grid = grid.unsqueeze(1) # Convert grid from (batch_size, num_banks) to (batch_size, 1, num_banks)

        return grid.unsqueeze(1), log_probs # Return the mask and the log probability of the mask - masks is of shape (batch_size, 1, 1, num_banks)