import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

def filter_mask_generator(
        batch_size: int, # Number of masks to generate
        shape: tuple, # Shape of the input data
        num_banks: int, # Number of frequency bands in the filter bank
        probability_of_drop: float, # Probability of dropping a frequency band
        dtype: torch.dtype = torch.float32, # Data type of the mask
):
        pad_size = (num_banks // 2, num_banks // 2)
        mask_shape = shape[:-1] + (num_banks,)
        length = mask_shape[-1]

        grid = (torch.rand(batch_size, 1, *(num_banks,)) < probability_of_drop).float()
        grid = F.pad(grid, pad_size, mode='reflect')
        shift = torch.randint(low=0, high=num_banks, size=(batch_size,))
        masks = torch.empty((batch_size, *mask_shape), dtype=dtype)

        for mask in range(batch_size):
                masks[mask] = grid[mask, :, shift[mask]:shift[mask] + length]
        
        yield masks

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
        log_probs = batch_distribution.log_prob(grid).sum(dim=1) # Log probability of the masks
        grid = grid.unsqueeze(1) # Convert grid from (batch_size, num_banks) to (batch_size, 1, num_banks)

        # Pad the grid with reflection and sample a shift
        if self.device == 'mps':
            grid = F.pad(grid, self.pad_size, mode='reflect') # Pad the grid with reflection, meaning the values at the edges are reflected
            shift = torch.randint(0, self.num_banks, size=(self.batch_size,)) # Randomly sample a shift
            masks = torch.empty((self.batch_size, *self.mask_shape), dtype = self.dtype) # Initialize the masks tensor
        else:
            grid = F.pad(grid, self.pad_size, mode='reflect').to(self.device) # Pad the grid with reflection, meaning the values at the edges are reflected
            shift = torch.randint(0, self.num_banks, size=(self.batch_size,), device=self.device) # Randomly sample a shift
            masks = torch.empty((self.batch_size, *self.mask_shape), device=self.device, dtype = self.dtype) # Initialize the masks tensor

        # Generate the masks
        for mask_i in range(self.batch_size):
            # Extract the mask from the grid with the correct shift and length
            masks[mask_i] = grid[mask_i, :, shift[mask_i]:shift[mask_i] + self.length]

        return masks, log_probs # Return the mask and the log probability of the mask - masks is of shape (batch_size, 1, 1, num_banks)