import torch
import torch.nn.functional as F

# Code from https://github.com/theabrusch/FreqRISE

def mask_generator(
    batch_size: int, # Number of masks to generate
    shape: tuple, # Shape of the input data
    device: str, # Device to use
    num_cells: int = 50, # Number of cells in the grid (Points to interpolate from)
    probability_of_drop: float = 0.5, # Probability of dropping a cell
    dtype = torch.float32, # Data type of the mask
    interpolation = 'linear'):
    """
    Generates a batch of masks by sampling Bernoulli random variables (probablity_of_drop) in a lower dimensional grid (num_cells)
    and upsamples the discrete masks using linear interpolation to obtain smooth continious mask in (0, 1).
    """
    length = shape[-1] # Length of the input data
    pad_size = (num_cells // 2, num_cells // 2) # Padding size for the grid
    
    # Generate a grid of Bernoulli random variables
    grid = (torch.rand(batch_size, 1, *((num_cells,))) < probability_of_drop).float().to(device)
    
    # Upsample the grid using linear interpolation
    grid_up = F.interpolate(grid, size=length, mode=interpolation, align_corners=False)

    # Pad the grid with reflection and sample a shift in the x and y directions
    if device == 'mps':
        grid_up = grid_up.to('cpu')
        grid_up = F.pad(grid_up, pad_size, mode='reflect')
        shift_x = torch.randint(0, num_cells, (batch_size,))
        shift_y = torch.randint(0, num_cells, (batch_size,))
        masks = torch.empty((batch_size, *shape), dtype = dtype)
    else:
        grid_up = F.pad(grid_up, pad_size, mode='reflect') # Pad the grid with reflection, meaning the values at the edges are reflected
        shift_x = torch.randint(0, num_cells, (batch_size,), device=device) # Randomly sample a shift in the x-direction
        shift_y = torch.randint(0, num_cells, (batch_size,), device=device) # Randomly sample a shift in the y-direction
        masks = torch.empty((batch_size, *shape), device=device, dtype = dtype) # Initialize the masks tensor

    # Generate the masks
    for mask_i in range(batch_size):
        # Extract the mask from the grid with the correct shift and length
        masks[mask_i] = grid_up[
            mask_i,
            :,
            shift_x[mask_i]:shift_x[mask_i] + length # 
            ]
    yield masks.to(device)