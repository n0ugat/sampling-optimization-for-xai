#@title RELAX
import torch
import torch.nn as nn
from torch.fft import rfft as tfft
from torch.fft import irfft as tifft

from src.explainability.masking_freqrise import mask_generator

class FreqRISE(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 batch_size: int = 10,
                 num_batches: int = 300,
                 device: str = 'cpu',
                 use_softmax = False,
                 ):

        super().__init__()

        self.batch_size = batch_size
        self.device = device

        self.num_batches = num_batches
        self.use_softmax = use_softmax
        self.encoder = encoder.eval().to(self.device) # function that evaluates the pretrained model on a given input

    def forward(self, input_data, mask_generator, **kwargs):
        """
        Compute the saliency map of the input data using FreqRISE.
        Args:
            input_data: torch.Tensor
                The input data for which to compute the saliency map.
            mask_generator: function
                A function that generates masks for the input data.
            **kwargs: dict
                Additional keyword arguments to pass to the mask
                generator function.
        Returns:
            torch.Tensor: The saliency map of the input data.
        """
        i = 0 
        p = []
        input_data = input_data.unsqueeze(0).to(self.device)
        # cast data to domain of interest
        # Fast Fourier Transform (To frequency domain)
        input_fft = tfft(input_data)
        mask_type = torch.complex64

        # generate masks
        # num_batches * batch_size = number of masks to generate, not number of inputs to process
        for _ in range(self.num_batches):
            for masks in mask_generator(self.batch_size, input_fft.shape, self.device, dtype = mask_type, **kwargs):
            # for masks in mask_generator(self.batch_size, **kwargs):
                if len(masks) == 2: # mask_generator returns a tuple ????
                    x_mask, masks = masks
                    # Shapes are ???
                else:
                    # apply mask
                    x_mask = input_fft*masks
                    # cast back to original domain
                    x_mask = tifft(x_mask, dim=-1) # Inverse Fast Fourier Transform (To time domain)
                # Shape is (batch_size, 1, 8000) for AudioNet
                
                with torch.no_grad(): 
                    # get predictions of the model with the masked input
                    predictions = self.encoder(x_mask.float().to(self.device), only_feats = False).detach()
                if self.device == 'mps':
                    predictions = predictions.cpu()
                if self.use_softmax:
                    # Why use softmax here?
                    predictions = torch.nn.functional.softmax(predictions, dim=1)
                # compute saliency of the masked input
                sal = torch.matmul(predictions.transpose(0,1).float(), masks.view(self.batch_size, -1).abs().float()).transpose(0,1).unsqueeze(0).cpu()
                # sal has shape (1, 4001, 2)
                p.append(sal)
                i += 1
        importance = torch.cat(p, dim=0).sum(dim=0)/(self.num_batches*self.batch_size)
        return importance # Importance here is the saliency map
    
    def forward_dataloader(self, dataloader, num_cells, probability_of_drop):
        """
        Compute the saliency map of the input data using FreqRISE for a given dataloader.
        Args:
            dataloader: torch.utils.data.DataLoader
                The dataloader containing the input data for which to compute the saliency map.
            num_cells: int
                The number of cells in the grid.
            probability_of_drop: float
                The probability of dropping a cell.
        Returns:
            list: The saliency maps of the input data. ??? (I don't know)
        """
        freqrise_scores = [] # List to store the saliency maps ??? (I don't know)
        i = 0
            
        for data, target in dataloader:
            batch_scores = [] # List to store the saliency maps for the current batch ??? (I don't know)
            print("Computing batch", i+1, "/", len(dataloader))
            i+=1
            j=0
            for sample, y in zip(data, target):
                print("Computing signal", j+1, "/", len(data))
                m_generator = mask_generator
                with torch.no_grad(): 
                    importance = self.forward(sample.float().squeeze(0), 
                                              mask_generator = m_generator, 
                                              num_cells = num_cells, 
                                              probability_of_drop = probability_of_drop)
                
                # Selects the importance values for the given class y
                importance = importance.cpu().squeeze()[...,y]/probability_of_drop
                # min max normalize
                importance = (importance - importance.min()) / (importance.max() - importance.min())
                batch_scores.append(importance)
                j += 1
            freqrise_scores.append(torch.stack(batch_scores))
        return freqrise_scores