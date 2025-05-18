#@title RELAX
import torch
import torch.nn as nn
from torch.fft import rfft as tfft
from torch.fft import irfft as tifft

import time
import os
import pickle

from src.explainability.masking_freqrise import mask_generator

class FreqRISE(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 batch_size: int = 10,
                 num_batches: int = 300,
                 device: str = 'cpu',
                 use_softmax = False,
                 save_signals_path = None
                 ):

        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.save_signals_path = save_signals_path

        self.num_batches = num_batches
        self.use_softmax = use_softmax
        self.encoder = encoder.eval().to(self.device) # function that evaluates the pretrained model on a given input

    def forward(self, input_data, target_class, probability_of_drop, mask_generator, num_cells, idx):
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
        p = []
        input_data = input_data.unsqueeze(0).to(self.device)
        # Fast Fourier Transform (To frequency domain)
        input_fft = tfft(input_data)
        mask_type = torch.complex64

        if self.save_signals_path:
            with torch.no_grad(): 
                # get predictions of the model with the original input
                pred_original = self.encoder(input_data.unsqueeze(0).float(), only_feats = False).detach().squeeze()
                if self.use_softmax:
                    pred_original = torch.softmax(pred_original, dim=-1)
            start_time = time.time()
        # generate masks
        for _ in range(self.num_batches):
            for masks in mask_generator(self.batch_size, input_fft.shape, self.device, num_cells, probability_of_drop, mask_type, 'linear'):
                # apply mask
                x_mask = input_fft*masks
                # cast back to original domain
                x_mask = tifft(x_mask, dim=-1) # Inverse Fast Fourier Transform (To time domain)
                
                with torch.no_grad(): 
                    # get predictions of the model with the masked input
                    predictions = self.encoder(x_mask.float(), only_feats = False).detach()
                if self.use_softmax:
                    # Why use softmax here?
                    predictions = torch.nn.functional.softmax(predictions, dim=1)
                if self.device == 'mps':
                    predictions = predictions.cpu()
                # compute saliency of the masked input
                sal = torch.matmul(predictions.transpose(0,1).float(), masks.view(self.batch_size, -1).abs().float()).transpose(0,1).unsqueeze(0)
                p.append(sal)
        importance = torch.cat(p, dim=0).sum(dim=0)/(self.num_batches*self.batch_size)
        # Selects the importance values for the given class y
        importance = importance.cpu().squeeze()[...,target_class]/probability_of_drop
        # min max normalize
        importance = (importance - importance.min()) / (importance.max() - importance.min())
        if self.save_signals_path:
            run_time = time.time() - start_time
            output_dict = {
                'signal': input_data.squeeze().cpu(),
                'signal_fft' : torch.abs(input_fft).squeeze().cpu(),
                'target_class': target_class.cpu(),
                'prediction' : torch.argmax(pred_original).cpu(),
                'importance': importance.cpu(),
                'run_time' : run_time,
            }
            sample_path = os.path.join(self.save_signals_path, f'sample_{idx}.pkl')
            with open(sample_path, mode='wb') as f:
                pickle.dump(output_dict, f)
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
            for j, (sample, y) in enumerate(zip(data, target)):
                print("Computing signal", j+1, "/", len(data))
                m_generator = mask_generator
                sample = sample.to(self.device)
                y = y.to(self.device)
                
                with torch.no_grad(): 
                    importance = self.forward(input_data = sample.float().squeeze(0), 
                                              target_class = y,
                                              probability_of_drop = probability_of_drop,
                                              mask_generator = m_generator, 
                                              num_cells = num_cells,
                                              idx = j)
                
                
                batch_scores.append(importance)
            freqrise_scores.append(torch.stack(batch_scores))
        return freqrise_scores