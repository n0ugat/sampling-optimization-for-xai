#@title RELAX
from torch.fft import rfft as tfft
from torch.fft import irfft as tifft
import torch
import torch.nn as nn
from src.explainability.masking import mask_generator

# import pickle

def plot(importances, sample_idx, digit: bool = False, identifier: str = ''):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(importances, marker='o')
    ax.set_title('Importances Plot')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('Importance Value')
    ax.grid()
    plt.tight_layout()
    plt.savefig(f'outputs/importances_plot_{"digit_" if digit else ""}{sample_idx}_{identifier}.png')
    plt.close()

class FreqRISE(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 batch_size: int = 10,
                 num_batches: int = 300,
                 device: str = 'cpu',
                 domain = 'fft',
                 use_softmax = False,
                 stft_params = None,
                 ):

        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.domain = domain
        self.stft_params = stft_params

        self.num_batches = num_batches
        self.use_softmax = use_softmax
        self.encoder = encoder.eval().to(self.device) # function that evaluates the pretrained model on a given input

    def forward(self, input_data, mask_generator, **kwargs) -> None:
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
        if self.domain == 'fft':
            # Fast Fourier Transform (To frequency domain)
            input_fft = tfft(input_data) 
        else:
            input_fft = input_data
            
        shape = input_fft.shape

        mask_type = torch.complex64 if self.domain == 'fft' else torch.float32

        # generate masks
        # num_batches * batch_size = number of masks to generate, not number of inputs to process
        for _ in range(self.num_batches):
            for masks in mask_generator(self.batch_size, shape, self.device, dtype = mask_type, **kwargs):
                if len(masks) == 2: # mask_generator returns a tuple ????
                    x_mask, masks = masks
                    # Shapes are ???
                else:
                    # apply mask
                    x_mask = input_fft*masks
                    # cast back to original domain
                    if self.domain == 'fft':
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
            
        for data, target in dataloader: # len(dataloader) = 1 ???
            batch_scores = [] # List to store the saliency maps for the current batch ??? (I don't know)
            print("Computing batch", i+1, "/", len(dataloader))
            i+=1
            # j=0
            for sample, y in zip(data, target):
                m_generator = mask_generator
                # sample has shape (1, 1, 8000) for AudioNet
                with torch.no_grad(): 
                    importance = self.forward(sample.float().squeeze(0), 
                                              mask_generator = m_generator, 
                                              num_cells = num_cells, 
                                              probablity_of_drop = probability_of_drop)
                
                # Selects the importance values for the given class y
                importance = importance.cpu().squeeze()[...,y]/probability_of_drop
                # min max normalize
                # breakpoint() # Step 1
                importance = (importance - importance.min()) / (importance.max() - importance.min())
                # breakpoint() # Step 2
                # pickle_output = {
                #     'time_signal': sample.float().squeeze(0).squeeze(0).cpu(),
                #     'freq_signal': tfft(sample.float().squeeze(0).squeeze(0).cpu().unsqueeze(0).to(self.device)).squeeze(0).cpu(),
                #     'importance': importance.squeeze(0).cpu(),
                #     'target': y.cpu(),
                # }
                # # Save the output to a pickle file
                # with open(f'outputs/signal_digit_{j}.pkl', 'wb') as f:
                #     pickle.dump(pickle_output, f)
                # # importance of one input sample has shape (4001)
                # batch_scores.append(importance)
                # j+=1
            # batch_scores has shape (batch_size, 4001)
            freqrise_scores.append(torch.stack(batch_scores))
        # freqrise_scores has shape (len(dataloader), batch_size, 4001)
        return freqrise_scores