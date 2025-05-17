#@title RELAX
import pickle
import os
import time

import torch
import torch.nn as nn
from torch.fft import rfft as tfft
from torch.fft import irfft as tifft

from src.explainability.masking_surl import MaskPolicy

class SURL(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 batch_size: int = 10,
                 num_batches: int = 300,
                 device: str = 'cpu',
                 use_softmax = False,
                 lr=1e-4,
                 alpha=1.00,
                 beta=0.01,
                 decay=0.9,
                 dataset = "AudioMNIST",
                 save_signals_path = None
                 ):

        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.lr=lr
        self.alpha=alpha
        self.beta=beta
        self.decay=decay
        self.dataset = dataset
        self.save_signals_path = save_signals_path

        self.num_batches = num_batches
        self.num_masks = num_batches * batch_size
        self.use_softmax = use_softmax
        self.encoder = encoder.eval().to(self.device) # function that evaluates the pretrained model on a given input


    def forward(self, input_data, target_class, num_cells, idx) -> None:
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
        # cast data to frequency domain
        input_fft = tfft(input_data) 

        mask_type = torch.complex64
        
        m_policy = MaskPolicy(batch_size=self.batch_size, shape=input_fft.shape, num_cells = num_cells, device = self.device, dtype=mask_type).to(self.device)
        optimizer = torch.optim.Adam(m_policy.parameters(), lr=self.lr)
        baseline = 0.0
        
        with torch.no_grad(): 
            # get predictions of the model with the original input
            pred_original = self.encoder(input_data.unsqueeze(0).float().to(self.device), only_feats = False).detach().squeeze()
            if self.use_softmax:
                pred_original = torch.softmax(pred_original, dim=-1)
        
        if self.save_signals_path:
            random_indices = torch.randint(0, num_cells, (4,))
            params_saved = []
            losses = []
            reward_list = []
            start_time = time.time()

        for _ in range(self.num_batches):
            masks, log_probs = m_policy() # Sample a batch of masks from the Bernoulli distribution

            x_masks = input_fft*masks
            x_masks = tifft(x_masks, dim=-1)

            with torch.no_grad():
                # Get the model prediction for the masked input
                predictions = self.encoder(x_masks.float(), only_feats = False).detach()
                if self.use_softmax:
                    predictions = torch.softmax(predictions, dim=-1)

            sals = torch.matmul(predictions.unsqueeze(2).float(), masks.abs().float()).transpose(1,2)
            p.append(sals)

            # Faithfulness: how close is the masked prediction to original
            faithfulness_rewards = -torch.abs(pred_original[target_class] - predictions[:,target_class])
            # smaller difference = better

            # Sparsity: fewer active frequencies = better
            sparsity_penalties = masks.abs().mean(dim=-1).squeeze()
            # encourage fewer active cells

            rewards = (self.alpha * faithfulness_rewards - self.beta * sparsity_penalties)# .to(self.device)

            mean_reward = rewards.mean()
            baseline = self.decay * baseline + (1 - self.decay) * mean_reward # Update the baseline
            loss = -((rewards - baseline) * log_probs).mean() # Reinforce loss, negative because we want to maximize the expected reward

            optimizer.zero_grad() # Zero the gradients  
            loss.backward() # Backpropagation
            optimizer.step() # Update the policy network

            if self.save_signals_path:
                params_saved.append(m_policy.logits[random_indices].cpu().tolist())
                losses.append(loss.cpu().item())
                reward_list.append(mean_reward.cpu().item())

        importance = torch.cat(p, dim=0).sum(dim=0)/(self.num_batches*self.batch_size)
        # Selects the importance values for the given class y
        importance = importance.squeeze()[...,target_class]#/probability_of_drop
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
                'logged_params': params_saved,
                'loss' : losses,
                'rewards' : reward_list,
                'run_time' : run_time,
            }
            sample_path = os.path.join(self.save_signals_path, f'sample_{idx}.pkl')
            with open(sample_path, mode='wb') as f:
                pickle.dump(output_dict, f)
        return importance 
    
    def forward_dataloader(self, dataloader, num_cells):
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
        freqrise_scores = [] 
        i = 0
            
        for data, target in dataloader: 
            batch_scores = [] 
            print("Computing batch", i+1, "/", len(dataloader))
            for j, (sample, y) in enumerate(zip(data, target)):
                print("Computing sample", j+1, "/", len(data))
                sample = sample.to(self.device)
                y = y.to(self.device)
                
                importance = self.forward(sample.float().squeeze(0), 
                                            target_class = y,
                                            num_cells = num_cells,
                                            idx=j)
                batch_scores.append(importance.cpu())
            freqrise_scores.append(torch.stack(batch_scores))
            i+=1
        return freqrise_scores