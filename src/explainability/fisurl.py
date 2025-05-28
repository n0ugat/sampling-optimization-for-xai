from torch.fft import rfft
import torch
import torch.nn as nn

from src.explainability.masking_filterbank import FilterbankMaskPolicy
from src.utils import apply_fir_filterbank_mask, create_fir_filterbank
import numpy as np
import os
import pickle
import time

class FiSURL(nn.Module): # FiSURL: Filterbank Sampling Using Reinforcement Learning
    def __init__(self, 
                encoder: nn.Module, # Black-box model to explain
                num_taps: int = 501,
                num_banks: int = 10,
                batch_size: int = 10, 
                num_batches: int = 300,
                keep_ratio: float = 0.05,
                device: str = 'cpu',
                use_softmax = False,
                lr: float = 1e-4,
                alpha: float = 1.00,
                beta: float = 0.01,
                decay: float = 0.9,
                save_signals_path = None
                ):

        super().__init__()

        # Initialize filter bank parameters
        self.num_taps = num_taps
        self.num_banks = num_banks

        # Initialize mask generator parameters
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.keep_ratio = keep_ratio
        self.device = device

        # Initialize encoder
        self.encoder = encoder.eval().to(self.device)
        self.use_softmax = use_softmax

        # Initialize RL parameters
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.decay = decay

        self.num_masks = num_batches * batch_size
        self.save_signals_path = save_signals_path

    def forward(self, input_data, target_class, idx):
        """
        Compute the saliency map of the input data using FiSURL.
        Args:
            input_data: torch.Tensor
                The input data for which to compute the saliency map.
            target_class: torch.Tensor
                The target class for which to compute the saliency map.
            idx: int
                The index of the sample in the batch.
        Returns:
            torch.Tensor: The saliency map of the input data.
        """
        p = []
        input_fft = rfft(input_data)
        bandwidth = (input_data.shape[-1] / 2) /  self.num_banks

        # if self.use_rl:
        m_policy = FilterbankMaskPolicy(self.batch_size, input_data.shape, self.num_banks, self.device)
        optimizer = torch.optim.Adam(m_policy.parameters(), lr=self.lr)
        baseline = 0.0

        with torch.no_grad(): 
            # Get predictions of the model with the original input
            pred_original = self.encoder(input_data.unsqueeze(0).float().to(self.device), only_feats = False).detach().squeeze()
            if self.use_softmax:
                pred_original = torch.softmax(pred_original, dim=-1)

        if self.save_signals_path:
            random_indices = torch.randint(0, self.num_banks, (4,)) # Randomly select 4 parameters to save
            params_saved = []
            losses = []
            reward_list = []
            start_time = time.time()

        for j in range(self.num_batches):
            masks, log_probs = m_policy()
            masks = masks.to(self.device)
            x_masked = apply_fir_filterbank_mask(input_data[0,0], self.filterbank, masks.view(self.batch_size, self.num_banks), self.num_taps).reshape(self.batch_size,1,1,-1)
            with torch.no_grad():
                predictions = self.encoder(x_masked.float().to(self.device), only_feats = False).detach()
                if self.use_softmax:
                    predictions = torch.softmax(predictions, dim=-1)
                 
            masks_up = torch.zeros((self.batch_size, 1, 1, int(input_data.shape[-1] / 2) + 1)).to(self.device)
            index = 0
            for i in range(1, self.num_banks+1):
                next_index = int(np.ceil(i * bandwidth))
                masks_up[:, :, :, index:next_index] = masks[:, :, :, i-1].unsqueeze(-1)
                index = next_index
            sals = torch.matmul(predictions.transpose(0,1).float(), masks_up.view(self.batch_size, -1).abs().float()).transpose(0,1).unsqueeze(0) # Compute saliency
            p.append(sals)

            # Faithfulness: how close is the masked prediction to original
            faithfulness_rewards = -torch.abs(pred_original[target_class] - predictions[:,target_class])
            # smaller difference = better

            # Sparsity: fewer active frequencies = better
            mask_sizes = masks.abs().mean(dim=-1).squeeze()
            # sparsity_penalties = mask_sizes  # encourage fewer active cells
            sparsity_penalties = torch.max(mask_sizes - torch.tensor(self.keep_ratio).to(self.device), torch.zeros(self.batch_size).to(self.device))

            rewards = (self.alpha * faithfulness_rewards - self.beta * sparsity_penalties)

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
        importance = importance.cpu().squeeze()[...,target_class]
        importance = (importance - importance.min()) / (importance.max() - importance.min()) # min max normalize
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

    def forward_dataloader(self, dataloader):
        fisurl_scores = []
        i = 0

        for data, target in dataloader:
            self.filterbank = create_fir_filterbank(self.num_banks, data.shape[-1], self.num_taps, device=self.device)
            batch_scores = []
            print("Computing batch", i+1, "/", len(dataloader))
            for j, (sample, y) in enumerate(zip(data, target)):
                print("Computing sample", j+1, "/", len(data))
                sample = sample.float().to(self.device)
                y = y.to(self.device)

                importance = self.forward(sample,
                                          target_class = y, 
                                          idx=j
                                          )

                batch_scores.append(importance.cpu())
            fisurl_scores.append(torch.stack(batch_scores))
            i += 1

        return fisurl_scores