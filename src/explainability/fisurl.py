from torch.fft import rfft
from torch.fft import irfft
import torch
import torch.nn as nn
from src.explainability.masking_filterbank import filter_mask_generator
from src.explainability.masking_filterbank import FilterbankMaskPolicy
from src.utils.filterbank import FilterBank
import numpy as np
from tqdm import tqdm
import os
import pickle
import time

class FiSURL(nn.Module): # FiSURL: Filterbank Sampling Using Reinforcement Learning (WIP Title)
    def __init__(self, 
                encoder: nn.Module, # Black-box model to explain
                num_taps: int = 501,
                num_banks: int = 10,
                fs: int = 8000,
                bandwidth = None,
                batch_size: int = 10, 
                num_batches: int = 300,
                keep_ratio: float = 0.05, # Between 0.05 and 0.15 in FLEXtime
                device: str = 'cpu',
                use_softmax = False,
                use_rl: bool = True,
                # rl_params: dict = {'lr': 1e-4, 'alpha': 1.00, 'beta': 0.01, 'decay': 0.9, 'reward_fn': 'pred'},
                rl_params: dict = {'lr': 0.05, 'alpha': 1.00, 'beta': 0.05, 'decay': 0.9}
                ):

        super().__init__()

        # Initialize filter bank parameters
        self.num_taps = num_taps
        self.num_banks = num_banks
        self.fs = fs
        self.bandwidth = bandwidth
        self.filter_bank = FilterBank(self.num_banks, self.fs, self.num_taps, self.bandwidth)

        # Initialize mask generator parameters
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.keep_ratio = keep_ratio
        self.device = device

        # Initialize encoder
        self.encoder = encoder.eval().to(self.device)
        self.use_softmax = use_softmax

        # Initialize RL parameters
        self.use_rl = use_rl
        self.rl_params = rl_params
        self.lr = rl_params['lr']
        self.alpha = rl_params['alpha']
        self.beta = rl_params['beta']
        self.decay = rl_params['decay']

    def forward(self, input_data, target_class, mask_generator, **kwargs):
        saliencies = []
        input_data = input_data.unsqueeze(0).to(self.device)
        shape = input_data.shape
        input_fft = rfft(input_data)

        if self.use_rl:
            start_time = time.time()
            m_policy = FilterbankMaskPolicy(self.batch_size, shape, self.num_banks, self.device)
            optimizer = torch.optim.Adam(m_policy.parameters(), lr=self.lr)
            baseline = 0.0

            with torch.no_grad(): 
                # Get predictions of the model with the original input
                pred_original = self.encoder(input_data.unsqueeze(0).float().to(self.device), only_feats = False).detach().squeeze()
                if self.use_softmax:
                    pred_original = torch.softmax(pred_original, dim=-1)

            random_indices = torch.randint(0, self.num_banks, (4,)) # Randomly select 4 parameters to save
            params_saved = []
            losses = []
            reward_list = []

        # Generate masks
        # for _ in tqdm(range(self.num_batches)): # Uncomment for progress bar
        for _ in range(self.num_batches):
            if self.use_rl:
                # Use RL mask generator
                masks, log_probs = m_policy()
                masks = masks.to(self.device)
                x_masked = self.filter_bank.forward(input_data, mask=masks) # Apply the mask to the input data - masked data is of shape (batch_size, 1, 1, fs)

                with torch.no_grad():
                    predictions = self.encoder(x_masked.float().to(self.device), only_feats = False).detach()
                    if self.use_softmax:
                        predictions = torch.softmax(predictions, dim=1)

                rewards = []
                sals = torch.matmul(predictions.transpose(0,1).float(), masks.view(self.batch_size, -1).abs().float()).transpose(0,1).unsqueeze(0).cpu() # Compute saliency
                saliencies.append(sals)
                # Faithfulness: how close is the masked prediction to original
                pred_diffs = torch.abs(pred_original[target_class] - predictions[:,target_class])
                faithfulness_rewards = -pred_diffs  # smaller difference = better

                # Sparsity: fewer active frequencies = better
                mask_sizes = masks.abs().mean(dim=-1).squeeze()
                # sparsity_penalties = mask_sizes  # encourage fewer active cells
                sparsity_penalties = torch.max(mask_sizes - torch.tensor(self.keep_ratio).to(self.device), torch.zeros(50).to(self.device))

                rewards = (self.alpha * faithfulness_rewards - self.beta * sparsity_penalties).to(self.device)

                mean_reward = rewards.mean()
                baseline = self.decay * baseline + (1 - self.decay) * mean_reward # Update the baseline
                loss = -((rewards - baseline) * log_probs).mean() # Reinforce loss, negative because we want to maximize the expected reward

                optimizer.zero_grad() # Zero the gradients  
                loss.backward() # Backpropagation
                optimizer.step() # Update the policy network

                params_saved.append(m_policy.logits[random_indices].tolist())
                losses.append(loss.item())
                for reward in rewards:
                    reward_list.append(reward.item())

            else:
                # Generate a batch of masks
                for masks in mask_generator(self.batch_size, shape, **kwargs):
                    masks = masks.to(self.device)
                    x_masked = self.filter_bank.forward(input_data, mask=masks) # Apply the mask to the input data - masked data is of shape (batch_size, 1, 1, fs)

                    with torch.no_grad():
                        predictions = self.encoder(x_masked.float().to(self.device), only_feats = False).detach() # Pass the masked input through the encoder
                        if self.use_softmax:
                            predictions = torch.softmax(predictions, dim=1)

                    # sal = torch.abs(predictions - self.encoder(input_data.float().to(self.device), only_feats = False).detach())
                    sal = torch.matmul(predictions.transpose(0,1).float(), masks.view(self.batch_size, -1).abs().float()).transpose(0,1).unsqueeze(0).cpu() # Compute saliency
                    saliencies.append(sal)

        importance = torch.cat(saliencies, dim=0).sum(dim=0)/(self.num_batches*self.batch_size)

        if self.use_rl:
            # Selects the importance values for the given class y
            importance_ = importance.cpu().squeeze()[...,target_class] # /kwargs.get('probability_of_drop')
            importance_ = (importance_ - importance_.min()) / (importance_.max() - importance_.min()) # min max normalize
            run_time = time.time() - start_time
            output_dict = {
                'signal': input_data.squeeze(),
                'signal_fft' : input_fft.squeeze(),
                'signal_fft_re' : torch.abs(input_fft).squeeze(),
                'target_class': target_class,
                'prediction_target_class': pred_original[target_class],
                'prediction_argmax': torch.argmax(pred_original),
                'importance': importance_,
                'logged_params': params_saved,
                'loss' : losses,
                'rewards' : reward_list,
                'run_time' : run_time
            }
            os.makedirs(f'notebooks/samples/fisurl_sm_{self.use_softmax}_batchsize_{self.batch_size}_numbatches_{self.num_batches}_lr_{self.lr}_alpha_{self.alpha}_beta_{self.beta}_decay_{self.decay}_numbanks_{self.num_banks}', exist_ok=True)
            with open(f'notebooks/samples/fisurl_sm_{self.use_softmax}_batchsize_{self.batch_size}_numbatches_{self.num_batches}_lr_{self.lr}_alpha_{self.alpha}_beta_{self.beta}_decay_{self.decay}_numbanks_{self.num_banks}/sample_idx_{kwargs.get('idx')}.pkl', mode='wb') as f:
                pickle.dump(output_dict, f)
        
        return importance

    def forward_dataloader(self, dataloader, num_banks, probability_of_drop):
        fisurl_scores = []
        i = 0
        # sample_count = 0

        for data, target in dataloader:
            batch_scores = []
            print("Computing batch", i+1, "/", len(dataloader))
            j = 0

            for sample, y in tqdm(zip(data, target), desc="Computing samples", total=data.shape[0]):
            # for sample, y in zip(data, target):
                # print("Computing sample", sample_count+1, "/", data.shape[0])
                # sample_count += 1
                m_generator = filter_mask_generator
                importance = self.forward(sample.float().squeeze(0),
                                          target_class = y, 
                                          mask_generator = m_generator, 
                                          num_banks = num_banks,
                                          probability_of_drop = probability_of_drop,
                                          idx=j*i+j
                                          )

                # Selects the importance values for the given class y
                importance = importance.cpu().squeeze()[...,y]/probability_of_drop
                importance = (importance - importance.min()) / (importance.max() - importance.min()) # min max normalize
                batch_scores.append(importance)
                j += 1

            fisurl_scores.append(torch.stack(batch_scores))
            i += 1

        return fisurl_scores