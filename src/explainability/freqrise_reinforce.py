#@title RELAX
from torch.fft import rfft as tfft
from torch.fft import irfft as tifft
import torch
import torch.nn as nn
from src.explainability.masking_reinforce import MaskPolicy
import pickle
import os
import time

class FreqRISE_Reinforce(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 batch_size: int = 10,
                 num_batches: int = 300,
                 device: str = 'cpu',
                 domain = 'fft',
                 use_softmax = False,
                 lr=1e-4,
                 alpha=1.00,
                 beta=0.01,
                 decay=0.9,
                 reward_fn = "pred"
                 ):

        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.domain = domain
        self.lr=lr
        self.alpha=alpha
        self.beta=beta
        self.decay=decay
        self.reward_fn = reward_fn

        self.num_batches = num_batches
        self.num_masks = num_batches * batch_size
        self.use_softmax = use_softmax
        self.encoder = encoder.eval().to(self.device) # function that evaluates the pretrained model on a given input
    
    def reward_fn_saliency(self, saliency, mask):
        """
        Reward = saliency score from masked input - penalty for mask size
        """
        sal_score = saliency.mean() # Sum saliency: higher is better
        mask_size = mask.abs().mean() # Penalize large mask
        reward = self.alpha * sal_score - self.beta * mask_size
        return reward
        
    def reward_fn_pred(self, pred_original, pred_masked, mask, target_class):
        """
        Reward = classification faithfulness - penalty for mask size
        """
        # Faithfulness: how close is the masked prediction to original
        pred_diff = torch.abs(pred_original[target_class] - pred_masked[target_class])
        faithfulness_reward = -pred_diff  # smaller difference = better

        # Sparsity: fewer active frequencies = better
        mask_size = mask.abs().mean()
        sparsity_penalty = mask_size  # encourage fewer active cells

        reward = self.alpha * faithfulness_reward - self.beta * sparsity_penalty
        return reward


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
        # cast data to domain of interest
        if self.domain == 'fft':
            # Fast Fourier Transform (To frequency domain)
            input_fft = tfft(input_data) 
        else:
            input_fft = input_data
            
        shape = input_fft.shape
        mask_type = torch.complex64 if self.domain == 'fft' else torch.float32
        start_time = time.time()
        
        m_policy = MaskPolicy(batch_size=self.batch_size, shape=shape, num_cells = num_cells, device = self.device, dtype=mask_type).to(self.device)
        optimizer = torch.optim.Adam(m_policy.parameters(), lr=self.lr)
        baseline = 0.0
        
        with torch.no_grad(): 
            # get predictions of the model with the original input
            pred_original = self.encoder(input_data.unsqueeze(0).float().to(self.device), only_feats = False).detach().squeeze()
            if self.use_softmax:
                pred_original = torch.softmax(pred_original, dim=-1)
        
        random_indices = torch.randint(0, num_cells, (4,))
        params_saved = []
        losses = []
        reward_list = []
        
        for i in range(self.num_batches):
            masks, log_probs = m_policy() # Sample a batch of masks from the Bernoulli distribution
            x_masks = input_fft*masks
            if self.domain == 'fft':
                x_masks = tifft(x_masks, dim=-1)

            with torch.no_grad():
                # Get the model prediction for the masked input
                predictions = self.encoder(x_masks.float(), only_feats = False).detach()
                if self.use_softmax:
                    predictions = torch.softmax(predictions, dim=-1)

            rewards = []

            for mask, pred_masked in zip(masks, predictions):
                sal = torch.matmul(pred_masked.unsqueeze(0).transpose(0,1).float(), mask.abs().float()).transpose(0,1).unsqueeze(0)
                p.append(sal)
                if self.reward_fn == "pred":
                    reward = self.reward_fn_pred(pred_original, pred_masked, mask, target_class) # Compute the reward using the saliency and the masks
                elif self.reward_fn == "saliency":
                    reward = self.reward_fn_saliency(sal, mask) # Compute the reward using the saliency and the masks
                rewards.append(reward)

            rewards = torch.stack(rewards).to(self.device)
            mean_reward = rewards.mean()
            baseline = self.decay * baseline + (1 - self.decay) * mean_reward # Update the baseline
            loss = -((rewards - baseline) * log_probs).mean() # Reinforce loss, negative because we want to maximize the expected reward

            optimizer.zero_grad() # Zero the gradients  
            loss.backward() # Backpropagation
            optimizer.step() # Update the policy network

            params_saved.append(m_policy.logits[random_indices].tolist())
            losses.append(loss.item())
            reward_list.append(reward.item())
            # if i % 10 == 0:
            #     print(f"Iteration {i}, Loss: {loss.item()}, Reward: {reward.item()}, Baseline: {baseline}")
        importance = torch.cat(p, dim=0).sum(dim=0)/(self.num_batches*self.batch_size)
        # Selects the importance values for the given class y
        importance = importance.cpu().squeeze()[...,target_class]#/probability_of_drop
        # min max normalize
        importance = (importance - importance.min()) / (importance.max() - importance.min())
        run_time = time.time() - start_time
        output_dict = {
            'signal': input_data.squeeze(),
            'signal_fft' : input_fft.squeeze(),
            'signal_fft_re' : torch.abs(input_fft).squeeze(),
            'target_class': target_class,
            'prediction' : torch.argmax(pred_original),
            'importance': importance,
            'logged_params': params_saved,
            'loss' : losses,
            'rewards' : reward_list,
            'run_time' : run_time,
        }
        os.makedirs(f'notebooks/samples/freqrise_sm_{self.use_softmax}_batchsize_{self.batch_size}_numbatches_{self.num_batches}_R_{self.reward_fn}_lr_{self.lr}_alpha_{self.alpha}_beta_{self.beta}_decay_{self.decay}_numcells_{num_cells}', exist_ok=True)
        with open(f'notebooks/samples/freqrise_sm_{self.use_softmax}_batchsize_{self.batch_size}_numbatches_{self.num_batches}_R_{self.reward_fn}_lr_{self.lr}_alpha_{self.alpha}_beta_{self.beta}_decay_{self.decay}_numcells_{num_cells}/sample_idx_{idx}.pkl', mode='wb') as f:
            pickle.dump(output_dict, f)
        return importance # Importance here is the saliency map
    
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
        freqrise_scores = [] # List to store the saliency maps ??? (I don't know)
        i = 0
            
        for data, target in dataloader: # len(dataloader) = 1 ???
            batch_scores = [] # List to store the saliency maps for the current batch ??? (I don't know)
            print("Computing batch", i+1, "/", len(dataloader))
            j = 0
            for sample, y in zip(data, target):
                print("Computing sample", j+1, "/", len(data))
                # sample has shape (1, 1, 8000) for AudioNet
                importance = self.forward(sample.float().squeeze(0), 
                                            target_class = y,
                                            num_cells = num_cells,
                                            idx=j*i+j)
                # Selects the importance values for the given class y
                # importance = importance.cpu().squeeze()[...,y]#/probability_of_drop
                # min max normalize
                # importance = (importance - importance.min()) / (importance.max() - importance.min())
                # # importance of one input sample has shape (4001)
                batch_scores.append(importance)
                j+=1
            # batch_scores has shape (batch_size, 4001)
            freqrise_scores.append(torch.stack(batch_scores))
            i+=1
        # freqrise_scores has shape (len(dataloader), batch_size, 4001)
        return freqrise_scores