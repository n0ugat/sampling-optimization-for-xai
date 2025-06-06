import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F

from quantus.metrics import Complexity

from src.explainability.metrics import relevance_rank_accuracy
from src.lrp import dft_lrp
from src.lrp import lrp_utils

# Code heavily inspired by https://github.com/theabrusch/FreqRISE

def deletion_curves(model, test_loader, importance, quantiles, device = 'cpu', cutoff = None, filterbank = None, method_name = ''):
    deletion = {
        'mean_prob': [],
        'mean_probs': [],
        'quantiles': quantiles
    }
    for quantile in quantiles:
        mean_true_class_probs = mask_and_predict(model, test_loader, importance, quantile, device = device, cutoff = cutoff, filterbank = filterbank, method_name = method_name)
        deletion['mean_prob'].append(np.mean(mean_true_class_probs))
        deletion['mean_probs'].append(mean_true_class_probs)
    deletion['mean_probs'] = np.array(deletion['mean_probs']).transpose()
    auc = np.trapz(deletion['mean_prob'], deletion['quantiles'])
    deletion['AUC'] = auc
    auc_s = []
    for i in range(len(deletion['mean_probs'])):
        auc_s.append(np.trapz(deletion['mean_probs'][i], deletion['quantiles']))
    deletion['AUC_std'] = np.std(auc_s)
    deletion['n'] = len(deletion['mean_probs'])
    return deletion

def complexity_scores(attributions, cutoff = None, only_pos = False):
    comp_scores = []
    comp = Complexity()
    for i, sample in enumerate(attributions):
        if only_pos:
            sample[sample < 0] = 0
        sample = (sample - np.min(sample))/(np.max(sample) - np.min(sample) + 1e-8) + 1e-8
        if cutoff is not None:
            cut = np.quantile(sample, cutoff)
            sample[sample < cut] = 0
        # flatten sample
        sample = sample.flatten()
        comp_scores.append(comp.evaluate_instance(sample, sample))
    return comp_scores

def localization_scores(attributions, labels, cutoff = None, only_pos = False):
    loc_scores = []
    freq_comps = attributions.shape[-1]
    frequency_classes = [int(freq_comps*0.2), int(freq_comps*0.5), int(freq_comps*0.8)] # Frequencies for the classes
    n_freqs = len(frequency_classes)
    n_classes = 1 << n_freqs
    classes_ = []
    for class_ in range(n_classes):
        classes = []
        for i in range(n_freqs):
            if class_ & (1 << i):
                classes.append(frequency_classes[i])
        classes_.append(tuple(classes))
    for i, sample in enumerate(attributions):
        if labels[i]== 0:
            continue
        if only_pos:
            sample[sample < 0] = 0
        sample = (sample - np.min(sample))/(np.max(sample) - np.min(sample))
        if cutoff is not None:
            cut = np.quantile(sample, cutoff)
            sample[sample < cut] = 0
        # flatten sample
        sample = sample.flatten()
        loc_scores.append(relevance_rank_accuracy(sample, np.array(classes_[labels[i]])))
    return loc_scores


def mask_and_predict(model, test_loader, importance, quantile, device = 'cpu', cutoff = None, filterbank = None, method_name = ''):
    model.eval().to(device)
    with torch.no_grad():
        total = 0
        total_batches = []
        mean_true_class_prob = 0
        mean_true_class_prob_batches = []
        for i, batch in enumerate(test_loader):
            data, true_label = batch
            data = torch.fft.rfft(data, dim=-1).to(device)
            shape = data.shape
            if importance == 'random':
                imp = torch.rand_like(data.abs()).float()
            elif importance == 'amplitude':
                imp = torch.abs(data)
            else:
                if filterbank is not None:
                    data = filterbank.batch_apply(data) # Shape: (test_batch_size, num_banks, 1, 1, fs)
                else:
                    imp = importance[i].reshape(-1, 1, 1, data.shape[-1])
                imp[imp < 0] = 0
            if cutoff is not None:
                if cutoff == 'mean':
                    flattened_imp = imp.reshape(shape[0], -1)
                    cutoff = flattened_imp.mean(-1, keepdim=True)
                    # set all values below mean to 0
                    flattened_imp[flattened_imp < cutoff] = 0
                    imp = flattened_imp.view(shape)
                else:
                    flattened_imp = imp.reshape(shape[0], -1)
                    cut = flattened_imp.quantile(cutoff, dim=-1, keepdim=True)
                    # set all values below mean to 0
                    flattened_imp[flattened_imp < cut] = 0
                    imp = flattened_imp.view(shape)
            # take q percent largest values 
            flattened_imp = imp.reshape(shape[0], -1)
            k = int(quantile * flattened_imp.size(1))
            # Find top 10% (T * D * 10%)
            topk_values, topk_indices = torch.topk(flattened_imp, k=k, dim=1)
            mask = torch.zeros_like(flattened_imp, dtype=torch.bool)
            # Set the positions of the top-k elements to True
            mask.scatter_(1, topk_indices, True)
            mask = mask.view(shape).to(device)
            data = data * (~mask)

            data = torch.fft.irfft(data, dim=-1)
            output = model(data.float()).detach().cpu()
            total_batches.append(true_label.size(0))
            total += total_batches[-1]
            # one hot encode true label
            mean_true_class_prob_batches.append(torch.take_along_dim(F.softmax(output, dim=1), true_label.unsqueeze(1), dim = 1).sum().item())
            mean_true_class_prob += mean_true_class_prob_batches[-1]

    a = np.array(mean_true_class_prob_batches)
    b = np.array(total_batches)
    mean_probs = a / b
    return mean_probs

def compute_gradient_scores(model, testloader, attr_method, device='cpu', save_signals_path=None):
    lrp_scores = []
    for sample, target in testloader:
        sample = sample.to(device)
        model = model.to(device)
        target = target.to(device)
        if save_signals_path:
            with torch.no_grad(): 
                predictions = model(sample.float(), only_feats = False).squeeze()
        relevance_time = lrp_utils.zennit_relevance(sample, model, target=target, attribution_method=attr_method)
        dftlrp = dft_lrp.DFTLRP(sample.shape[-1], 
                                leverage_symmetry=True, 
                                precision=32,
                                device = device,
                                create_stdft=False,
                                create_inverse=False
                                )
        signal_freq, relevance_freq = dftlrp.dft_lrp(relevance_time, sample.float(), real=False, short_time=False)
        if save_signals_path:
            for idx in range(sample.shape[0]):
                output_dict = {
                    'signal': sample[idx].squeeze().cpu(),
                    'signal_fft' : np.abs(signal_freq[idx]).squeeze(),
                    'target_class': target[idx].cpu(),
                    'prediction' : torch.argmax(predictions[idx]).cpu(),
                    'importance': relevance_freq[idx].squeeze()
                }
                sample_path = os.path.join(save_signals_path, f'sample_{idx}.pkl')
                with open(sample_path, mode='wb') as f:
                    pickle.dump(output_dict, f)
        lrp_scores.append(torch.tensor(relevance_freq))
    return lrp_scores