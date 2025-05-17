import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.generators import synthetic_dataset_generator
from src.data.dataloader import AudioNetDataset

def load_data(args):
    if args.dataset == 'synthetic':
        test_data, test_labels = synthetic_dataset_generator(args.n_samples, length=args.synth_sig_len, noiselevel=args.noise_level, add_random_peaks=not args.no_random_peaks, seed=None)
        test_data, test_labels = torch.tensor(test_data.reshape(args.n_samples, 1,1,-1)).float(), torch.tensor(test_labels).squeeze().long()
        test_loader = DataLoader(TensorDataset(test_data, test_labels.unsqueeze(0) if test_labels.dim() == 0 else test_labels), batch_size=64)
    elif args.dataset == 'AudioMNIST':
        test_dset = AudioNetDataset(args.data_path, True, 'test', splits = [0], labeltype = args.labeltype, subsample = args.n_samples, seed = 0, add_noise=False)  
        test_loader = DataLoader(test_dset, batch_size=10, shuffle=False)
    
    return test_loader
