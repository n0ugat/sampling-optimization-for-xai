import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import sys
import os

import torch
from torch.fft import rfft as tfft

# Add repo directory to system path
repo_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(repo_dir)

from src.data.generators import synthetic_dataset_generator
from importance_plots import ts_importance


def plot_and_print_signal_from_path(speaker_idx, digit, sample_idx, gender, title=None):
    signal_path = f"data/preprocessed_data/{speaker_idx}/AudioNet_{digit}_{speaker_idx}_{sample_idx}.hdf5"
    
    with h5.File(signal_path, 'r') as f:
        signal = torch.tensor(f['data']).squeeze()
    signal_fft = tfft(signal)
    
    signal_fft_re = torch.abs(signal_fft)
    signal_fft_re = signal_fft_re.numpy()
    signal = signal.numpy()
    
    # Plot the signal
    fig, ax = plt.subplots(1,2, figsize=(6, 3))
    ax[0].plot(signal)
    ax[0].set_title(f"Time-Domain", fontsize=14)
    ax[0].set_xlabel("Time", fontsize=14)
    ax[0].set_ylabel("Amplitude", fontsize=14)
    
    ax[1].plot(signal_fft_re)
    ax[1].set_title(f"Frequency-Domain", fontsize=14)
    ax[1].set_xlabel("Frequency", fontsize=14)
    ax[1].set_ylabel("Magnitude", fontsize=14)
    
    fig.suptitle(f"Gender {gender}, Digit {digit}", fontsize=18)
    if title:
        fig.suptitle(title, fontsize=16)
        
    plt.tight_layout()
    os.makedirs(f"outputs/figures/signals", exist_ok=True)
    plt.savefig(f"outputs/figures/signals/AudioMNIST_{speaker_idx}_{digit}_{sample_idx}.png")
    

def plot_synthetic_signal(n_samples, length, noiselevel, seed, add_random_peaks, const_class):
    signals, labels = synthetic_dataset_generator(n_samples=n_samples, length=length, noiselevel=noiselevel, seed=seed, add_random_peaks=add_random_peaks, const_class=const_class)
    signal, label = signals[0], int(labels[0][0])
    
    freq_comps = length // 2
    frequency_classes = [int(freq_comps*0.2), int(freq_comps*0.5), int(freq_comps*0.8)] # Frequencies for the classes
    class_ = label
    present_important_frequencies = []
    
    for freq_idx in range(len(frequency_classes)):
        if class_ & (1 << freq_idx):
            present_important_frequencies.append(frequency_classes[freq_idx])
            
    data_gt = np.zeros((1, length))
    for freq in present_important_frequencies:
        data_gt += np.sin(2 * np.pi * freq / length * np.arange(length) + np.random.uniform(0, 2 * np.pi))
    
    data_gt_fft_im = tfft(torch.tensor(data_gt))
    data_gt_fft = torch.abs(data_gt_fft_im)
    importance_gt = data_gt_fft.squeeze().numpy()
    
    signal_fft_im = tfft(torch.tensor(signal))
    signal_fft = torch.abs(signal_fft_im)
    signal_fft_ = signal_fft.squeeze().numpy()
    
    os.makedirs(f"outputs/figures/signals", exist_ok=True)
    
    fig, axs = plt.subplots(1,2, figsize=(12, 3))
    fig.suptitle(f"Synthetic Signal Example", fontsize=24)
    axs[0].plot(np.linspace(0,1,len(signal)), signal)
    axs[0].set_title(f"Time-Domain", fontsize=18)
    axs[0].set_xlabel("Time", fontsize=18)
    axs[0].set_ylabel("Amplitude", fontsize=18)

    axs[1].set_xlabel("Frequency", fontsize=18)
    axs[1].set_ylabel("Magnitude", fontsize=18)
    axs[1].set_title(f"Frequency-Domain", fontsize=18)
    ts_importance(
            ax=axs[1], 
            importance=importance_gt, 
            timeseries=signal_fft_,
            colorbar=False
    )
    plt.tight_layout()
    plt.savefig(f"outputs/figures/signals/synthetic_timeandfreqdomain_signal.png")
    plt.close()


if __name__ == '__main__':
    # Found from data/audioMNIST_meta.txt file
    male_speaker_idx = "01"
    female_speaker_idx = "12"
    male_speaker_digit = "9"
    female_speaker_digit = "0"
    male_speaker_sample_idx = "32"
    female_speaker_sample_idx = "41"
    
    plot_and_print_signal_from_path(male_speaker_idx, male_speaker_digit, male_speaker_sample_idx, "male")
    plot_and_print_signal_from_path(female_speaker_idx, female_speaker_digit, female_speaker_sample_idx, "female")
    
    plot_synthetic_signal(n_samples=1, length=100, noiselevel=0.0, seed=43, add_random_peaks=True, const_class=6)