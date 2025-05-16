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
    ax[0].set_title(f"Time-Domain")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude")
    
    ax[1].plot(signal_fft_re)
    ax[1].set_title(f"Frequency-Domain")
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Magnitude")
    
    fig.suptitle(f"Gender {gender}, Digit {digit}", fontsize=16)
    if title:
        fig.suptitle(title, fontsize=16)
        
    plt.tight_layout()
    plt.savefig(f"outputs/figures/signal_plot_{speaker_idx}_{digit}_{sample_idx}.png")
    

def plot_synthetic_signal():
    signals, labels = synthetic_dataset_generator(n_samples=1, length=400, noiselevel=0.5, seed=43, add_random_peaks=True, const_class=6)
    signal, label = signals[0], int(labels[0][0])
    signal_length = signal.shape[0]
    
    frequency_classes = [50, 100, 150]
    class_ = label
    present_important_frequencies = []
    
    for freq_idx in range(len(frequency_classes)):
        if class_ & (1 << freq_idx):
            present_important_frequencies.append(frequency_classes[freq_idx])
            
    data = np.zeros((1, signal_length))
    for freq in present_important_frequencies:
        data += np.sin(2 * np.pi * freq / signal_length * np.arange(signal_length) + np.random.uniform(0, 2 * np.pi))
    
    data_fft = tfft(torch.tensor(data))
    data_fft_re = torch.abs(data_fft)
    importance_gt = data_fft_re.squeeze().numpy()
    
    signal_fft = tfft(torch.tensor(signal))
    signal_fft_re = torch.abs(signal_fft)
    signal_fft_ = signal_fft_re.squeeze().numpy()
    
    fig, ax = plt.subplots(1,1, figsize=(6, 3))
    fig.suptitle(f"Synthetic Signal Example. Class: {class_}", fontsize=12)
    ax.plot(np.linspace(0,1,len(signal)), signal)
    ax.set_title(f"Time-Domain", fontsize=11)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/synthetic_signal_plot_timedomain.png")
    plt.close()
    fig, ax = plt.subplots(1,1, figsize=(6, 3))
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Frequency-Domain with ground truth importance", fontsize=11)
    ts_importance(
            ax=ax, 
            importance=importance_gt, 
            timeseries=signal_fft_
    )
    plt.tight_layout()
    plt.savefig(f"outputs/figures/synthetic_signal_plot_freqdomain.png")
    plt.close()


if __name__ == '__main__':
    # Found from data/audioMNIST_meta.txt file
    male_speaker_idx = "01"
    female_speaker_idx = "12"
    male_speaker_digit = "9"
    female_speaker_digit = "0"
    male_speaker_sample_idx = "32"
    female_speaker_sample_idx = "41"
    
    # plot_and_print_signal_from_path(male_speaker_idx, male_speaker_digit, male_speaker_sample_idx, "male")
    # plot_and_print_signal_from_path(female_speaker_idx, female_speaker_digit, female_speaker_sample_idx, "female")
    
    plot_synthetic_signal()