# Used to create fft-plot in paper
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

import torch
from torch.fft import rfft
from torch.fft import irfft

# Add repo directory to system path
repo_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(repo_dir)

from src.data import synthetic_dataset_generator

def plot_fft_fakemake_example(signal, length=1):
    # Define signal parameters
    N = len(signal)   # Number of samples (1-second signal)
    fs = N / length  # Sampling frequency in Hz
    t = torch.arange(N*length) / fs  # Time vector
    
    # Compute the real FFT
    fft_result = rfft(torch.Tensor(signal))
    # Compute frequency axis
    frequencies = torch.fft.rfftfreq(N*length, d=1/fs)
    fft_result = torch.abs(fft_result)  # Get magnitude of FFT result
    # Apply a mask on random frequencies
    masked_fft_result = fft_result.clone()
    for i in range(0, len(masked_fft_result), 10):
        if np.random.randint(0, 2) == 1:
            masked_fft_result[i:i+10] = 0
    # Compute the inverse FFT
    recovered_signal = irfft(masked_fft_result, n=N*length)
    # Plotting
    fig, ax = plt.subplots(4, 1, figsize=(8, 6))
    ax[0].plot(t.numpy(), signal, label='Original Signal')
    ax[0].set_title('(1) Original Signal')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frequencies.numpy(), fft_result.numpy(), label='FFT of Original Signal')
    ax[1].set_title('(2) RFFT of Original Signal')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Magnitude')
    ax[2].plot(frequencies.numpy(), masked_fft_result.numpy(), label='Masked FFT', color='orange')
    ax[2].set_title('(3) RFFT of Masked Signal')
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_ylabel('Magnitude')
    ax[3].plot(t.numpy(), recovered_signal.numpy(), label='Recovered Signal', color='orange')
    ax[3].set_title('(4) Recovered Masked Signal using IRFFT')
    ax[3].set_xlabel('Time [s]')
    ax[3].set_ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(f"outputs/figures/fft_process_example.png")


if __name__ == "__main__":
    signals, _ = synthetic_dataset_generator(n_samples=1, length=400, noiselevel=1, add_random_peaks=True, const_class=7, seed=42)
    signal = signals[0]
    
    plot_fft_fakemake_example(signal, length=1) # Length of the signal in seconds