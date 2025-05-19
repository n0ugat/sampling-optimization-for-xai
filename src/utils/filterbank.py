import numpy as np
from scipy import signal
import torch

class FilterBank:
    def __init__(self, num_banks, fs, num_taps=None, bandwidth=None):
        self.num_banks = num_banks # Number of filter banks
        self.fs = fs # Sampling frequency
        self.num_taps = num_taps if num_taps else 8 * num_banks # Number of taps in each filter (Higher order = sharper cutoff, but more computation)
        if self.num_taps % 2 == 0: # Ensure odd number of taps
            self.num_taps += 1
        self.bandwidth = bandwidth if bandwidth else (fs / 2) /  num_banks # Bandwidth of each filter bank
        self.banks = []

        self.create_banks()

    def create_banks(self):
        h = signal.firwin(self.num_taps, self.bandwidth, fs=self.fs, pass_zero='lowpass')
        self.banks.append(h)
        band_start = self.bandwidth

        for _ in range(1, self.num_banks - 1):
            h = signal.firwin(self.num_taps, [band_start, band_start + self.bandwidth], fs=self.fs, pass_zero='bandpass')
            self.banks.append(h)
            band_start += self.bandwidth

        h = signal.firwin(self.num_taps, band_start, fs=self.fs, pass_zero='highpass')
        self.banks.append(h)
        self.banks = np.array(self.banks)
        self.banks = torch.from_numpy(self.banks).float()
    
    def apply(self, x, return_bank_borders=False):
        # Apply each filter bank to the input signal
        y = []
        bank_borders = []
        device = x.device
        for i in range(self.num_banks):
            y.append(signal.lfilter(self.banks[i], 1, x.cpu(), axis=-1))
            y[i] = torch.from_numpy(y[i])
            if return_bank_borders:
                bank_borders.append(self.bandwidth * (i + 1))
        y = torch.stack(y, dim=0).to(device) # Shape: (num_banks, 1, 1, fs)
        if return_bank_borders:
            return y, bank_borders
        return y
    
    def batch_apply(self, x):
        ys = []
        for i in range(x.shape[0]):
            y = self.apply(x[i])
            ys.append(y)
        ys = torch.stack(ys, dim=0)
        return ys # Shape: (test_batch_size, num_banks, 1, 1, fs)

    def forward(self, x, mask=None, return_bank_borders=False):
        if return_bank_borders:
            y, bank_borders = self.apply(x, return_bank_borders=True)
        else:
            y = self.apply(x)
        # Mask shape is (batch_size, 1, 1, num_banks)
        # y shape is (num_banks, 1, 1, self.fs)
        if mask is not None:
            y = y.permute(1, 2, 0, 3) # [num_banks, 1, 1, fs] -> [batch_size, 1, num_banks, fs]
            mask = mask.permute(0, 1, 3, 2) # [batch_size, 1, 1, num_banks] -> [batch_size, 1, num_banks, 1]
            # mask = mask[:, None, None, None]
            # mask = mask.unsqueeze(1)
            masked = mask * y # [50, 1, 10, 8000] - (batch_size, 1, num_banks, fs)
            masked = torch.sum(masked, dim=2).unsqueeze(2) # [50, 1, 1, 8000] # Sum the filtered signals along the frequency bands
            # y = y * mask

        # if return_bank_borders:
        #     return torch.sum(y, dim=0), bank_borders
        # return torch.sum(y, dim=0) # Sum the filtered signals along the frequency bands
        if mask is None:
            y = y.permute(1, 2, 0, 3)
            masked = torch.sum(y, dim=2).unsqueeze(2)
        if return_bank_borders:
            return masked, bank_borders
        return masked # Shape: (batch_size, 1, 1, fs)
    

    #####
    def batch_forward(self, x, mask=None):
        ys = []

        for i in range(x.shape[0]):
            y = self.forward(x[i], mask=mask, return_bank_borders=False)
            ys.append(y)

        ys = torch.stack(ys, dim=0).squeeze(1) # Shape: (batch_size, 1, 1, fs)
        return ys
