import torch
from scipy.signal import firwin
import numpy as np

def create_fir_filterbank(n_filters, sample_rate, filter_order=101):
    """
    Creates a bank of FIR bandpass filters using scipy's firwin.
    Returns a list of filters (each of length filter_order).
    """
    nyquist = sample_rate / 2
    band_edges = np.linspace(0, nyquist, n_filters + 1)
    filters = []

    for i in range(n_filters):
        low = band_edges[i] / nyquist
        high = band_edges[i + 1] / nyquist

        if low == 0:
            coeffs = firwin(filter_order, high, pass_zero='lowpass')
        elif high == 1:
            coeffs = firwin(filter_order, low, pass_zero='highpass')
        else:
            coeffs = firwin(filter_order, [low, high], pass_zero=False)

        filters.append(torch.tensor(coeffs, dtype=torch.float32))

    return filters  # List of tensors


def apply_fir_filterbank_mask(signal_time, filterbank, masks, numtaps=101):
    """
    Applies the FIR filterbank with binary mask to a time-domain signal.
    Each filter's output is included or excluded based on the mask.
    """

    filtered_bands = []
    for coeffs in filterbank:
        # Apply FIR filter via 1D convolution (equivalent to lfilter)
        filtered = torch.nn.functional.conv1d(
            signal_time.view(1, 1, -1),  # (N, C, L)
            coeffs.view(1, 1, -1),      # (out_channels, in_channels, kernel_size)
            padding=numtaps // 2        # keep output same size
        ).squeeze()
        filtered_bands.append(filtered)
    filtered_bands = torch.stack(filtered_bands)
    return torch.matmul(masks, filtered_bands)