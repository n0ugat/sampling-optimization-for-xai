import numpy as np
from itertools import chain, combinations


def synthetic_dataset_generator(
    n_samples=1000, 
    length=400, 
    noiselevel=0.1, 
    add_random_peaks=True, 
    const_class=None, 
    seed=42):
    """
    Generates a synthetic dataset of sine waves with added noise.
    
    Parameters:
    n_samples (int): Number of samples to generate.
    length (int): Length of each sample.
    noiselevel (float): Standard deviation of the Gaussian noise to be added.
    add_random_peaks (bool): If True, adds random peaks (sine bumps) far from class frequencies.
    const_class (int): If not None, all samples will have this class.
    seed (int): Random seed for reproducibility.
    
    Returns:
    data (numpy.ndarray): Generated dataset of shape (n_samples, length).
    labels (list): List of labels corresponding to each sample.
    """
    
    np.random.seed(seed)
    frequency_classes = [50, 100, 150]
    n_freqs = len(frequency_classes)
    data = np.random.normal(0, noiselevel, (n_samples, length))
    labels = []

    for i in range(n_samples):
<<<<<<< HEAD
        frequency_classes = [int(length*0.2), int(length*0.5), int(length*0.8)] # Frequencies for the classes
        if not const_class:
            class_ = np.random.choice(np.uint8([0,1,2,3,4,5,6,7]), ) # One for each possible combination of frequencies
        else:
            class_ = const_class
        data[i] += np.random.normal(0, noiselevel, length) # Add Gaussian noise with noiselevel as standard deviation
        for freq_idx in range(len(frequency_classes)):
            if class_ & (1 << freq_idx): # Check if the frequency is in the class, using bitwise AND
                freq = frequency_classes[freq_idx]
                data[i] += np.sin(2 * np.pi * freq / length * np.arange(length) + np.random.uniform(0, 2 * np.pi)) # Add sine wave with specified frequency and a random phase
=======
        class_ = const_class if const_class is not None else np.random.choice(1 << n_freqs)
        t = np.arange(length)
>>>>>>> a51cc633e9740b99395f5c2a5351d107d57be385
        
        # Add sine waves for each class frequency based on class_ bitmask
        for j, freq in enumerate(frequency_classes):
            if class_ & (1 << j):
                phase = np.random.uniform(0, 2 * np.pi)
                data[i] += np.sin(2 * np.pi * freq / length * t + phase)

        # Optionally add random peaks (sine bumps) far from class freqs
        if add_random_peaks:
            for _ in range(np.random.randint(0, 3)):
                peak_pos = np.random.randint(0, length // 2)
                peak_height = np.random.uniform(0.5, 2.0)
                if all(abs(peak_pos - f) >= 8 for f in frequency_classes):
                    phase = np.random.uniform(0, 2 * np.pi)
                    data[i] += np.sin(peak_height * np.pi * peak_pos / length * t + phase)

        labels.append([class_])

    return data, labels



def powerset(iterable):
    s = list(iterable)  # Convert the input iterable to a list.
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

def frequency_lrp_dataset(samples, length =  2560, noiselevel = 0.01, M_min=None, M_max = None, integer_freqs = True, return_ks = False, seed = 42):
    ks = np.array([5, 16, 32, 53])
    if not integer_freqs:
        # set seed
        np.random.seed(seed)
        ks = ks + np.random.uniform(0, 1, ks.shape)
        # remove seed
        np.random.seed(None)
    classes_ = powerset(ks)
    all_freqs = np.linspace(1, 60, 60, dtype = np.int32)
    for k in ks:
        idx = np.where(all_freqs == k)[0]
        all_freqs = np.delete(all_freqs, idx)
    data = np.zeros((samples, length))
    labels = []
    for i in range(samples):
        class_ = np.random.randint(0, len(classes_))
        freqs = np.array(classes_[class_])
        data[i] += np.random.normal(0, noiselevel, length)
        # if M is a number then we add M random frequencies
        if M_min is not None:
            M = np.random.randint(M_min, M_max)
            if integer_freqs:
                # append to freqs
                freqs = np.append(freqs, np.random.choice(all_freqs, M-len(freqs), replace = False))
            else:
                # sample uniformly, but exclude a range of 1 Hz around frequencies in ks
                while len(freqs) < M:
                    f = np.random.uniform(1, 60)
                    if np.all(np.abs(ks - f) > 1):
                        freqs = np.append(freqs, f)

            data[i] += np.sum([np.sin(2*np.pi*freq/length*np.arange(0, length) + np.random.uniform(0, 2*np.pi)) for freq in freqs], axis = 0)
        labels.append(class_)
    if return_ks:
        return data, labels, ks
    return data, labels


if __name__ == "__main__":
    import pickle
    data, labels = synthetic_dataset_generator(n_samples=10, length=400, noiselevel=0.5, seed=42)
    print(data.shape)
    print(labels)
    
    with open('notebooks/samples/synthetic_test_dataset.pkl', 'wb') as f:
        pickle.dump({"signals":data, "labels":labels}, f)