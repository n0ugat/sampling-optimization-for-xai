import numpy as np

def synthetic_dataset_generator(
    n_samples=1000, 
    length=100, 
    noiselevel=0.0, 
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
    freq_comps = length // 2
    frequency_classes = [int(freq_comps*0.2), int(freq_comps*0.5), int(freq_comps*0.8)]
    n_freqs = len(frequency_classes)
    data = np.random.normal(0, noiselevel, (n_samples, length))
    labels = []

    for i in range(n_samples):
        # Select a class based on the bitmask
        class_ = const_class if const_class is not None else np.random.choice(1 << n_freqs)
        t = np.arange(length)
        
        # Add sine waves for each class frequency based on class_ bitmask
        for j, freq in enumerate(frequency_classes):
            if class_ & (1 << j):
                phase = np.random.uniform(0, 2 * np.pi)
                data[i] += np.sin(2 * np.pi * freq / length * t + phase)

        # Optionally add random peaks (sine bumps) far from class freqs
        if add_random_peaks:
            for _ in range(np.random.randint(0, 8)):
                peak_pos = np.random.randint(0, length // 2)
                peak_height = np.random.uniform(0.5, 2.0)
                if all(abs(peak_pos - f) >= 5 for f in frequency_classes):
                    phase = np.random.uniform(0, 2 * np.pi)
                    data[i] += np.sin(peak_height * np.pi * peak_pos / length * t + phase)

        labels.append([class_])
    return data, labels

if __name__ == "__main__":
    import pickle
    data, labels = synthetic_dataset_generator(n_samples=10, length=400, noiselevel=0.5, seed=42)
    print(data.shape)
    print(labels)
    
    with open('notebooks/samples/synthetic_test_dataset.pkl', 'wb') as f:
        pickle.dump({"signals":data, "labels":labels}, f)