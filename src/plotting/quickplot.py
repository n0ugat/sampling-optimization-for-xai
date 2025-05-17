import matplotlib.pyplot as plt
import numpy as np
import os

def quickplot(signal, output_name = "quick_plot"):
    # Create the output directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    
    # Plot the signal
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(signal)), signal)
    
    plt.grid()
    plt.savefig(os.path.join("temp", f"{output_name}.png"))
    plt.close()
    