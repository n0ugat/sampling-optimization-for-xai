import matplotlib.pyplot as plt
import numpy as np
import os

def quickplot(signal, axis=None, output_name = "quick_plot", title=None, xlabel=None, ylabel=None, titlefontsize=18, xlabelfontsize=14, ylabelfontsize=14, xfigsize=6, yfigsize=3, grid=True, noticks=False, bottom_padding = None, linewidth=None):
    # Create the output directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    
    # Plot the signal
    plt.figure(figsize=(xfigsize, yfigsize))
    if axis is not None:
        plt.plot(axis, signal, linewidth=(1 if not linewidth else linewidth))
    else:
        plt.plot(np.arange(len(signal)), signal)
    if title:
        plt.title(title, fontsize=titlefontsize)
    if xlabel:
        plt.xlabel(xlabel, fontsize=xlabelfontsize)
    if bottom_padding is not None:
        plt.subplots_adjust(bottom=bottom_padding)  # Adjust as needed
    if ylabel:
        plt.ylabel(ylabel, fontsize=ylabelfontsize)
    if noticks:
        plt.xticks([])
        plt.yticks([])
    if grid:
        plt.grid()
    plt.savefig(os.path.join("temp", f"{output_name}.png"))
    plt.close()
    