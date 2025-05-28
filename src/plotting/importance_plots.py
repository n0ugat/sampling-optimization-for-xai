# plot frequency with importance
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

def ts_importance(ax, importance, timeseries, width = 1.0, cmap = 'Greens', alpha = 0.7, colorbar = True):
    axis = np.arange(len(timeseries))
    
    my_cmap = cm.get_cmap(cmap)
    importance = importance.squeeze()
    
    # Normalize using actual min/max of input (still needed for colorbar)
    norm = Normalize(vmin=np.nanmin(importance), vmax=np.nanmax(importance))
    
    # Map importance values to colors
    plot_col_mean = my_cmap(norm(importance))
    
    # Plot importance-colored bars
    ax.bar(axis, np.ones_like(importance) * (np.max(timeseries) - np.min(timeseries)) * 2,
           bottom=np.min(timeseries), width=width, color=plot_col_mean, alpha=alpha)
    
    # Plot the time series line
    ax.plot(axis, timeseries, color='black', alpha=1.0, linewidth=1)
    
    # Adjust y-limits
    ax.set_ylim(timeseries.min(), timeseries.max() + (timeseries.max() - timeseries.min()) * 0.10)
    
    # Add colorbar
    if colorbar:
       sm = ScalarMappable(cmap=my_cmap, norm=norm)
       sm.set_array([])  # Required for colorbar to work correctly
       cbar = ax.figure.colorbar(sm, ax=ax, orientation='vertical')
       cbar.set_label('Importance')