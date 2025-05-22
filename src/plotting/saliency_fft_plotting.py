import matplotlib.pyplot as plt

import os
import sys
import pickle
import argparse
import numpy as np

# Add repo directory to system path
repo_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(repo_dir)

from src.plotting.importance_plots import ts_importance

def plot_saliency_with_fft(input_path, output_path, dataname,  method_name=None, cutoff=None):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    signal_fft = data['signal_fft']
    importance = data['importance']
    target_class = data['target_class']
    pred_class = data['prediction']
    pred_correct = target_class == pred_class
    
    if cutoff is not None:
        quantile = np.quantile(importance, cutoff)
        importance = np.where(importance > quantile, importance, 0)
    
    fig, ax = plt.subplots(1,1, figsize=(6, 4))
    ts_importance(
            ax=ax, 
            importance=(importance.numpy() if not isinstance(importance, np.ndarray) else importance), 
            timeseries=(signal_fft.numpy() if not isinstance(signal_fft, np.ndarray) else importance))
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title(f'{dataname} using {method_name}{"_Incorrect Prediction" if not pred_correct else ""}{" cutoff=0.8" if cutoff is not None else ""}')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type = str, default = 'outputs', help='Path to samples and where to save plots')
    parser.add_argument('--sample_id', type = str, default = None, help='ID of the sample to plot. Example "dReYlZE9".')
    parser.add_argument('--sample_idx', type = int, default = None, help='idx of the sample to plot. None for all samples')
    parser.add_argument('--method_name', type = str, default = 'all', choices=['all', 'SURL', 'FiSURL', 'FreqRISE', 'IG', 'LRP', 'Saliency'], help='Choose method to plot loss and reward for.')
    parser.add_argument('--debug_mode', action='store_true', help='Run in debug mode. Stores outputs in deletable files')
    
    parser.add_argument('--dataset', type = str, default = 'AudioMNIST', choices=['AudioMNIST', 'synthetic'], help='Dataset to use')
    # AudioMNIST
    parser.add_argument('--labeltype', type = str, default = 'digit', choices=['digit', 'gender'], help='Labeltype to use for AudioMNIST')
    
    parser.add_argument('--importance_cutoff', type = eval, default = 0.8, help='Cutoff percent for FreqRISE, SURL and FiSURL during evaluation')
    args = parser.parse_args()
    
    args.method_name = args.method_name.lower()
    
    if args.method_name == 'all':
        method_names = ['SURL', 'FiSURL', 'FreqRISE', 'IG', 'LRP', 'Saliency']
    else:
        method_names = [args.method_name]
        
    sample_ids = []
    if not args.sample_id:
        parent_dir = os.path.join(args.output_path, 'samples')
        for name in os.listdir(parent_dir):
            if os.path.isdir(os.path.join(parent_dir, name)):
                sample_ids.append(name)
    else:
        sample_ids.append(args.sample_id)
    if args.dataset == 'AudioMNIST':
        dataname = f'AudioMNIST: {args.labeltype}'
    else:
        dataname = 'Synthetic'
    for sample_id in sample_ids:
        for method_name in method_names:
            if os.path.exists(os.path.join(args.output_path, 'samples', f'{sample_id}{'_debug' if args.debug_mode else ''}', method_name.lower())):
                input_paths = []
                output_paths = []
                if args.sample_idx is None:
                    search_dir = os.path.join(args.output_path, 'samples', f'{sample_id}{'_debug' if args.debug_mode else ''}', method_name.lower())
                    for i in range(len(os.listdir(search_dir)) - 1):
                        input_path = os.path.join(search_dir, f"sample_{i}.pkl")
                        output_path = os.path.join(args.output_path, 'figures', 'samples', f'{sample_id}{'_debug' if args.debug_mode else ''}', method_name, f'{i}.png')
                        input_paths.append(input_path)
                        output_paths.append(output_path)
                else:
                    input_path = os.path.join(args.output_path, 'samples', f'{sample_id}{'_debug' if args.debug_mode else ''}', method_name.lower(), f"sample_{args.sample_idx}.pkl")
                    output_path = os.path.join(args.output_path, 'figures', 'samples', f'{sample_id}{'_debug' if args.debug_mode else ''}', method_name, f'{args.sample_idx}.png')
                    input_paths.append(input_path)
                    output_paths.append(output_path)
                
                for input_path, output_path in zip(input_paths, output_paths):
                    plot_saliency_with_fft(input_path, output_path.replace('.png', f'_importance_fft_co{args.importance_cutoff}.png'), dataname, method_name, cutoff = args.importance_cutoff)
                    plot_saliency_with_fft(input_path, output_path.replace('.png', '_importance_fft.png'), dataname, method_name)