# Used to create loss and reward plots for the paper
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import argparse

def plot_loss(input_path, output_path, dataname, method_name, ax=None):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    loss = data['loss']
    num_batches = len(loss)
    x = np.arange(num_batches)
    
    if ax:
        ax.plot(x, loss)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss')
        ax.grid()
    else:
        plt.figure(figsize=(6, 4))
        plt.plot(x, loss, label='Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title(f'{dataname} using {method_name}')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        
        plt.savefig(output_path)
        plt.close()

def plot_rewards(input_path, output_path, dataname, method_name, ax=None):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    rewards = data['rewards']
    num_batches = len(rewards)
    x = np.arange(num_batches)
    
    if ax:
        ax.plot(x, rewards)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Rewards')
        ax.set_title(f'Rewards')
        ax.grid()
    else:
        plt.figure(figsize=(6, 4))
        plt.plot(x, rewards, label='Rewards')
        plt.xlabel('Batch')
        plt.ylabel('Rewards')
        plt.title(f'{dataname} using {method_name}')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type = str, default = 'outputs', help='Path to samples and where to save plots')
    parser.add_argument('--sample_id', type = str, default = None, help='ID of the sample to plot. Example "dReYlZE9".')
    parser.add_argument('--sample_idx', type = int, default = None, help='idx of the sample to plot. None for all samples')
    parser.add_argument('--plot_type', type = str, default = 'both', choices=['both', 'loss', 'rewards'], help='Type of plot to create')
    parser.add_argument('--method_name', type = str, default = 'both', choices=['both', 'SURL', 'FiSURL'], help='Choose method to plot loss and reward for.')
    parser.add_argument('--debug_mode', action='store_true', help='Run in debug mode. Stores outputs in deletable files')
    
    parser.add_argument('--dataset', type = str, default = 'AudioMNIST', choices=['AudioMNIST', 'synthetic'], help='Dataset to use')
    # AudioMNIST
    parser.add_argument('--labeltype', type = str, default = 'digit', choices=['digit', 'gender'], help='Labeltype to use for AudioMNIST')
    args = parser.parse_args()
    
    args.method_name = args.method_name.lower()
    
    if args.method_name == 'both':
        method_names = ['SURL', 'FiSURL']
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
                os.makedirs(os.path.join(args.output_path, 'figures', 'samples', f'{sample_id}{'_debug' if args.debug_mode else ''}', method_name), exist_ok=True)
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
                    if args.plot_type == 'both':
                        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                        fig.suptitle(f'{method_name} training on {dataname} prediction task')
                        plot_loss(input_path, output_path.replace('.png', '_loss.png'), dataname, method_name, ax=axs[0])
                        plot_rewards(input_path, output_path.replace('.png', '_rewards.png'), dataname, method_name, ax=axs[1])
                        plt.tight_layout()
                        plt.savefig(output_path)
                        plt.close()
                    elif args.plot_type == 'loss':
                        plot_loss(input_path, output_path.replace('.png', '_loss.png'), dataname, method_name)
                    elif args.plot_type == 'rewards':
                        plot_rewards(input_path, output_path.replace('.png', '_rewards.png'), dataname, method_name)