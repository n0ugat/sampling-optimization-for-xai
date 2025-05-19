# Used to create loss and reward plots for the paper
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import argparse

def plot_loss(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    loss = data['loss']
    num_batches = len(loss)
    x = np.arange(num_batches)
    
    plt.figure(figsize=(6, 4))
    plt.plot(x, loss, label='Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss over batches')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

def plot_rewards(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    rewards = data['rewards']
    num_batches = len(rewards)
    x = np.arange(num_batches)
    
    plt.figure(figsize=(6, 4))
    plt.plot(x, rewards, label='Rewards')
    plt.xlabel('Batch')
    plt.ylabel('Rewards')
    plt.title('Rewards over batches')
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
    args = parser.parse_args()
    
    args.method_name = args.method_name.lower()
    assert args.sample_id is not None, "sample_id must be provided"
    
    if args.method_name == 'both':
        method_names = ['SURL', 'FiSURL']
    else:
        method_names = [args.method_name]
    
    for method_name in method_names:
        if os.path.exists(os.path.join(args.output_path, 'samples', f'{args.sample_id}{'_debug' if args.debug_mode else ''}', method_name.lower())):
            os.makedirs(os.path.join(args.output_path, 'figures', f'{args.sample_id}{'_debug' if args.debug_mode else ''}', method_name), exist_ok=True)
            input_paths = []
            output_paths = []
            if args.sample_idx is None:
                search_dir = os.path.join(args.output_path, 'samples', f'{args.sample_id}{'_debug' if args.debug_mode else ''}', method_name.lower())
                for i in range(len(os.listdir(search_dir)) - 1):
                    input_path = os.path.join(search_dir, f"sample_{i}.pkl")
                    output_path = os.path.join(args.output_path, 'figures', 'samples', f'{args.sample_id}{'_debug' if args.debug_mode else ''}', method_name, f'{i}.png')
                    input_paths.append(input_path)
                    output_paths.append(output_path)
            else:
                input_path = os.path.join(args.output_path, 'samples', f'{args.sample_id}{'_debug' if args.debug_mode else ''}', method_name.lower(), f"sample_{args.sample_idx}.pkl")
                output_path = os.path.join(args.output_path, 'figures', 'samples', f'{args.sample_id}{'_debug' if args.debug_mode else ''}', method_name, f'{args.sample_idx}.png')
                input_paths.append(input_path)
                output_paths.append(output_path)
            
            for input_path, output_path in zip(input_paths, output_paths):
                if args.plot_type == 'both':
                    plot_loss(input_path, output_path.replace('.png', '_loss.png'))
                    plot_rewards(input_path, output_path.replace('.png', '_rewards.png'))
                elif args.plot_type == 'loss':
                    plot_loss(input_path, output_path.replace('.png', '_loss.png'))
                elif args.plot_type == 'rewards':
                    plot_rewards(input_path, output_path.replace('.png', '_rewards.png'))