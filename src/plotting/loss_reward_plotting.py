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
    parser.add_argument('--sample_id', type = str, default = None, help='ID of the sample to plot')
    parser.add_argument('--sample_idx', type = int, default = None, help='idx of the sample to plot. None for all samples')
    parser.add_argument('--plot_type', type = str, default = 'both', choices=['both', 'loss', 'rewards'], help='Type of plot to create')
    args = parser.parse_args()
    
    assert args.sample_id is not None, "sample_id must be provided"
    
    os.makedirs(os.path.join(args.output_path, 'figures', args.sample_id), exist_ok=True)
    input_paths = []
    output_paths = []
    if args.sample_idx is None:
        for i in range(len(os.listdir(os.path.join(args.output_path, 'samples', args.sample_id))) - 1):
            input_path = os.path.join(args.output_path, 'samples', args.sample_id, f"sample_{i}.pkl")
            output_path = os.path.join(args.output_path, 'figures', args.sample_id, f'{i}.png')
            input_paths.append(input_path)
            output_paths.append(output_path)
    else:
        input_path = os.path.join(args.output_path, 'samples', args.sample_id, f"sample_{args.sample_idx}.pkl")
        output_path = os.path.join(args.output_path, 'figures', args.sample_id, f'{args.sample_idx}.png')
        input_paths.append(input_path)
        output_paths.append(output_path)
    
    for input_path, output_path in zip(input_paths, output_paths):
        if args.plot_type == 'both':
            plot_loss(input_path, output_path.replace('.png', '_loss.png'))
            plot_rewards(input_path, output_path.replace('.png', '_rewards.png'))
        elif args.plot_type == 'loss':
            plot_loss(input_path, output_path)
        elif args.plot_type == 'rewards':
            plot_rewards(input_path, output_path)