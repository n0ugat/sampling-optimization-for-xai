import sys
import os
import pickle
import argparse

import torch

def main(args):
    jobname = args.job_name
    search_folder = f'{args.output_path}/{args.job_name}'
    
    output_filepath = f'{args.output_path}/{args.dataset}_attributions_'
    if args.dataset == 'synthetic':
        output_filepath += f'{args.noise_level}_{args.synth_sig_len}_{not args.no_random_peaks}.pkl'
    elif args.dataset == 'AudioMNIST':
        output_filepath += f'{args.labeltype}.pkl'
    output_filepath = output_filepath.replace('.pkl', f'_{args.n_samples}.pkl')
    if args.debug_mode:
        output_filepath = output_filepath.replace('.pkl', '_debug.pkl')
    if os.path.exists(output_filepath):
        with open(output_filepath, 'rb') as f:
            attributions = pickle.load(f)
    else:
        attributions = {}
        
    for filename in os.listdir(search_folder):
        filepath = os.path.join(search_folder, filename)
        if filepath.endswith('.pkl'):
            print(f'Loading {filepath}')
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                attributions.update(data)
    
    # Save the merged attributions
    with open(output_filepath, 'wb') as f:
        pickle.dump(attributions, f)
        
    
    

if __name__ == '__main__':
    print("Running merge_outputs_from_jobarray.py")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type = str, default = 'outputs', help='Path to save output in')
    parser.add_argument('--dataset', type = str, default = 'AudioMNIST', help='Dataset to use')
    parser.add_argument('--debug_mode', action='store_true', help='Run in debug mode. Stores outputs in deletable .pkl file')
    parser.add_argument('--job_name', type = str, default = None, help='Job name for hpc batch jobs. Used to see which folder to look for attributions in.')
    parser.add_argument('--job_id', type = str, default = None, help='Id of this job')
    parser.add_argument('--n_samples', type = int, default = 10, help='Number of samples to use for evaluation')
    # AudioMNIST
    parser.add_argument('--labeltype', type = str, default = 'digit', help='Type of label to use for AudioMNIST')
    # Synthetic
    parser.add_argument('--noise_level', type = float, default = 0.5, help='Noise level for synthetic dataset. Either 0.8 or 0.01.')
    parser.add_argument('--synth_sig_len', type = int, default = 50, help='Length of the synthetic signals.')
    parser.add_argument('--no_random_peaks', action='store_true', help='Add random peaks to the signals')
    
    args = parser.parse_args()    

    main(args)