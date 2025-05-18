import torch
import argparse
from src.explainability import deletion_curves, complexity_scores, localization_scores
from src.data import load_data
from src.models import load_model
import pickle
import os
import numpy as np


def main(args):
    print('Loading data')
    test_loader = load_data(args)
    print('Loading model')
    model = load_model(args)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu'
    print(f'Using device: {device}')
    if args.dataset == 'synthetic':
        attributions_path = f'{args.output_path}/{args.dataset}_attributions_{args.noise_level}_{args.synth_sig_len}_{not args.no_random_peaks}.pkl'
        output_path = f'{args.output_path}/{args.dataset}_evaluation_{args.noise_level}_{args.synth_sig_len}_{not args.no_random_peaks}.pkl'
    elif args.dataset == 'AudioMNIST':
        attributions_path = f'{args.output_path}/{args.dataset}_attributions_{args.labeltype}.pkl'
        output_path = f'{args.output_path}/{args.dataset}_evaluation_{args.labeltype}.pkl'
    attributions_path = attributions_path.replace('.pkl', f'_{args.n_samples}.pkl')
    output_path = output_path.replace('.pkl', f'_{args.n_samples}.pkl')
    if args.debug_mode:
        attributions_path = attributions_path.replace('.pkl', '_debug.pkl')
        output_path = output_path.replace('.pkl', '_debug.pkl')

    if os.path.exists(attributions_path):
        with open(attributions_path, 'rb') as f:
            attributions = pickle.load(f)
    else:
        # raise error
        raise FileNotFoundError(f'Attributions not found at {attributions_path}')
    
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            evaluation = pickle.load(f)
    else:
        evaluation = {}
    
    if not 'deletion curves' in evaluation:
        evaluation['deletion curves'] = {}
    if args.compute_deletion_scores:
        print('Computing deletion scores')
        quantiles = np.arange(0, 1, 0.05)
        for key, value in attributions.items():
            if key == 'predictions' or key == 'labels':
                continue
            if (not key in evaluation['deletion curves']) or args.debug_mode:
                if key.startswith('freqrise') or key.startswith('surl') or key.startswith('fisurl'):
                    cutoff = args.freqrise_cutoff
                else:
                    cutoff = None
                evaluation['deletion curves'][key] = deletion_curves(model, test_loader, value, quantiles, device=device, cutoff = cutoff, method_name=key)

        if not 'random' in evaluation['deletion curves']:
            # compute random deletion scores
            evaluation['deletion curves']['random'] = deletion_curves(model, test_loader, 'random', quantiles, device = device, method_name='random')
            # get amplitude mask
            evaluation['deletion curves']['amplitude'] = deletion_curves(model, test_loader, 'amplitude', quantiles, device = device, method_name='amplitude')
        print('Deletion scores computed')
    
    if not 'complexity scores' in evaluation:
        evaluation['complexity scores'] = {}
    if args.compute_complexity_scores:
        print('Computing complexity scores')
        for key, value in attributions.items():
            if key in ['predictions', 'labels']:
                continue
            if not key in evaluation['complexity scores'] or args.debug_mode:
                value = torch.cat(value).numpy()
                if key.startswith('freqrise') or key.startswith('surl') or key.startswith('fisurl'):
                    cutoff = args.freqrise_cutoff
                    only_pos = False
                else:
                    cutoff = None
                    only_pos = True
                evaluation['complexity scores'][key] = np.mean(complexity_scores(value, cutoff = cutoff, only_pos = only_pos))
        print('Complexity scores computed')

    if not 'localization scores' in evaluation and args.dataset == 'synthetic':
        evaluation['localization scores'] = {}
    if args.compute_localization_scores:
        print('Computing localization scores')
        for key, value in attributions.items():
            if key in ['predictions', 'labels']:
                continue
            if not key in evaluation['localization scores'] or args.debug_mode:
                value = torch.cat(value).numpy()
                if key.startswith('freqrise') or key.startswith('surl') or key.startswith('fisurl'):
                    cutoff = args.freqrise_cutoff
                    only_pos = False
                else:
                    cutoff = None
                    only_pos = True
                evaluation['localization scores'][key] = np.mean(localization_scores(value, attributions['labels'], cutoff = cutoff, only_pos = only_pos))
        print('Localization scores computed')
    
    with open(output_path, 'wb') as f:
        pickle.dump(evaluation, f)
    return None



if __name__ == '__main__':
    print("Running main_evaluation.py")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = 'models', help='Path to model')
    parser.add_argument('--data_path', type = str, default = 'data/', help='Path to AudioMNIST data')
    parser.add_argument('--output_path', type = str, default = 'outputs', help='Path to save output')
    parser.add_argument('--dataset', type = str, default = 'AudioMNIST', choices=['AudioMNIST', 'synthetic'], help='Dataset to use')
    parser.add_argument('--debug_mode', action='store_true', help='Run in debug mode. Stores outputs in deletable files')
    parser.add_argument('--n_samples', type = int, default = 10, help='Number of samples to use for evaluation')
    # AudioMNIST
    parser.add_argument('--labeltype', type = str, default = 'gender', choices=['gender', 'digit'], help='Type of label to use for AudioMNIST')
    # Synthetic
    parser.add_argument('--noise_level', type = int, default = 0, help='Noise level for synthetic data')
    parser.add_argument('--synth_sig_len', type = int, default = 50, help='Length of the synthetic signals.')
    parser.add_argument('--no_random_peaks', action='store_true', help='Add random peaks to the signals')
    parser.add_argument('--seed', type = int, default = 42, help='Seed for random number generator')
    
    parser.add_argument('--freqrise_cutoff', type = eval, default = None, help='Cutoff percent for FreqRISE during evaluation')

    parser.add_argument('--compute_deletion_scores', action='store_true', help='Compute deletion scores')
    parser.add_argument('--compute_localization_scores', action='store_true', help='Compute localization scores. NB only for synthetic data.')
    parser.add_argument('--compute_complexity_scores', action='store_true', help='Compute complexity scores')
    args = parser.parse_args()
    
    if args.compute_localization_scores:
        assert args.dataset == 'synthetic', "Localization scores can only be computed for synthetic data"
    main(args)