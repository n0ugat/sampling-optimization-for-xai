import pickle
import argparse
import os

def main(args):
    # Create path where the evaluation data is stored
    if args.dataset == 'synthetic':
        evaluation_path = f'{args.output_path}/{args.dataset}_evaluation_{args.noise_level}_{args.synth_sig_len}_{not args.no_random_peaks}.pkl'
    elif args.dataset == 'AudioMNIST':
        evaluation_path = f'{args.output_path}/{args.dataset}_evaluation_{args.labeltype}.pkl'
    evaluation_path = evaluation_path.replace('.pkl', f'_{args.n_samples}.pkl')
    if args.debug_mode:
        evaluation_path = evaluation_path.replace('.pkl', '_debug.pkl')
    
    # Maske the evaluation data exists, and then read it
    assert os.path.exists(evaluation_path), "No evaluation file those parameters exists"
    with open(evaluation_path, 'rb') as f:
        data = pickle.load(f)
    
    # Make directory for the output
    output_dir = os.path.join(args.output_path, 'evaluation_scores')
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the file with the evaluation scores
    with open(os.path.join(output_dir, evaluation_path.replace(f'{args.output_path}/', '')).replace('.pkl', '.txt'), 'w') as output_file:
        output_file.write('Evaluation Scores:\n')
        output_file.write(f'Dataset: {args.dataset}\n')
        # Parameters for the synthetic dataset
        if args.dataset == 'synthetic':
            output_file.write(f'Noise Level: {args.noise_level}\n')
            output_file.write(f'Signal Length: {args.synth_sig_len}\n')
            output_file.write(f'Random Peaks: {not args.no_random_peaks}\n')
        # Parameter for AudioMNIST
        elif args.dataset == 'AudioMNIST':
            output_file.write(f'Labeltype: {args.labeltype}\n')
        # Write evaluation scores for every method in the data
        for method in data['complexity scores'].keys():
            output_file.write(f'\n{method}\n')
            output_file.write(f'Complexity Score: {float(data['complexity scores'][method])}\n')
            output_file.write(f'Faithfulness Score: {float(data['deletion curves'][method]['AUC'])}\n')
            # Only synthetic dataset has a localization score
            if args.dataset == 'synthetic':
                output_file.write(f'Localization Score: {float(data['localization scores'][method])}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type = str, default = 'outputs', help='Path to save output')
    parser.add_argument('--dataset', type = str, default = 'AudioMNIST', choices=['AudioMNIST', 'synthetic'], help='Dataset to use')
    parser.add_argument('--debug_mode', action='store_true', help='Run in debug mode. Stores outputs in deletable files')
    parser.add_argument('--n_samples', type = int, default = 10, help='Number of samples to use for evaluation')
    # AudioMNIST
    parser.add_argument('--labeltype', type = str, default = 'gender', choices=['gender', 'digit'], help='Type of label to use for AudioMNIST')
    # Synthetic
    parser.add_argument('--noise_level', type = float, default = 0.0, help='Noise level for synthetic data')
    parser.add_argument('--synth_sig_len', type = int, default = 50, help='Length of the synthetic signals.')
    parser.add_argument('--no_random_peaks', action='store_true', help='Add random peaks to the signals')
    args = parser.parse_args() 
    
    main(args)