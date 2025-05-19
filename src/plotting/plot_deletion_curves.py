# Used to deletion curves in paper
import matplotlib.pyplot as plt
import os
import pickle
import argparse

def plot_deletion_curve(method_name_with_params, mean_prob, quantiles, AUC, output_path, dataname):
    method_names = ['FreqRISE', 'SURL', 'FiSURL', 'random', 'amplitude', 'saliency', 'IG', 'LRP']
    method_name = ""
    for mn in method_names:
        if method_name_with_params.lower().startswith(mn.lower()):
            method_name = mn
            break
    plt.figure(figsize=(6, 4))
    plt.plot(quantiles, mean_prob, label=f'{method_name} (AUC = {AUC:.2f})')
    plt.xlabel('Quantile')
    plt.ylabel('Mean True Class Probability')
    plt.title(f'Deletion Curve for {method_name} on {dataname}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    output_file = os.path.join(output_path, f'{method_name_with_params}_deletion_curve.png')
    plt.savefig(output_file)
    plt.close()
    

def plot_joined_deletion_curves(deletion_curves, output_path, dataname):
    method_names_and_colors = {'FreqRISE':'blue', 'SURL':'orange', 'FiSURL':'green', 'random':'red', 'amplitude':'purple', 'saliency':'brown', 'IG':'pink', 'LRP':'gray'}
    
    plt.figure(figsize=(6, 4))
    for method_name_with_params, deletion_curve in deletion_curves.items():
        mean_prob = deletion_curve['mean_prob']
        quantiles = deletion_curve['quantiles']
        AUC = deletion_curve['AUC']
        method_name = ""
        for mn in method_names_and_colors.keys():
            if method_name_with_params.lower().startswith(mn.lower()):
                method_name = mn
                break
    
        plt.plot(quantiles, mean_prob, label=method_name, color=method_names_and_colors[method_name])
    plt.xlabel('Quantile')
    plt.ylabel('Mean True Class Probability')
    plt.title(f'Deletion Curves for {dataname}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
    plt.grid()
    plt.tight_layout()
    
    output_file = os.path.join(output_path, f'all_deletion_curves.png')
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type = str, default = 'outputs', help='Path to samples and where to save plots')
    parser.add_argument('--dataset', type = str, default = 'AudioMNIST', choices=['AudioMNIST', 'synthetic'], help='Dataset to use')
    parser.add_argument('--debug_mode', action='store_true', help='Debug mode')
    parser.add_argument('--n_samples', type = int, default = 10, help='Number of samples to use for evaluation')
    # AudioMNIST
    parser.add_argument('--labeltype', type = str, default = 'digit', choices=['digit', 'gender'], help='Labeltype to use for AudioMNIST')
    # synthetic
    parser.add_argument('--noise_level', type = float, default = 0.0, help='Noise level to use for synthetic dataset')
    parser.add_argument('--synth_sig_len', type = int, default = 50, help='Signal length to use for synthetic dataset')
    parser.add_argument('--no_random_peaks', action='store_true', help='Add random peaks to the signals')
    args = parser.parse_args()
    
    assert args.dataset in ['AudioMNIST', 'synthetic'], "Dataset must be either AudioMNIST or synthetic"
    
    if args.dataset == 'AudioMNIST':
        input_name = f'{args.dataset}_evaluation_{args.labeltype}.pkl'
    elif args.dataset == 'synthetic':
        input_name = f'{args.dataset}_evaluation_{args.noise_level}_{args.synth_sig_len}_{not args.no_random_peaks}.pkl'
    input_name = input_name.replace('.pkl', f'_{args.n_samples}.pkl')
    if args.debug_mode:
        input_name = input_name.replace('.pkl', '_debug.pkl')
    input_path = os.path.join(args.output_path, input_name)
    assert os.path.exists(input_path), f"Input path {input_path} does not exist"
    
    output_path = os.path.join(args.output_path, 'figures', 'deletion_curves', input_name)
    os.makedirs(output_path, exist_ok=True)
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        deletion_curves = data['deletion curves']
    
    dataname = args.dataset
    if args.dataset == 'AudioMNIST':
        dataname = f'AudioMNIST_{args.labeltype}'
    for method_name in deletion_curves.keys():
        plot_deletion_curve(method_name, deletion_curves[method_name]['mean_prob'], deletion_curves[method_name]['quantiles'], deletion_curves[method_name]['AUC'], output_path, dataname)
        
    plot_joined_deletion_curves(deletion_curves, output_path, dataname)