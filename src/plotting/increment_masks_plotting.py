# Used to deletion curves in paper
import matplotlib.pyplot as plt
import os
import pickle
import argparse 
import string
import random   

def plot_incrementing_masks(data, output_path, dataname):
    method_names_and_colors = {'FreqRISE':'blue', 'SURL':'orange', 'FiSURL':'green'}
    
    n_masks_list = set()
    for key, value in data['n_masks'].items():
        # breakpoint()
        if key.lower().startswith('freqrise'): # Could be any of the models
            n_masks_list.add(value)
    n_masks_list = list(n_masks_list)
    n_masks_list.sort()
    
    freqrise_complexities = []
    freqrise_localizations = []
    freqrise_faithfulness = []
    surl_complexities = []
    surl_localizations = []
    surl_faithfulness = []
    fisurl_complexities = []
    fisurl_localizations = []
    fisurl_faithfulness = []
    
    chars = string.ascii_letters + string.digits  # a-zA-Z0-9
    random_ID = ''.join(random.choices(chars, k=5))
    
    for n_masks in n_masks_list:
        for key, value in data['n_masks'].items():
            if value == n_masks:
                if key.lower().startswith('freqrise'):
                    freqrise_complexities.append(data['complexity scores'][key])
                    freqrise_faithfulness.append(data['deletion curves'][key]['AUC'])
                    if args.dataset == 'synthetic':
                        freqrise_localizations.append(data['localization scores'][key])
                elif key.lower().startswith('surl'):
                    surl_complexities.append(data['complexity scores'][key])
                    surl_faithfulness.append(data['deletion curves'][key]['AUC'])
                    if args.dataset == 'synthetic':
                        surl_localizations.append(data['localization scores'][key])
                elif key.lower().startswith('fisurl'):
                    fisurl_complexities.append(data['complexity scores'][key])
                    fisurl_faithfulness.append(data['deletion curves'][key]['AUC'])
                    if args.dataset == 'synthetic':
                        fisurl_localizations.append(data['localization scores'][key])
    
    # Plot complexities together
    plt.figure(figsize=(6, 4))
    plt.plot(n_masks_list, freqrise_complexities, label='FreqRISE', color=method_names_and_colors['FreqRISE'])
    plt.plot(n_masks_list, surl_complexities, label='SURL', color=method_names_and_colors['SURL'])
    plt.plot(n_masks_list, fisurl_complexities, label='FiSURL', color=method_names_and_colors['FiSURL'])
    plt.xlabel('Number of masks')
    plt.ylabel('Complexity')
    plt.title(f'Complexity vs Number of masks for {dataname} prediction task')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{random_ID}_complexity.png'))
    plt.close()
    
    # Plot faithfulness together
    plt.figure(figsize=(6, 4))
    plt.plot(n_masks_list, freqrise_faithfulness, label='FreqRISE', color=method_names_and_colors['FreqRISE'])
    plt.plot(n_masks_list, surl_faithfulness, label='SURL', color=method_names_and_colors['SURL'])
    plt.plot(n_masks_list, fisurl_faithfulness, label='FiSURL', color=method_names_and_colors['FiSURL'])
    plt.xlabel('Number of masks')
    plt.ylabel('Faithfulness')
    plt.title(f'Faithfulness vs Number of masks for {dataname} prediction task')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{random_ID}_faithfulness.png'))
    plt.close()
    
    # Plot localizations together
    if args.dataset == 'synthetic':
        plt.figure(figsize=(6, 4))
        plt.plot(n_masks_list, freqrise_localizations, label='FreqRISE', color=method_names_and_colors['FreqRISE'])
        plt.plot(n_masks_list, surl_localizations, label='SURL', color=method_names_and_colors['SURL'])
        plt.plot(n_masks_list, fisurl_localizations, label='FiSURL', color=method_names_and_colors['FiSURL'])
        plt.xlabel('Number of masks')
        plt.ylabel('Localization')
        plt.title(f'Localization vs Number of masks for {dataname} prediction task')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{random_ID}_localization.png'))
        plt.close()
        
    # Create metafile with the method names
    with open(os.path.join(output_path, f'{random_ID}_metafile.txt'), 'w') as metafile:
        metafile.write('Method names and parameters:\n')
        for key, value in data['n_masks'].items():
            if key.lower().startswith('freqrise'):
                metafile.write(f'FreqRISE: {key}\n')
            elif key.lower().startswith('surl'):
                metafile.write(f'SURL: {key}\n')
            elif key.lower().startswith('fisurl'):
                metafile.write(f'FiSURL: {key}\n')
                    
    

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
    input_name = input_name.replace('.pkl', '_im.pkl')
    input_path = os.path.join(args.output_path, input_name)
    
    assert os.path.exists(input_path), f"Input path {input_path} does not exist"
    
    output_path = os.path.join(args.output_path, 'figures', 'incrementing_masks', input_name)
    os.makedirs(output_path, exist_ok=True)
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    if args.dataset == 'AudioMNIST':
        dataname = f'AudioMNIST: {args.labeltype}'
    else:
        dataname = 'Synthetic'    
    
    plot_incrementing_masks(data, output_path, dataname)