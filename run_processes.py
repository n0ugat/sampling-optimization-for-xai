import subprocess
import string
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'AudioMNIST', choices=['AudioMNIST', 'synthetic'], help='Dataset to use')
parser.add_argument('--labeltype', type = str, default = 'digit', choices=['gender', 'digit'], help='Type of label to use for AudioMNIST')
args = parser.parse_args()

vars = {
    'data_path' : 'data/',
    'model_path': 'models',
    'output_path': 'outputs',
    'dataset' : args.dataset,
    'debug_mode' : False,
    'save_signals' : False,
# What to compute
    'incrementing_masks' : False,
    'make_deletion_curve_plots' : True,
    'save_evaluation_scores_in_txt' : True,
# Methods
    'use_FreqRISE' : True,
    'use_SURL' : True,
    'use_FiSURL' : True,
    'use_baselines' : True,
# AudioMNIST
    'labeltype' : args.labeltype,
# Synthetic
    'noise_level' : '0',
    'synth_sig_len' : '100',
    'no_random_peaks' : False,
    'seed' : '43',
# Hyperparams
    'n_samples' : '1000',
    'n_masks' : '10000',
    'batch_size' : '10',
    'num_cells' : '15',
    'use_softmax' : False,
# FreqRISE
    'probability_of_drop' : '0.5',
# Reinforce
    'lr_S' : '0.1',
    'alpha_S' : '1.0',
    'beta_S' : '30.0',
    'decay' : '0.9',
# FiSURL
    'lr_F' : '0.1',
    'alpha_F' : '1.0',
    'beta_F' : '30.0',
    'num_banks' : '15',
    'num_taps' : '1001',
    'keep_ratio' : '0.05'
}

if vars['incrementing_masks'] and vars['use_baselines']:
    response = input("Incrementing masks and use_baselines shouldn't be used together. Are you sure you want to continue? (y/n): ")
    if response.lower() not in ['y', 'ye', 'yes']:
        print("Exiting...")
        exit(0)
        
    
assert vars['dataset'] in ['AudioMNIST', 'synthetic'], "Dataset must be either 'AudioMNIST' or 'synthetic'"
if vars['dataset'] == 'AudioMNIST':
    assert vars['labeltype'] in ['digit', 'gender'], "Labeltype must be either 'digit' or 'gender'"
if vars['dataset'] == 'synthetic':
    assert float(vars['noise_level']) >= 0.0, "Noise level must be greater than or equal to 0.0"
    assert int(vars['synth_sig_len']) > 0, "Synthetic signal length must be greater than 0"
if vars['use_FreqRISE']:
    assert float(vars['probability_of_drop']) >= 0.0 and float(vars['probability_of_drop']) <= 1.0, "Probability of drop must be between 0.0 and 1.0"
if vars['use_FreqRISE'] or vars['use_SURL']:
    assert int(vars['num_cells']) > 0, "Number of cells must be greater than 0 and less than or equal to fs"
if vars['use_FiSURL']:
    assert int(vars['num_banks']) > 0, "Number of banks must be greater than 0 and less than or equal to fs"
    assert int(vars['num_taps']) > 0, "Number of taps must be greater than 0"
    assert float(vars['keep_ratio']) >= 0.0 and float(vars['keep_ratio']) <= 1.0, "Keep ratio must be between 0.0 and 1.0"
    
chars = string.ascii_letters + string.digits  # a-zA-Z0-9
random_ID = ''.join(random.choices(chars, k=8))

# Each script with its arguments
scripts = [
    ["python", "main_attributions.py", 
        "--data_path", vars['data_path'], 
        "--model_path", vars['model_path'],
        "--output_path", vars['output_path'],
        "--dataset", vars['dataset'],
        ("--debug_mode" if vars['debug_mode'] else ""),
        ("--save_signals" if vars['save_signals'] else ""),
        ("--incrementing_masks" if vars['incrementing_masks'] else ""),
        ("--use_FreqRISE" if vars['use_FreqRISE'] else ""),
        ("--use_SURL" if vars['use_SURL'] else ""),
        ("--use_FiSURL" if vars['use_FiSURL'] else ""),
        ("--use_baselines" if vars['use_baselines'] else ""),
        "--labeltype", vars['labeltype'], 
        "--noise_level", vars['noise_level'],
        "--synth_sig_len", vars['synth_sig_len'],
        ("--no_random_peaks" if vars['no_random_peaks'] else ""),
        ("--seed" if vars['seed'] else ""), (vars['seed'] if vars['seed'] else ""), 
        "--n_samples", vars['n_samples'],
        "--n_masks", vars['n_masks'],
        "--batch_size", vars['batch_size'], 
        "--num_cells", vars['num_cells'],
        ("--use_softmax" if vars['use_softmax'] else ""),
        "--probability_of_drop", vars['probability_of_drop'], 
        "--lr_S", vars['lr_S'],
        "--alpha_S", vars['alpha_S'],
        "--beta_S", vars['beta_S'], 
        "--decay", vars['decay'],
        "--lr_F", vars['lr_F'], 
        "--alpha_F", vars['alpha_F'],
        "--beta_F", vars['beta_F'],
        "--num_banks", vars['num_banks'], 
        "--num_taps", vars['num_taps'],
        "--keep_ratio", vars['keep_ratio'], 
        "--random_ID", random_ID
    ],
    ["python", "main_evaluation.py", 
        "--model_path", vars['model_path'],
        "--data_path", vars['data_path'],
        "--output_path", vars['output_path'],
        "--dataset", vars['dataset'],
        ("--debug_mode" if vars['debug_mode'] else ""),
        "--n_samples", vars['n_samples'],
        ("--incrementing_masks" if vars['incrementing_masks'] else ""),
        "--labeltype", vars['labeltype'],
        "--noise_level", vars['noise_level'],
        "--synth_sig_len", vars['synth_sig_len'],
        ("--no_random_peaks" if vars['no_random_peaks'] else ""),
        ("--seed" if vars['seed'] else ""), (vars['seed'] if vars['seed'] else ""),
        "--compute_deletion_scores",
        "--compute_complexity_scores",
        ("--compute_localization_scores" if vars['dataset'] == 'synthetic' else '')  
    ],
    ["python", "compute_evaluation_scores.py",
        "--output_path", vars['output_path'],
        "--dataset", vars['dataset'],
        ("--debug_mode" if vars['debug_mode'] else ""),
        "--n_samples", vars['n_samples'],
        ("--incrementing_masks" if vars['incrementing_masks'] else ""),
        "--labeltype", vars['labeltype'],
        "--noise_level", vars['noise_level'],
        "--synth_sig_len", vars['synth_sig_len'],
        ("--no_random_peaks" if vars['no_random_peaks'] else "")
    ],
    ["python", "src/plotting/plot_deletion_curves.py",
        "--output_path", vars['output_path'],
        "--dataset", vars['dataset'],
        ("--debug_mode" if vars['debug_mode'] else ""),
        "--n_samples", vars['n_samples'],
        ("--incrementing_masks" if vars['incrementing_masks'] else ""),
        "--labeltype", vars['labeltype'],
        "--noise_level", vars['noise_level'],
        "--synth_sig_len", vars['synth_sig_len'],
        ("--no_random_peaks" if vars['no_random_peaks'] else "")
    ],
    ["python", "src/plotting/loss_reward_plotting.py",
        "--output_path", vars['output_path'],
        "--sample_id", random_ID,
        ("--debug_mode" if vars['debug_mode'] else ""),
        "--dataset", vars['dataset'],
        "--labeltype", vars['labeltype']
    ],
    ["python", "src/plotting/saliency_fft_plotting.py",
        "--output_path", vars['output_path'],
        "--sample_id", random_ID,
        ("--debug_mode" if vars['debug_mode'] else ""),
        "--dataset", vars['dataset'],
        "--labeltype", vars['labeltype']
    ],
    ["python", "src/plotting/increment_masks_plotting.py",
        "--output_path", vars['output_path'],
        "--dataset", vars['dataset'],
        ("--debug_mode" if vars['debug_mode'] else ""),
        "--n_samples", vars['n_samples'],
        "--labeltype", vars['labeltype'],
        "--noise_level", vars['noise_level'],
        "--synth_sig_len", vars['synth_sig_len'],
        ("--no_random_peaks" if vars['no_random_peaks'] else "")
    ]
]

final_scripts = [scripts[0], scripts[1]]
if vars['save_evaluation_scores_in_txt']:
    final_scripts.append(scripts[2])
if vars['make_deletion_curve_plots']:
    final_scripts.append(scripts[3])
if vars['save_signals']:
    final_scripts.append(scripts[4])
    final_scripts.append(scripts[5])
if vars['incrementing_masks']:
    final_scripts.append(scripts[6])
    
# Run them in sequence
for cmd in final_scripts:
    cmd_cleaned = list(filter(None, cmd))
    print(f"Running: {' '.join(cmd_cleaned)}")
    subprocess.run(cmd_cleaned, check=True)  # Raises error if script fails