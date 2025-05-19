import subprocess
import string
import random

vars = {
    'data_path' : 'data/',
    'model_path': 'models',
    'output_path': 'outputs',
    'dataset' : 'synthetic',
    'debug_mode' : True,
    'save_signals' : True,
# Methods
    'use_FreqRISE' : True,
    'use_SURL' : True,
    'use_FiSURL' : True,
    'use_baselines' : True,
# AudioMNIST
    'labeltype' : 'digit',
# Synthetic
    'noise_level' : '0',
    'synth_sig_len' : '50',
    'no_random_peaks' : False,
    'seed' : '42',
# Hyperparams
    'n_samples' : '10',
    'n_masks' : '3000',
    'batch_size' : '50',
    'num_cells' : '10',
    'use_softmax' : False,
# FreqRISE
    'probability_of_drop' : '0.5',
# Reinforce
    'lr_S' : '0.1',
    'alpha_S' : '1.0',
    'beta_S' : '0.01',
    'decay' : '0.9',
# FiSURL
    'lr_F' : '0.1',
    'alpha_F' : '1.0',
    'beta_F' : '0.01',
    'num_banks' : '10',
    'num_taps' : '501',
    'keep_ratio' : '0.05'
}

if vars['dataset'] == 'AudioMNIST':
    vars['fs'] = '8000'
else:
    vars['fs'] = vars['synth_sig_len']
    
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
        "--fs", vars['fs'],
        "--random_ID", random_ID
    ],
    ["python", "main_evaluation.py", 
        "--model_path", vars['model_path'],
        "--data_path", vars['data_path'],
        "--output_path", vars['output_path'],
        "--dataset", vars['dataset'],
        ("--debug_mode" if vars['debug_mode'] else ""),
        "--n_samples", vars['n_samples'],
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
        "--labeltype", vars['labeltype'],
        "--noise_level", vars['noise_level'],
        "--synth_sig_len", vars['synth_sig_len'],
        ("--no_random_peaks" if vars['no_random_peaks'] else "")
    ],
    ["python", "src/plotting/loss_reward_plotting.py",
        "--output_path", vars['output_path'],
        "--sample_id", random_ID,
        ("--debug_mode" if vars['debug_mode'] else "")
    ],
    ["python", "src/plotting/plot_deletion_curves.py",
        "--output_path", vars['output_path'],
        "--dataset", vars['dataset'],
        ("--debug_mode" if vars['debug_mode'] else ""),
        "--n_samples", vars['n_samples'],
        "--labeltype", vars['labeltype'],
        "--noise_level", vars['noise_level'],
        "--synth_sig_len", vars['synth_sig_len'],
        ("--no_random_peaks" if vars['no_random_peaks'] else "")
    ],
    ["python", "src/plotting/saliency_fft_plotting.py",
        "--output_path", vars['output_path'],
        "--sample_id", random_ID,
        ("--debug_mode" if vars['debug_mode'] else "")
    ]
]

# Run them in sequence
for cmd in scripts:
    cmd_cleaned = list(filter(None, cmd))
    print(f"Running: {' '.join(cmd_cleaned)}")
    subprocess.run(cmd_cleaned, check=True)  # Raises error if script fails