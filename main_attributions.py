import torch
import argparse
import pickle
import os
import random
import string

from src.explainability import FreqRISE, FiSURL, SURL, compute_gradient_scores
from src.data import load_data
from src.models import load_model

def main(args):
    print('Loading data')
    test_loader = load_data(args)
    print('Loading model')
    model = load_model(args)
    # device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    if args.dataset == 'synthetic':
        output_path = f'{args.output_path}/{args.dataset}_attributions_{args.noise_level}_{args.synth_sig_len}_{not args.no_random_peaks}.pkl'
    elif args.dataset == 'AudioMNIST':
        output_path = f'{args.output_path}/{args.dataset}_attributions_{args.labeltype}.pkl'
    output_path = output_path.replace('.pkl', f'_{args.n_samples}.pkl')
    if args.debug_mode:
        output_path = output_path.replace('.pkl', '_debug.pkl')
    if args.job_name:
        output_path = output_path.replace(f'{args.output_path}', f'{args.output_path}/{args.job_name}')
    if args.job_idx:
        output_path = output_path.replace('.pkl', f'_ID_{args.job_idx}.pkl')

    # check if attributions are already computed
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            attributions = pickle.load(f)
    else:
        attributions = {}

    # Random ID of this run. For saving samples of signals
    random_ID = args.random_ID
    print('Random_ID: ', random_ID)

    ## Compute baseline attributions
    if args.use_baselines:
        if not 'saliency' in attributions or args.debug_mode:
            # compute saliency
            print('Computing saliency')
            random_ID_dir = None
            if args.save_signals:
                # Save metadata to a txt file
                random_ID_dir = os.path.join(args.output_path, 'samples', f'{random_ID}{'_debug' if args.debug_mode else ''}', 'saliency')
                os.makedirs(random_ID_dir, exist_ok=True)
                with open(os.path.join(random_ID_dir, f'metadata_{random_ID}.txt'), 'w') as meta_file:
                    meta_file.write(f'Metadata for Saliency. ID: {random_ID}\n')
                    meta_file.write(f'Dataset: {args.dataset}\n')
                    if args.dataset == 'AudioMNIST':
                        meta_file.write(f'Label Type: {args.labeltype}\n')
                    elif args.dataset == 'synthetic':
                        meta_file.write(f'Noise Level: {args.noise_level}\n')
                        meta_file.write(f'Synthetic Signal Length: {args.synth_sig_len}\n')
                        meta_file.write(f'Add Random Peaks: {not args.no_random_peaks}\n')
                    meta_file.write(f'Num Samples: {args.n_samples}\n')
            attributions['saliency'] = compute_gradient_scores(model, test_loader, attr_method = 'gxi', device=device, save_signals_path=random_ID_dir)
            print('Saliency computed')
        if not 'lrp' in attributions or args.debug_mode:
            # compute LRP
            print('Computing LRP')
            random_ID_dir = None
            if args.save_signals:
                # Save metadata to a txt file
                random_ID_dir = os.path.join(args.output_path, 'samples', f'{random_ID}{'_debug' if args.debug_mode else ''}', 'lrp')
                os.makedirs(random_ID_dir, exist_ok=True)
                with open(os.path.join(random_ID_dir, f'metadata_{random_ID}.txt'), 'w') as meta_file:
                    meta_file.write(f'Metadata for LRP. ID: {random_ID}\n')
                    meta_file.write(f'Dataset: {args.dataset}\n')
                    if args.dataset == 'AudioMNIST':
                        meta_file.write(f'Label Type: {args.labeltype}\n')
                    elif args.dataset == 'synthetic':
                        meta_file.write(f'Noise Level: {args.noise_level}\n')
                        meta_file.write(f'Synthetic Signal Length: {args.synth_sig_len}\n')
                        meta_file.write(f'Add Random Peaks: {not args.no_random_peaks}\n')
                    meta_file.write(f'Num Samples: {args.n_samples}\n')
            attributions['lrp'] = compute_gradient_scores(model, test_loader, attr_method = 'lrp', device=device, save_signals_path=random_ID_dir)
            print('LRP computed')
        if not 'IG' in attributions or args.debug_mode:
            # compute integrated gradients
            print('Computing IG')
            random_ID_dir = None
            if args.save_signals:
                # Save metadata to a txt file
                random_ID_dir = os.path.join(args.output_path, 'samples', f'{random_ID}{'_debug' if args.debug_mode else ''}', 'ig')
                os.makedirs(random_ID_dir, exist_ok=True)
                with open(os.path.join(random_ID_dir, f'metadata_{random_ID}.txt'), 'w') as meta_file:
                    meta_file.write(f'Metadata for IG. ID: {random_ID}\n')
                    meta_file.write(f'Dataset: {args.dataset}\n')
                    if args.dataset == 'AudioMNIST':
                        meta_file.write(f'Label Type: {args.labeltype}\n')
                    elif args.dataset == 'synthetic':
                        meta_file.write(f'Noise Level: {args.noise_level}\n')
                        meta_file.write(f'Synthetic Signal Length: {args.synth_sig_len}\n')
                        meta_file.write(f'Add Random Peaks: {not args.no_random_peaks}\n')
                    meta_file.write(f'Num Samples: {args.n_samples}\n')
            attributions['IG'] = compute_gradient_scores(model, test_loader, attr_method = 'ig', device=device, save_signals_path=random_ID_dir)
            print('IG computed')
    
    
    model.to(device)
    num_batches = args.n_masks//args.batch_size
    filename_start = f'_ns_{args.n_samples}_nm_{args.n_masks}_bs_{args.batch_size}_nc_{args.num_cells}_us_{args.use_softmax}'
    
    # FreqRISE
    freqrise_filename = 'freqrise' + filename_start + f'_dropprob_{args.probability_of_drop}'
    if args.use_FreqRISE and (not freqrise_filename in attributions or args.debug_mode):
        # compute FreqRISE
        print('Creating FreqRISE')
        random_ID_dir = None
        if args.save_signals:
            # Save metadata to a txt file
            random_ID_dir = os.path.join(args.output_path, 'samples', f'{random_ID}{'_debug' if args.debug_mode else ''}', 'freqrise')
            os.makedirs(random_ID_dir, exist_ok=True)
            with open(os.path.join(random_ID_dir, f'metadata_{random_ID}.txt'), 'w') as meta_file:
                meta_file.write(f'Metadata for FreqRISE. ID: {random_ID}\n')
                meta_file.write(f'Dataset: {args.dataset}\n')
                if args.dataset == 'AudioMNIST':
                    meta_file.write(f'Label Type: {args.labeltype}\n')
                elif args.dataset == 'synthetic':
                    meta_file.write(f'Noise Level: {args.noise_level}\n')
                    meta_file.write(f'Synthetic Signal Length: {args.synth_sig_len}\n')
                    meta_file.write(f'Add Random Peaks: {not args.no_random_peaks}\n')
                meta_file.write(f'Num Samples: {args.n_samples}\n')
                meta_file.write(f'Num Masks: {args.n_masks}\n')
                meta_file.write(f'Batch Size: {args.batch_size}\n')
                meta_file.write(f'Num Cells: {args.num_cells}\n')
                meta_file.write(f'Use Softmax: {args.use_softmax}\n')
                meta_file.write(f'Probability of Drop: {args.probability_of_drop}\n')
        freqrise = FreqRISE(model, batch_size=args.batch_size, num_batches=num_batches, device=device, use_softmax=args.use_softmax, save_signals_path=random_ID_dir)
        print('Computing FreqRISE')
        attributions[freqrise_filename] = freqrise.forward_dataloader(test_loader, args.num_cells, args.probability_of_drop)
        print('FreqRISE computed')
    
    # SURL
    surl_filename = 'surl' + filename_start + f'_lr_{args.lr_S}_alpha_{args.alpha_S}_beta_{args.beta_S}_decay_{args.decay}'
    if args.use_SURL and (not surl_filename in attributions or args.debug_mode):
        # compute SURL
        print('Creating SURL')
        random_ID_dir = None
        if args.save_signals:
            # Save metadata to a txt file
            random_ID_dir = os.path.join(args.output_path, 'samples', f'{random_ID}{'_debug' if args.debug_mode else ''}', 'surl')
            os.makedirs(random_ID_dir, exist_ok=True)
            with open(os.path.join(random_ID_dir, f'metadata_{random_ID}.txt'), 'w') as meta_file:
                meta_file.write(f'Metadata for SURL. ID: {random_ID}\n')
                meta_file.write(f'Dataset: {args.dataset}\n')
                if args.dataset == 'AudioMNIST':
                    meta_file.write(f'Label Type: {args.labeltype}\n')
                elif args.dataset == 'synthetic':
                    meta_file.write(f'Noise Level: {args.noise_level}\n')
                    meta_file.write(f'Synthetic Signal Length: {args.synth_sig_len}\n')
                    meta_file.write(f'Add Random Peaks: {not args.no_random_peaks}\n')
                meta_file.write(f'Num Samples: {args.n_samples}\n')
                meta_file.write(f'Num Masks: {args.n_masks}\n')
                meta_file.write(f'Batch Size: {args.batch_size}\n')
                meta_file.write(f'Num Cells: {args.num_cells}\n')
                meta_file.write(f'Use Softmax: {args.use_softmax}\n')
                meta_file.write(f'Learning Rate: {args.lr_S}\n')
                meta_file.write(f'Alpha: {args.alpha_S}\n')
                meta_file.write(f'Beta: {args.beta_S}\n')
                meta_file.write(f'Decay: {args.decay}')
        surl = SURL(model, batch_size=args.batch_size, num_batches=num_batches, device=device, use_softmax=args.use_softmax, lr=args.lr_S, alpha=args.alpha_S, beta=args.beta_S, decay=args.decay, save_signals_path=random_ID_dir)
        print('Computing SURL')
        attributions[surl_filename] = surl.forward_dataloader(test_loader, args.num_cells)
        print('SURL computed')
        
    # FiSURL
    fisurl_filename = 'fisurl' + filename_start + f'_lr_{args.lr_F}_alpha_{args.alpha_F}_beta_{args.beta_F}_decay_{args.decay}'
    if args.use_FiSURL and (not fisurl_filename in attributions or args.debug_mode):
        # compute FiSURL
        print('Creating FiSURL')
        random_ID_dir = None
        if args.save_signals:
            # Save metadata to a txt file
            random_ID_dir = os.path.join(args.output_path, 'samples', f'{random_ID}{'_debug' if args.debug_mode else ''}', 'fisurl')
            os.makedirs(random_ID_dir, exist_ok=True)
            with open(os.path.join(random_ID_dir, f'metadata_{random_ID}.txt'), 'w') as meta_file:
                meta_file.write(f'Metadata for FiSURL. ID: {random_ID}\n')
                meta_file.write(f'Dataset: {args.dataset}\n')
                if args.dataset == 'AudioMNIST':
                    meta_file.write(f'Label Type: {args.labeltype}\n')
                elif args.dataset == 'synthetic':
                    meta_file.write(f'Noise Level: {args.noise_level}\n')
                    meta_file.write(f'Synthetic Signal Length: {args.synth_sig_len}\n')
                    meta_file.write(f'Add Random Peaks: {not args.no_random_peaks}\n')
                meta_file.write(f'Num Samples: {args.n_samples}\n')
                meta_file.write(f'Num Masks: {args.n_masks}\n')
                meta_file.write(f'Batch Size: {args.batch_size}\n')
                meta_file.write(f'Num Banks: {args.num_banks}\n')
                meta_file.write(f'Use Softmax: {args.use_softmax}\n')
                meta_file.write(f'Learning Rate: {args.lr_F}\n')
                meta_file.write(f'Alpha: {args.alpha_F}\n')
                meta_file.write(f'Beta: {args.beta_F}\n')
                meta_file.write(f'Decay: {args.decay}\n')
                meta_file.write(f'Number of Taps: {args.num_taps}\n')
                meta_file.write(f'Bandwidth: {args.bandwidth}\n')
                meta_file.write(f'Keep Ratio: {args.keep_ratio}\n')
        fisurl = FiSURL(model, num_taps=args.num_taps, num_banks=args.num_banks, fs=args.fs, bandwidth=args.bandwidth, batch_size=args.batch_size, num_batches=num_batches, keep_ratio=args.keep_ratio, device=device, use_softmax=args.use_softmax, lr=args.lr_F, alpha=args.alpha_F, beta=args.beta_F, decay=args.decay, save_signals_path=random_ID_dir)
        print('Computing FiSURL')
        attributions[fisurl_filename] = fisurl.forward_dataloader(test_loader)
        print('FiSURL computed')

    # get predictions and labels
    if not 'predictions' in attributions:
        # get predictions and labels
        predictions = []
        labels = []
        for data, target in test_loader:
            data = data.to(device)
            output = model(data.float())
            predictions.append(output.detach().cpu())
            labels.append(target)
        attributions['predictions'] = torch.cat(predictions, dim=0)
        attributions['labels'] = torch.cat(labels, dim=0)

    # save attributions
    with open(output_path, 'wb') as f:
        pickle.dump(attributions, f)



if __name__ == '__main__':
    print("Running main_attributions.py")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = 'data/', help='Path to AudioMNIST data')
    parser.add_argument('--model_path', type = str, default = 'models', help='Path to models folder')
    parser.add_argument('--output_path', type = str, default = 'outputs', help='Path to save output')
    parser.add_argument('--dataset', type = str, default = 'AudioMNIST', choices=['AudioMNIST', 'synthetic'], help='Dataset to use')
    parser.add_argument('--debug_mode', action='store_true', help='Run in debug mode. Stores outputs in deletable files')
    parser.add_argument('--job_idx', type = int, default = None, help='Job idx for hpc batch job. Used to store output seperately for parallel jobs')
    parser.add_argument('--job_name', type = str, default = None, help='Job name for hpc batch job. Used to create a folder to store seperate outputs in')
    parser.add_argument('--save_signals', action='store_true', help='Save signals when running attributions.')
    parser.add_argument('--random_ID', type = str, default = "01234567", help='Random_ID for storing samples of signals.')
    # Explanation methods
    parser.add_argument('--use_FreqRISE', action='store_true', help=f'Explain with FreqRISE.')
    parser.add_argument('--use_SURL', action='store_true', help=f'Explain with SURL.')
    parser.add_argument('--use_FiSURL', action='store_true', help=f'Explain with FiSURL.')
    # AudioMNIST
    parser.add_argument('--labeltype', type = str, default = 'digit', choices=['gender', 'digit'], help='Type of label to use for AudioMNIST')
    # Synthetic
    parser.add_argument('--noise_level', type = float, default = 0.0, help='Noise level for synthetic dataset. Either 0.8 or 0.01.')
    parser.add_argument('--synth_sig_len', type = int, default = 50, help='Length of the synthetic signals.')
    parser.add_argument('--no_random_peaks', action='store_true', help='Add random peaks to the signals')
    parser.add_argument('--seed', type = int, default = 42, help='Seed for random number generator')
    # hyperparams
    parser.add_argument('--n_samples', type = int, default = 10, help='Number of samples to compute attributions for')
    parser.add_argument('--n_masks', type = int, default = 3000, help='Number of samples to use to compute FreqRISE')
    parser.add_argument('--batch_size', type = int, default = 10, help='Batch size for masks')
    parser.add_argument('--num_cells', type = int, default = 10, help='Number of cells in mask. Should be lower than synth_sig_len if using synthetic dataset')
    parser.add_argument('--use_softmax', action='store_true', help='use softmax for FreqRISE')
    # Baselines
    parser.add_argument('--use_baselines', action='store_true', help='Run baseline models')
    # FreqRISE
    parser.add_argument('--probability_of_drop', type = float, default = 0.5, help='Probability of dropping')
    # Reinforce
    parser.add_argument('--lr_S', type = float, default = 0.1, help='Learning rate for reinforce algorithm')
    parser.add_argument('--alpha_S', type = float, default = 1.0, help='Weight of primary reward towards loss for reinforce algorithm')
    parser.add_argument('--beta_S', type = float, default = 0.01, help='Weight of mask size towards loss for reinforce algorithm')
    parser.add_argument('--decay', type = float, default = 0.9, help='weight of baseline towards loss for reinforce algorithm')
    # FiSURL
    parser.add_argument('--lr_F', type = float, default = 0.1, help='Learning rate for reinforce algorithm')
    parser.add_argument('--alpha_F', type = float, default = 1.0, help='Weight of primary reward towards loss for reinforce algorithm')
    parser.add_argument('--beta_F', type = float, default = 0.01, help='Weight of mask size towards loss for reinforce algorithm')
    parser.add_argument('--num_banks', type = int, default = 128, help='Number of banks to use for FiSURL')
    parser.add_argument('--num_taps', type = int, default = 501, help='Number of taps to use for FiSURL')
    parser.add_argument('--fs', type = int, default = 8000, help='Sampling frequency to use for FiSURL')
    parser.add_argument('--bandwidth', type = float, default = None, help='Bandwidth to use for FiSURL')
    parser.add_argument('--keep_ratio', type = float, default = 0.05, help='Ratio parameter below which the sparsity constraint is ignored for FiSURL')
    
    args = parser.parse_args()    
    
    if args.job_idx and args.job_name:
        jobarray_vals = [
            {'u_FR':True,'u_S':True,'u_FS':True,'nm':10000,'bs':250,'nc':128,'us':False,'ub':True,'pd':0.5,'lr_S':0.1,'a_S':1.00,'b_S':0.01,'d':0.9,'lr_F':0.1,'a_F':1.00,'b_F':0.01,'nb':128,'nt':501,'bw':None,'kr':0.05},
            {'u_FR':True,'u_S':True,'u_FS':True,'nm':10000,'bs':250,'nc':128,'us':False,'ub':True,'pd':0.5,'lr_S':0.2,'a_S':1.00,'b_S':0.01,'d':0.9,'lr_F':0.2,'a_F':1.00,'b_F':0.01,'nb':128,'nt':501,'bw':None,'kr':0.05},
            {'u_FR':True,'u_S':True,'u_FS':True,'nm':10000,'bs':250,'nc':128,'us':False,'ub':True,'pd':0.5,'lr_S':0.5,'a_S':1.00,'b_S':0.01,'d':0.9,'lr_F':0.5,'a_F':1.00,'b_F':0.01,'nb':128,'nt':501,'bw':None,'kr':0.05},
            {'u_FR':True,'u_S':True,'u_FS':True,'nm':10000,'bs':250,'nc':128,'us':False,'ub':True,'pd':0.5,'lr_S':1.0,'a_S':1.00,'b_S':0.01,'d':0.9,'lr_F':1.0,'a_F':1.00,'b_F':0.01,'nb':128,'nt':501,'bw':None,'kr':0.05},
            {'u_FR':True,'u_S':True,'u_FS':True,'nm':10000,'bs':250,'nc':128,'us':False,'ub':True,'pd':0.5,'lr_S':0.8,'a_S':1.00,'b_S':0.01,'d':0.9,'lr_F':0.8,'a_F':1.00,'b_F':0.01,'nb':128,'nt':501,'bw':None,'kr':0.05},
            {'u_FR':True,'u_S':True,'u_FS':True,'nm':10000,'bs':250,'nc':128,'us':False,'ub':True,'pd':0.5,'lr_S':0.5,'a_S':1.00,'b_S':0.01,'d':0.9,'lr_F':0.5,'a_F':1.00,'b_F':0.01,'nb':128,'nt':501,'bw':None,'kr':0.05},
            {'u_FR':True,'u_S':True,'u_FS':True,'nm':10000,'bs':250,'nc':128,'us':False,'ub':True,'pd':0.5,'lr_S':0.5,'a_S':1.00,'b_S':1.00,'d':0.9,'lr_F':0.5,'a_F':1.00,'b_F':1.00,'nb':128,'nt':501,'bw':None,'kr':0.05},
            {'u_FR':True,'u_S':True,'u_FS':True,'nm':10000,'bs':250,'nc':128,'us':False,'ub':True,'pd':0.5,'lr_S':0.5,'a_S':1.00,'b_S':0.10,'d':0.9,'lr_F':0.5,'a_F':1.00,'b_F':0.10,'nb':128,'nt':501,'bw':None,'kr':0.05},
            {'u_FR':True,'u_S':True,'u_FS':True,'nm':10000,'bs':250,'nc':128,'us':False,'ub':True,'pd':0.5,'lr_S':0.5,'a_S':10.0,'b_S':0.01,'d':0.9,'lr_F':0.5,'a_F':10.0,'b_F':0.01,'nb':128,'nt':501,'bw':None,'kr':0.05},
            {'u_FR':True,'u_S':True,'u_FS':True,'nm':10000,'bs':250,'nc':128,'us':False,'ub':True,'pd':0.5,'lr_S':0.5,'a_S':1.00,'b_S':10.0,'d':0.9,'lr_F':0.5,'a_F':1.00,'b_F':10.0,'nb':128,'nt':501,'bw':None,'kr':0.05},
            {'u_FR':True,'u_S':True,'u_FS':True,'nm':10000,'bs':250,'nc':128,'us':False,'ub':True,'pd':0.5,'lr_S':0.5,'a_S':1.00,'b_S':0.50,'d':0.9,'lr_F':0.5,'a_F':1.00,'b_F':0.50,'nb':128,'nt':501,'bw':None,'kr':0.05}
        ]
        
        job_vals = jobarray_vals[args.job_idx]
        args.use_FreqRISE =         job_vals['u_FR']
        args.use_SURL =             job_vals['u_S']
        args.use_FiSURL =           job_vals['u_FS']
        args.n_masks =              job_vals['nm']
        args.batch_size =           job_vals['bs']
        args.num_cells =            job_vals['nc']
        args.use_softmax =          job_vals['us']
        args.use_baselines =        job_vals['ub']
        args.probability_of_drop =  job_vals['pd']
        args.lr_S =                 job_vals['lr_S']
        args.alpha_S =              job_vals['a_S']
        args.beta_S =               job_vals['b_S']
        args.decay =                job_vals['d']
        args.lr_F =                 job_vals['lr_F']
        args.alpha_F =              job_vals['a_F']
        args.beta_F =               job_vals['b_F']
        args.num_banks =            job_vals['nb']
        args.num_taps =             job_vals['nt']
        args.fs =                   8000 if args.dataset == "AudioMNIST" else args.synth_sig_len
        args.bandwidth =            job_vals['bw']
        args.keep_ratio =           job_vals['kr']
        
    if args.dataset == 'synthetic' and args.use_FiSURL:
        assert args.synth_sig_len == args.fs, "Synthetic signal length should be equal to fs if using FiSURL, as that would always be the case for 1 second signals"
    
    if args.dataset == 'AudioMNIST' and args.use_FiSURL:
        assert 8000 == args.fs, "AudioMNIST signal length should be equal to fs if using FiSURL, as that would always be the case for 1 second signals"
    
    if args.dataset == 'synthetic' and (args.use_FreqRISE or args.use_SURL or args.use_FiSURL):
        assert args.synth_sig_len > args.num_cells, "Number of cells should be lower than synth_sig_len if using synthetic dataset"
        
    main(args)