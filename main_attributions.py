import torch
import argparse
import pickle
import os

from src.explainability import FreqRISE, SURL, compute_gradient_scores
from src.data import load_data
from src.models import load_model

def main(args):
    print('Loading data')
    test_loader = load_data(args)
    print('Loading model')
    model = load_model(args)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu'
    print(f'Using device: {device}')
    
    if args.dataset == 'synthetic':
        output_path = f'{args.output_path}/{args.dataset}_attributions_{args.noise_level}_{args.synth_sig_len}.pkl'
    elif args.dataset == 'AudioMNIST':
        output_path = f'{args.output_path}/{args.dataset}_attributions_{args.labeltype}.pkl'
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

    ## Compute baseline attributions
    lrp_stft_args = {'n_fft': args.lrp_window, 'hop_length': args.lrp_hop, 'center': False}
    if args.use_baselines:
        if not 'saliency' in attributions:
            # compute saliency
            print('Computing saliency')
            attributions['saliency'] = compute_gradient_scores(model, test_loader, attr_method = 'gxi', domain = 'fft', stft_params = lrp_stft_args)
            print('Saliency computed')
        if not 'lrp' in attributions:
            # compute LRP
            print('Computing LRP')
            attributions['lrp'] = compute_gradient_scores(model, test_loader, attr_method = 'lrp', domain = 'fft', stft_params = lrp_stft_args)
            print('LRP computed')
        if not 'IG' in attributions:
            # compute integrated gradients
            print('Computing IG')
            attributions['IG'] = compute_gradient_scores(model, test_loader, attr_method = 'ig', domain = 'fft', stft_params = lrp_stft_args)
            print('IG computed')
    
    
    model.to(device)
    num_batches = args.n_masks//args.batch_size
    filename_start = f'_ns_{args.n_samples}_nm_{args.n_masks}_bs_{args.batch_size}_nc_{args.num_cells}_us_{args.use_softmax}'
    
    # FreqRISE
    freqrise_filename = 'freqrise' + filename_start + f'_dropprob_{args.probability_of_drop}'
    if args.use_FreqRISE and not freqrise_filename in attributions:
        # compute FreqRISE
        print('Creating FreqRISE')
        freqrise = FreqRISE(model, batch_size=args.batch_size, num_batches=num_batches, device=device, use_softmax=args.use_softmax)
        print('Computing FreqRISE')
        attributions[freqrise_filename] = freqrise.forward_dataloader(test_loader, args.num_cells, args.probability_of_drop)
        print('FreqRISE computed')
    
    # SURL
    surl_filename = 'surl' + filename_start + f'_lr_{args.lr}_alpha_{args.alpha}_beta_{args.beta}_decay_{args.decay}'
    if args.use_SURL and not surl_filename in attributions:
        # compute FreqRISE
        print('Creating SURL')
        freqrise = SURL(model, batch_size=args.batch_size, num_batches=num_batches, device=device, use_softmax=args.use_softmax, lr=args.lr, alpha=args.alpha, beta=args.beta, decay=args.decay, dataset=args.dataset)
        print('Computing SURL')
        attributions[surl_filename] = freqrise.forward_dataloader(test_loader, args.num_cells)
        print('SURL computed')
        
    # FiSURL
    fisurl_filename = 'fisurl' + filename_start + f'_lr_{args.lr}_alpha_{args.alpha}_beta_{args.beta}_decay_{args.decay}'
    if args.use_FiSURL and not fisurl_filename in attributions:
        print('FiSURL not implemented yet')

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
    parser.add_argument('--dataset', type = str, default = 'AudioMNIST', help='Dataset to use')
    parser.add_argument('--debug_mode', action='store_true', help='Run in debug mode. Stores outputs in deletable .pkl file')
    parser.add_argument('--job_idx', type = int, default = None, help='Job idx for hpc batch job. Used to store output seperately for parallel jobs')
    parser.add_argument('--job_name', type = str, default = None, help='Job name for hpc batch job. Used to create a folder to store seperate outputs in')
    # Explanation methods
    parser.add_argument('--use_FreqRISE', action='store_true', help=f'Explain with FreqRISE.')
    parser.add_argument('--use_SURL', action='store_true', help=f'Explain with SURL.')
    parser.add_argument('--use_FiSURL', action='store_true', help=f'Explain with FiSURL.')
    # AudioMNIST
    parser.add_argument('--labeltype', type = str, default = 'digit', help='Type of label to use for AudioMNIST')
    # Synthetic
    parser.add_argument('--noise_level', type = float, default = 0.5, help='Noise level for synthetic dataset. Either 0.8 or 0.01.')
    parser.add_argument('--synth_sig_len', type = int, default = 50, help='Length of the synthetic signals.')
    # hyperparams
    parser.add_argument('--n_samples', type = int, default = 10, help='Number of samples to compute attributions for')
    parser.add_argument('--n_masks', type = int, default = 3000, help='Number of samples to use to compute FreqRISE')
    parser.add_argument('--batch_size', type = int, default = 10, help='Batch size for masks')
    parser.add_argument('--num_cells', type = int, default = 10, help='Number of cells in mask. Should be lower than synth_sig_len if using synthetic dataset')
    parser.add_argument('--use_softmax', action='store_true', help='use softmax for FreqRISE')
    # Baselines
    parser.add_argument('--use_baselines', action='store_true', help='Run baseline models')
    parser.add_argument('--lrp_window', type = int, default = 800, help='Window size for LRP')
    parser.add_argument('--lrp_hop', type = int, default = 800, help='Hop size for LRP')
    # FreqRISE
    parser.add_argument('--probability_of_drop', type = float, default = 0.5, help='Probability of dropping')
    # Reinforce
    parser.add_argument('--lr', type = float, default = 0.1, help='Learning rate for reinforce algorithm')
    parser.add_argument('--alpha', type = float, default = 1.0, help='Weight of primary reward towards loss for reinforce algorithm')
    parser.add_argument('--beta', type = float, default = 0.01, help='Weight of mask size towards loss for reinforce algorithm')
    parser.add_argument('--decay', type = float, default = 0.9, help='weight of baseline towards loss for reinforce algorithm')
    
    args = parser.parse_args()    
    
    if args.job_idx and args.job_name:
        jobarray_vals = [
            {'u_FR':False,'u_S':False,'u_FS':False,'ns':10,'nm':150,'bs':10,'nc':10,'us':False,'ub':True,'lw':800,'lh':800,'pd':0.5,'lr':0.1,'a':1.0,'b':0.01,'d':0.9},
            {'u_FR':False,'u_S':False,'u_FS':False,'ns':10,'nm':150,'bs':10,'nc':10,'us':False,'ub':True,'lw':800,'lh':800,'pd':0.5,'lr':0.1,'a':1.0,'b':0.01,'d':0.9},
            {'u_FR':True,'u_S':False,'u_FS':False,'ns':10,'nm':150,'bs':10,'nc':10,'us':False,'ub':False,'lw':800,'lh':800,'pd':0.5,'lr':0.1,'a':1.0,'b':0.01,'d':0.9},
            {'u_FR':True,'u_S':True,'u_FS':True,'ns':10,'nm':150,'bs':10,'nc':10,'us':False,'ub':False,'lw':800,'lh':800,'pd':0.5,'lr':0.1,'a':1.0,'b':0.01,'d':0.9},
            {'u_FR':False,'u_S':True,'u_FS':False,'ns':10,'nm':150,'bs':10,'nc':10,'us':False,'ub':False,'lw':800,'lh':800,'pd':0.5,'lr':0.1,'a':1.0,'b':0.01,'d':0.9}
        ]
        
        job_vals = jobarray_vals[args.job_idx]
        args.use_FreqRISE =         job_vals['u_FR']
        args.use_SURL =             job_vals['u_S']
        args.use_FiSURL =           job_vals['u_FS']
        args.n_samples =            job_vals['ns']
        args.n_masks =              job_vals['nm']
        args.batch_size =           job_vals['bs']
        args.num_cells =            job_vals['nc']
        args.use_softmax =          job_vals['us']
        args.use_baselines =        job_vals['ub']
        args.lrp_window =           job_vals['lw']
        args.lrp_hop =              job_vals['lh']
        args.probability_of_drop =  job_vals['pd']
        args.lr =                   job_vals['lr']
        args.alpha =                job_vals['a']
        args.beta =                 job_vals['b']
        args.decay =                job_vals['d']
    
    if args.dataset == 'synthetic' and (args.use_FreqRISE or args.use_SURL or args.use_FiSURL):
        assert args.synth_sig_len > args.num_cells, "Number of cells should be lower than synth_sig_len if using synthetic dataset"
    main(args)