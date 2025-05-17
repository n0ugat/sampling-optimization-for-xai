import torch
import argparse
from src.explainability.fisurl import FiSURL
from src.explainability.evaluation import compute_gradient_scores
from src.data.load_data import load_data
from src.models.load_model import load_model
from src.utils.filterbank import FilterBank
import pickle
import os

def main(args):
    print('Loading data')
    test_loader = load_data(args)
    print('Loading model')
    model = load_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if args.dataset == 'synthetic':
        output_path = f'{args.output_path}/{args.dataset}_{args.noise_level}_attributions_{args.explanation_domain}_{args.n_samples}.pkl'
    elif args.dataset == 'AudioMNIST':
        output_path = f'{args.output_path}/{args.dataset}_{args.labeltype}_attributions_{args.explanation_domain}_{args.n_samples}.pkl'
    # check if attributions are already computed
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            attributions = pickle.load(f)
    else:
        attributions = {}
    # Compute all attributions
    lrp_stft_args = {'n_fft': args.lrp_window, 'hop_length': args.lrp_hop, 'center': False}
    if not 'saliency' in attributions:
        # compute saliency
        print('Computing saliency')
        attributions['saliency'] = compute_gradient_scores(model, test_loader, attr_method = 'gxi', domain = args.explanation_domain, stft_params = lrp_stft_args)
        print('Saliency computed')
    if not 'lrp' in attributions:
        # compute LRP
        print('Computing LRP')
        attributions['lrp'] = compute_gradient_scores(model, test_loader, attr_method = 'lrp', domain = args.explanation_domain, stft_params = lrp_stft_args)
        print('LRP computed')
    if not 'IG' in attributions:
        # compute integrated gradients
        print('Computing IG')
        attributions['IG'] = compute_gradient_scores(model, test_loader, attr_method = 'ig', domain = args.explanation_domain, stft_params = lrp_stft_args)
        print('IG computed')
    model.to(device)
    # Create FiSURL
    rl_params = {'lr': 0.05, 'alpha': 1.00, 'beta': 1.00, 'decay': 0.9}
    fisurl = FiSURL(model, num_taps=args.num_taps, num_banks=args.num_banks, fs=args.fs, bandwidth=args.bandwidth, batch_size=50, num_batches=args.fisurl_samples//10, keep_ratio=args.keep_ratio, device=device, use_softmax=args.use_softmax, use_rl=args.use_rl, rl_params=rl_params)
    # num_batches=args.fisurl_samples//50
    # rl_params={'lr': 1e-4, 'alpha': 1.00, 'beta': 0.01, 'decay': 0.9, 'reward_fn': 'pred'}
    # rl_params={'lr': 0.05, 'alpha': 1.00, 'beta': 0.05, 'decay': 0.9, 'reward_fn': 'pred'}
    # Compute FiSURL
    if not f'fisurl_{args.num_banks}_{args.fisurl_samples}_dropprob_{args.probability_of_drop}' in attributions or not f'fisurl_{args.num_banks}_{args.fisurl_samples}_rl_{rl_params['lr']}_{rl_params['alpha']}_{rl_params['beta']}_{rl_params['decay']}' in attributions:
        if args.use_rl:
            print('Computing FiSURL')
            attributions[f'fisurl_{args.num_banks}_{args.fisurl_samples}_rl_{rl_params['lr']}_{rl_params['alpha']}_{rl_params['beta']}_{rl_params['decay']}'] = fisurl.forward_dataloader(test_loader, args.num_banks, args.probability_of_drop)
            print('FiSURL computed')
        else:
            print('Computing FiSURL with Random Sampling')
            attributions[f'fisurl_{args.num_cells}_{args.fisurl_samples}_dropprob_{args.probability_of_drop}'] = fisurl.forward_dataloader(test_loader, args.num_banks, args.probability_of_drop)
            print('FiSURL computed')
    
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

    # Save attributions
    with open(output_path, 'wb') as f:
        pickle.dump(attributions, f)
    print(f'Attributions saved to {output_path}')

if __name__ == '__main__':
    print("Running main_fisurl.py")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AudioMNIST', help='Dataset to use') # synthetic or AudioMNIST
    parser.add_argument('--data_path', type = str, default = 'data/', help='Path to AudioMNIST data')
    parser.add_argument('--model_path', type = str, default = 'models', help='Path to models folder')

    parser.add_argument('--labeltype', type=str, default='gender', help='Label type to use')
    parser.add_argument('--explanation_domain', type=str, default='fft', help='Domain to use for explanation')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to use')
    parser.add_argument('--output_path', type=str, default='outputs', help='Path to save the outputs')
    # parser.add_argument('--noise_level', type=float, default=0.01, help='Noise level to use')
    parser.add_argument('--noise_level', type = float, default = 0.5, help='Noise level for synthetic dataset.')

    parser.add_argument('--num_cells', type=int, default=200, help='Number of cells to use') #10
    parser.add_argument('--num_banks', type=int, default=128, help='Number of banks to use')
    parser.add_argument('--keep_ratio', type=float, default=0.05, help='Ratio parameter below which the sparsity constraint is ignored')
    parser.add_argument('--num_taps', type=int, default=501, help='Number of taps to use')
    parser.add_argument('--fs', type=int, default=8000, help='Sampling frequency to use')
    parser.add_argument('--bandwidth', type=float, default=None, help='Bandwidth to use')
    parser.add_argument('--fisurl_samples', type=int, default=3000, help='Number of samples to use for FiSURL')
    parser.add_argument('--probability_of_drop', type=float, default=0.5, help='Probability of dropping a filterbank')
    parser.add_argument('--use_softmax', action='store_true', help='Use softmax for predictions')
    parser.add_argument('--use_rl', action='store_true', help='Use reinforcement learning for FiSURL')
    parser.add_argument('--lrp_window', type=int, default=800, help='Window size for LRP') # 455
    parser.add_argument('--lrp_hop', type=int, default=800, help='Hop size for LRP') # 455-420
    args = parser.parse_args()

    main(args)
    print("Finished running main_fisurl.py")