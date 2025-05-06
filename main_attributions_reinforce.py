import torch
import argparse
from src.explainability.freqrise_reinforce import FreqRISE_Reinforce
from src.explainability.evaluation import compute_gradient_scores
from src.data.load_data import load_data
from src.models.load_model import load_model
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

    ## Compute all attributions
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
    num_batches_freqrise = args.freqrise_samples//args.batch_size
    freqrise_file_output = f'freqrise_sm_{args.use_softmax}_batchsize_{args.batch_size}_numbatches_{num_batches_freqrise}_R_{args.reward_fn}_lr_{args.lr}_alpha_{args.alpha}_beta_{args.beta}_decay_{args.decay}_numcells_{args.num_cells}_reinforce'
    if not freqrise_file_output in attributions:
        # compute FreqRISE
        print('Creating FreqRISE')
        freqrise = FreqRISE_Reinforce(model, batch_size=args.batch_size, num_batches=num_batches_freqrise, device=device, domain=args.explanation_domain, use_softmax=args.use_softmax, lr=args.lr, alpha=args.alpha, beta=args.beta, decay=args.decay, reward_fn=args.reward_fn)
        print('Computing FreqRISE')
        attributions[freqrise_file_output] = freqrise.forward_dataloader(test_loader, args.num_cells)
        print('FreqRISE computed')

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

    with open(output_path, 'wb') as f:
        pickle.dump(attributions, f)
    
    return None



if __name__ == '__main__':
    print("Running main_attributions.py")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = 'data/', help='Path to AudioMNIST data')
    parser.add_argument('--model_path', type = str, default = 'models', help='Path to models folder')

    parser.add_argument('--dataset', type = str, default = 'AudioMNIST', help='Dataset to use')
    parser.add_argument('--labeltype', type = str, default = 'digit', help='Type of label to use for AudioMNIST')
    parser.add_argument('--noise_level', type = float, default = 0.5, help='Noise level for synthetic dataset. Either 0.8 or 0.01.')
    
    parser.add_argument('--explanation_domain', type = str, default = 'fft', help='Domain of explanation')
    parser.add_argument('--num_cells', type = int, default = 200, help='Number of cells in mask')
    parser.add_argument('--n_samples', type = int, default = 10, help='Number of samples to compute attributions for')
    parser.add_argument('--freqrise_samples', type = int, default = 3000, help='Number of samples to use to compute FreqRISE')
    parser.add_argument('--lrp_window', type = int, default = 800, help='Window size for LRP')
    parser.add_argument('--lrp_hop', type = int, default = 800, help='Hop size for LRP')
    
    parser.add_argument('--lr', type = float, default = 1e-4, help='Learning rate for reinforce algorithm')
    parser.add_argument('--alpha', type = float, default = 1.0, help='Weight of primary reward towards loss for reinforce algorithm')
    parser.add_argument('--beta', type = float, default = 0.01, help='Weight of mask size towards loss for reinforce algorithm')
    parser.add_argument('--decay', type = float, default = 0.9, help='weight of baseline towards loss for reinforce algorithm')
    parser.add_argument('--reward_fn', type = str, default = "pred", help='reward function to maximize for reinforce algorithm')
    parser.add_argument('--use_softmax', type = str, default = "False", help='use softmax for FreqRISE')
    parser.add_argument('--batch_size', type = int, default = 10, help='Batch size for FreqRISE')
    
    parser.add_argument('--output_path', type = str, default = 'outputs', help='Path to save output')
    args = parser.parse_args()
    print("Use_Softmax: ", args.use_softmax)
    args.use_softmax = args.use_softmax == 'True'
    print("Bool(Use_Softmax): ", args.use_softmax)
    main(args)