import torch
import os

from src.models import AudioNet, LinearModel
from train_synthetic import train_synthetic

def load_model(args):
    if args.dataset == 'AudioMNIST':
        model_path = f'{args.model_path}/AudioNet_{args.labeltype}.pt'
        if args.labeltype == 'gender':
            model = AudioNet(input_shape=(1, 8000), num_classes=2).eval()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
        else:
            model = AudioNet(input_shape=(1, 8000), num_classes=10).eval()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
    elif args.dataset == 'synthetic':
        model_path = f'{args.model_path}/synthetic_{args.noise_level}_{args.synth_sig_len}.pt'
        if not os.path.exists(model_path):
            print("Synthetic model not found, training a new one...")
            train_synthetic(1000, args.noise_level, args.synth_sig_len, args.model_path)
        model = LinearModel(input_size=args.synth_sig_len, hidden_layers=2, hidden_size=128, output_size=8)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
    model.eval()
    return model