import argparse
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data import synthetic_dataset_generator
from src.models import LinearModel

def train_synthetic(n_samples, noise_level, synth_sig_len, epochs=100, model_path="models", add_random_peaks=True, seed=42):
    print('Generating dataset')
    signals, labels = synthetic_dataset_generator(n_samples, length=synth_sig_len, noiselevel=noise_level, add_random_peaks=add_random_peaks, seed=seed)
    print('Dataset generated')
    print('Creating dataloaders')
    signals, labels = torch.tensor(signals).float(), torch.tensor(labels).long()
    split = int(n_samples * 0.8)
    val_signals, val_labels = signals[split:], labels[split:]    
    
    dloader = DataLoader(TensorDataset(signals, labels), batch_size=64, shuffle=True)
    val_dloader = DataLoader(TensorDataset(val_signals, val_labels), batch_size=64, shuffle=True)
    print('Dataloaders created')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = LinearModel(input_size=synth_sig_len, hidden_layers=2, hidden_size=128, output_size=8).to(device)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = epochs
    collect_train_accuracy = []
    collect_train_loss = []
    collect_val_accuracy = []
    collect_val_loss = []
    print('Training model...')
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0
        for batch in dloader:
            optimizer.zero_grad()
            output = model(batch[0].to(device))
            batch_onehot = torch.nn.functional.one_hot(batch[1], num_classes=8).float().squeeze()
            _, predicted = torch.max(output, 1)
            total += batch[1].size(0)
            correct += (predicted == batch[1].squeeze().to(device)).sum().item()
            loss = loss_fn(output, batch_onehot.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_train_loss = epoch_loss/len(dloader)
        collect_train_loss.append(epoch_train_loss)
        collect_train_accuracy.append(100 * correct / total)
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            epoch_loss = 0
            for batch in val_dloader:
                output = model(batch[0].to(device))
                batch_onehot = torch.nn.functional.one_hot(batch[1], num_classes=8).float().squeeze()
                loss = loss_fn(output, batch_onehot.to(device))
                _, predicted = torch.max(output, 1)
                total += batch[1].size(0)
                correct += (predicted == batch[1].squeeze().to(device)).sum().item()
                epoch_loss += loss.item()
        epoch_val_loss = epoch_loss/len(val_dloader)
        collect_val_loss.append(epoch_val_loss)
        collect_val_accuracy.append(100 * correct / total)
        
        print(f"Epoch {epoch} train Loss: {epoch_train_loss}", end=", ")
        print(f"Epoch {epoch} val Loss: {epoch_val_loss}", end=", ")
        print(f"Train Accuracy: {collect_train_accuracy[-1]}", end=", ")
        print(f"Validation Accuracy: {collect_val_accuracy[-1]}")
    
    print('Model trained')
    
    output_path = f'outputs/figures/synthetic_training/ns_{n_samples}_nl_{noise_level}_ssl_{synth_sig_len}_epochs_{epochs}_adp_{add_random_peaks}_seed_{seed}'
    os.makedirs(output_path, exist_ok=True)
    
    # Plotting both Loss and Accuracy in one horizontal subplot
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    # Loss subplot
    axs[0].plot(collect_train_loss, label='Train Loss')
    axs[0].plot(collect_val_loss, label='Validation Loss')
    axs[0].set_xlabel('Epochs', fontsize=12)
    axs[0].set_ylabel('Loss', fontsize=12)
    axs[0].set_title('Loss', fontsize=15)
    axs[0].legend(fontsize=10)

    # Accuracy subplot
    axs[1].plot(collect_train_accuracy, label='Train Accuracy')
    axs[1].plot(collect_val_accuracy, label='Validation Accuracy')
    axs[1].set_xlabel('Epochs', fontsize=12)
    axs[1].set_ylabel('Accuracy', fontsize=12)
    axs[1].set_title('Accuracy', fontsize=15)
    axs[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'training_plots.png'))
    plt.close()
    
    # Save the model
    torch.save(model.state_dict(), f'{model_path}/synthetic_{noise_level}_{synth_sig_len}_{add_random_peaks}.pt')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = 'models', help='Path to save model')
    parser.add_argument('--n_samples', type = int, default = 1000, help='Number of samples to generate')
    parser.add_argument('--noise_level', type = float, default = 0.0, help='Noise in dataset')
    parser.add_argument('--synth_sig_len', type = int, default = 100, help='Length of the synthetic signals.')
    parser.add_argument('--no_random_peaks', action='store_true', help='Add random peaks to the signals')
    parser.add_argument('--seed', type = int, default = 42, help='Seed for random number generator')
    parser.add_argument('--epochs', type = int, default = 100, help='Number of epochs to train for')

    args = parser.parse_args()
    train_synthetic(args.n_samples, args.noise_level, args.synth_sig_len, args.epochs, args.model_path, not args.no_random_peaks, args.seed)