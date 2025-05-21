from src.data import load_data
from src.models import load_model
import torch
import argparse
from tqdm import tqdm

# Load data and model
def main(args):
    dataloader = load_data(args)
    model = load_model(args)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(torch.float32)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AudioMNIST Inference')
    parser.add_argument('--dataset', type=str, default='AudioMNIST', help='Dataset to use (default: AudioMNIST)')
    parser.add_argument('--model_path', type=str, default='models', help='Path to the model directory (default: models)')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to the data directory (default: data/)')
    parser.add_argument('--labeltype', type=str, default='gender', help='Label type (default: digit)')
    parser.add_argument('--n_samples', type=int, default=30000, help='Number of samples to load (default: 1000)')
    args = parser.parse_args()
    main(args)