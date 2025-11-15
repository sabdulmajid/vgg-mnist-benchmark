import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from pathlib import Path
from train import VGG11

def test_with_transform(model, transform, device, description):
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'{description}: {accuracy:.2f}%')
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    model = VGG11(num_classes=10).to(device)
    model.load_state_dict(torch.load('results/vgg11_mnist.pth', map_location=device))
    
    results = {}
    
    # Baseline (no transformation)
    print('Testing generalization...\n')
    base_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    baseline_acc = test_with_transform(model, base_transform, device, 'Baseline (no augmentation)')
    results['baseline'] = baseline_acc
    
    # Horizontal flip
    print('\n--- Flip Tests ---')
    hflip_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    hflip_acc = test_with_transform(model, hflip_transform, device, 'Horizontal flip')
    results['horizontal_flip'] = hflip_acc
    
    # Vertical flip
    vflip_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    vflip_acc = test_with_transform(model, vflip_transform, device, 'Vertical flip')
    results['vertical_flip'] = vflip_acc
    
    # Gaussian noise tests
    print('\n--- Gaussian Noise Tests ---')
    noise_variances = [0.01, 0.1, 1.0]
    results['gaussian_noise'] = {}
    
    for var in noise_variances:
        noise_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x + (var ** 0.5) * torch.randn_like(x))
        ])
        noise_acc = test_with_transform(model, noise_transform, device, f'Gaussian noise (var={var})')
        results['gaussian_noise'][str(var)] = noise_acc
    
    # Save results
    with open('results/generalization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n--- Analysis ---')
    print('Flip effect: The model is not invariant to flips because MNIST digits have')
    print('specific orientations that change meaning when flipped (e.g., 6 becomes 9).')
    print('\nNoise effect: Higher noise variance progressively degrades accuracy as')
    print('it obscures the digit features the model learned during training.')
    
    with open('results/generalization_analysis.txt', 'w') as f:
        f.write('Generalization Analysis\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Baseline accuracy: {baseline_acc:.2f}%\n\n')
        f.write('Flip Tests:\n')
        f.write(f'  Horizontal flip: {hflip_acc:.2f}%\n')
        f.write(f'  Vertical flip: {vflip_acc:.2f}%\n')
        f.write('  Effect: Flips reduce accuracy because MNIST digits have orientation-dependent meanings.\n\n')
        f.write('Gaussian Noise Tests:\n')
        for var in noise_variances:
            f.write(f'  Variance {var}: {results["gaussian_noise"][str(var)]:.2f}%\n')
        f.write('  Effect: Increasing noise variance progressively degrades accuracy by obscuring learned features.\n')
    
    print('\nResults saved to results/generalization_results.json')
    print('Analysis saved to results/generalization_analysis.txt')

if __name__ == '__main__':
    main()
