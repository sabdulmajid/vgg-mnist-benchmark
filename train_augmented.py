import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from pathlib import Path
from train import VGG11, train_epoch, evaluate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data augmentation transforms
    # Using random rotation, slight translation, and scaling for robustness
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    model = VGG11(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    num_epochs = 5
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    print(f'\nTraining with data augmentation for {num_epochs} epochs...')
    print('Augmentations: random rotation (±10°), random translation (±10%)\n')
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%')
    
    # Save augmented model
    torch.save(model.state_dict(), 'results/vgg11_mnist_augmented.pth')
    with open('results/history_augmented.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print('\nAugmented model saved to results/vgg11_mnist_augmented.pth')
    
    # Test generalization with augmented model
    print('\n--- Testing augmented model generalization ---')
    from test_generalization import test_with_transform
    
    results = {}
    
    base_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    baseline_acc = test_with_transform(model, base_transform, device, 'Baseline')
    results['baseline'] = baseline_acc
    
    # Test with noise
    print('\nGaussian Noise Tests:')
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
    
    with open('results/generalization_results_augmented.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\nAugmented model results saved to results/generalization_results_augmented.json')

if __name__ == '__main__':
    main()
