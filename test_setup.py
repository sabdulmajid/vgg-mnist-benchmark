#!/usr/bin/env python3
"""Quick test to verify all components are working."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from train import VGG11
import sys

def test_model():
    print("Testing VGG11 model...")
    model = VGG11()
    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    assert y.shape == (2, 10), f"Expected shape (2, 10), got {y.shape}"
    print(f"✓ Model works ({sum(p.numel() for p in model.parameters()):,} params)")

def test_data_loading():
    print("\nTesting data loading...")
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Use small subset for testing
    train_subset = Subset(train_dataset, range(128))
    test_subset = Subset(test_dataset, range(128))
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    batch = next(iter(train_loader))
    assert batch[0].shape == (32, 1, 32, 32), f"Expected shape (32, 1, 32, 32), got {batch[0].shape}"
    print(f"✓ Data loading works")
    
    return train_loader, test_loader

def test_training_step():
    print("\nTesting training step...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    model = VGG11().to(device)
    train_loader, test_loader = test_data_loading()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train one batch
    model.train()
    inputs, labels = next(iter(train_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step works (loss: {loss.item():.4f})")
    
    # Test one batch
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(test_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        acc = predicted.eq(labels).sum().item() / labels.size(0) * 100
    
    print(f"✓ Evaluation works (acc: {acc:.2f}%)")

def main():
    print("=" * 60)
    print("VGG11 MNIST Setup Verification")
    print("=" * 60)
    
    try:
        test_model()
        test_data_loading()
        test_training_step()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! Ready to run experiments.")
        print("=" * 60)
        print("\nTo run full training:")
        print("  source venv/bin/activate")
        print("  bash run_all.sh")
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
