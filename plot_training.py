import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_history():
    with open('results/history.json', 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training accuracy
    ax1.plot(epochs, history['train_acc'], 'b-', marker='o', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # Test accuracy
    ax2.plot(epochs, history['test_acc'], 'r-', marker='o', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # Training loss
    ax3.plot(epochs, history['train_loss'], 'b-', marker='o', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss')
    ax3.grid(True, alpha=0.3)
    
    # Test loss
    ax4.plot(epochs, history['test_loss'], 'r-', marker='o', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Test Loss')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_plots.png', dpi=300, bbox_inches='tight')
    print('Training plots saved to results/training_plots.png')
    plt.close()

if __name__ == '__main__':
    plot_training_history()
