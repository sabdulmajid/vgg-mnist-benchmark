#!/usr/bin/env python3
"""
VGG11 MNIST Benchmark - Complete Pipeline (script made for Windows PC + NVIDIA GPU)
Runs all experiments end-to-end: training, plotting, generalization tests, and augmented training.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 70)
    print(f"{description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return False

def main():
    print("=" * 70)
    print("VGG11 MNIST BENCHMARK - COMPLETE PIPELINE")
    print("=" * 70)
    print("\nThis will run all experiments:")
    print("  1. Baseline training (5 epochs)")
    print("  2. Generate training plots")
    print("  3. Test generalization (flips and noise)")
    print("  4. Train with data augmentation (5 epochs)")
    print("\nEstimated time: 10-15 minutes with GPU")
    print("=" * 70)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    experiments = [
        ('train.py', 'Step 1/4: Training baseline VGG11 model'),
        ('plot_training.py', 'Step 2/4: Generating training plots'),
        ('test_generalization.py', 'Step 3/4: Testing generalization (flips and noise)'),
        ('train_augmented.py', 'Step 4/4: Training with data augmentation'),
    ]
    
    results = []
    for script, description in experiments:
        success = run_script(script, description)
        results.append((script, success))
        if not success:
            print(f"\n⚠ Warning: {script} failed, continuing with remaining experiments...")
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    for script, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {script}")
    
    print("\n" + "=" * 70)
    print("RESULTS LOCATION")
    print("=" * 70)
    print("  results/vgg11_mnist.pth                      - Baseline model")
    print("  results/vgg11_mnist_augmented.pth            - Augmented model")
    print("  results/history.json                         - Baseline training history")
    print("  results/history_augmented.json               - Augmented training history")
    print("  results/training_plots.png                   - Training visualization")
    print("  results/generalization_results.json          - Baseline generalization tests")
    print("  results/generalization_results_augmented.json - Augmented generalization tests")
    print("  results/generalization_analysis.txt          - Analysis summary")
    print("=" * 70)
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    else:
        print("\n⚠ Some experiments failed. Check the output above for details.")
    
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
