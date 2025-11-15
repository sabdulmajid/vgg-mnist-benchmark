#!/bin/bash

echo "VGG11 MNIST Benchmark - Running all experiments"
echo "================================================"
echo ""

echo "Step 1: Training baseline model..."
python3 train.py
echo ""

echo "Step 2: Generating training plots..."
python3 plot_training.py
echo ""

echo "Step 3: Testing generalization (flips and noise)..."
python3 test_generalization.py
echo ""

echo "Step 4: Training with data augmentation..."
python3 train_augmented.py
echo ""

echo "All experiments completed!"
echo "Results are saved in the results/ directory"
