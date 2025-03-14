CIFAR-10 Image Classification with Customised ResNet-18

Overview
This project implements a custom ResNet-18 model to classify images from the CIFAR-10 dataset. The model is trained from scratch while adhering to competition constraints (<5M parameters, no pre-trained weights). It utilizes advanced data augmentation, mixed precision training (AMP), and learning rate scheduling to optimize performance. The entire pipeline runs in a single Python script or notebook, handling data loading, training, validation, and test prediction generation.

Features
✔️ Custom ResNet-18 architecture with optimized bottleneck blocks
✔️ CutMix & MixUp augmentation to improve generalization
✔️ SGD with Momentum & OneCycleLR for efficient training
✔️ Automatic Mixed Precision (AMP) for reduced memory usage and faster computation
✔️ Gradient Clipping & Label Smoothing to stabilize training
✔️ Best model saving & checkpoints for reproducibility
✔️ Generates submission file (submission.csv) in required format

Model Architecture
1. Modified ResNet-18 with optimized layers for CIFAR-10. 
2. Bottleneck Blocks replace standard residual blocks to reduce parameters while maintaining performance. 
3. SiLU activation instead of ReLU for better gradient flow. Adaptive Average Pooling before the fully connected layer. 
4. Dropout Regularization (0.3) to reduce overfitting. 
5. Total trainable parameters < 5 million to comply with competition constraints.

Project Structure
Since the entire project runs inside one file (.ipynb or .py), the structure is: cifar10-classification (Project root), cifar10_classifier.ipynb (Full Kaggle notebook with all code), cifar10_classifier.py (Single Python script version), README.md (Project documentation), submission.csv (Final test predictions), training_results.png (Saved plots of loss & accuracy). Additionally, datasets and models are downloaded/stored within the notebook runtime (not manually placed in folders).

Training Pipeline
1. Data Loading - Downloads CIFAR-10 dataset and applies transformations (flips, crops, color jitter). Uses CutMix & MixUp for better generalization. Splits dataset into training (80%) and validation (20%) sets. 
2. Model Initialization - Loads a custom ResNet-18 model, replacing standard layers with optimized versions. Ensures total trainable parameters are under 5M. 
3. Training - Uses SGD optimizer with momentum and OneCycleLR scheduler. Enables AMP (Automatic Mixed Precision) for faster training. Applies label smoothing (0.15) for better calibration. Uses gradient clipping to prevent exploding gradients. 
4. Validation - Runs inference on the validation set after every epoch. Saves the best-performing model based on validation accuracy. 
5. Testing & Inference - Loads the best model and runs it on test images (unlabeled CIFAR-10 data). Stores results in a dictionary: {image_id: predicted_label}. 
6. Submission Generation - Formats predictions into a CSV file (submission.csv) with the correct structure: ID, Labels 1, 3 2, 5 3, 

Results
Best Validation Accuracy: 91.24%. 


Authors
Devanshi Bhavsar (dnb7638), Nikhil Arora (na4063).





