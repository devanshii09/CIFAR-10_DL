CIFAR-10 Image Classification with Customised ResNet-18

Overview
  This project implements a custom ResNet-18 model to classify images from the CIFAR-10 dataset. The model is trained from scratch while adhering to competition constraints (<5M parameters, no pre-trained weights). It utilizes advanced data augmentation, mixed precision training (AMP), and learning rate scheduling to optimize performance. The entire pipeline runs in a single Python script or notebook, handling data loading, training, validation, and test prediction generation.

Features
  1. Custom ResNet-18 architecture with optimized bottleneck blocks
  2. CutMix & MixUp augmentation to improve generalization
  3. SGD with Momentum & OneCycleLR for efficient training
  4. Automatic Mixed Precision (AMP) for reduced memory usage and faster computation
  5. Gradient Clipping & Label Smoothing to stabilize training
  6. Best model saving & checkpoints for reproducibility
  7. Generates submission file (submission.csv) in required format

Model Architecture
  1. Modified ResNet-18 with optimized layers for CIFAR-10. 
  2. Bottleneck Blocks replace standard residual blocks to reduce parameters while maintaining performance. 
  3. SiLU activation instead of ReLU for better gradient flow. Adaptive Average Pooling before the fully connected layer. 
  4. Dropout Regularization (0.3) to reduce overfitting. 
  5. Total trainable parameters < 5 million to comply with competition constraints.


Training Pipeline
  1. Data Loading - Downloads CIFAR-10 dataset and applies transformations (flips, crops, color jitter). Uses CutMix & MixUp for better generalization. Splits dataset into training (80%) and validation (20%) sets. 
  2. Model Initialization - Loads a custom ResNet-18 model, replacing standard layers with optimized versions. Ensures total trainable parameters are under 5M. 
  3. Training - Uses SGD optimizer with momentum and OneCycleLR scheduler. Enables AMP (Automatic Mixed Precision) for faster training. Applies label smoothing (0.15) for better calibration. Uses gradient clipping to prevent exploding gradients. 
  4. Validation - Runs inference on the validation set after every epoch. Saves the best-performing model based on validation accuracy. 
  5. Testing & Inference - Loads the best model and runs it on test images (unlabeled CIFAR-10 data). Stores results in a dictionary: {image_id: predicted_label}. 
  6. Submission Generation - Formats predictions into a CSV file (submission.csv) with the correct structure: ID, Labels 1, 3 2, 5 3, 

Requirements Installation
  To install all dependencies required for this project, use the provided requirements.txt file.
  
  1. Create and activate a virtual environment (optional but recommended)
          python -m venv venv  
          source venv/bin/activate  # For macOS/Linux  
          venv\Scripts\activate  # For Windows  

  2. Install all required dependencies
          pip install -r requirements.txt or
          !pip install torch torchvision numpy pandas matplotlib tqdm scikit-learn

Results
Best Validation Accuracy: 91.24%. 

Authors
Devanshi Bhavsar (dnb7638), Nikhil Arora (na4063)
