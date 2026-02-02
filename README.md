Robust Image Classification under Noisy Supervision

Overview
This repository contains an end-to-end image classification pipeline designed to handle noisy training labels. Developed as a solution for the Max Healthcare technical assessment, the project focuses on distinguishing 7 classes of skin images (28$\times$28) using a lightweight Convolutional Neural Network (CNN).

The core objective was to build a system that remains robust despite unreliable training annotations, leveraging clean validation data for model selection and ensuring a streamlined inference process for future evaluation.

The Challenge
In real-world medical imaging, obtaining "gold standard" annotations is expensive and rare. This project addresses a scenario where:

Training Data: Contains noisy labels (incorrect annotations).

Validation Data: Contains clean labels (ground truth).

Goal: Train a model that generalizes to unseen data without overfitting to the noise in the training set.

Methodology
1. Data Preprocessing

The input data consists of 28$\times$28 RGB images. To standardize inputs for the model:

Normalization: Pixel values are scaled to the range [0,1].

Grayscale Conversion: RGB channels are collapsed to a single channel to reduce dimensionality and focus on structural features.

Tensor Formatting: Data is reshaped to (N,1,28,28) for PyTorch ingestion.

2. Model Architecture

A compact CNN was chosen intentionally. Large models often memorize noise; a smaller architecture acts as a regularizer, forcing the model to learn dominant patterns rather than specific incorrect labels.

Conv Layers: 2 layers with ReLU activation for feature extraction.

Pooling: Max pooling to reduce spatial dimensions and computation.

Classifier: Fully connected layers with Dropout to prevent overfitting.

3. Noise Mitigation Strategy

To combat the noisy training labels, Label Smoothing was implemented within the loss function.

Instead of forcing the model to predict a definitive "1" or "0" (which is harmful if the label is wrong), label smoothing encourages softer probability distributions.

This prevents the model from becoming over-confident in incorrect training examples, leading to better generalization on the clean validation set.

Implementation Details
Stack

Framework: PyTorch

Language: Python 3.x

Key Libraries: NumPy, Matplotlib, Scikit-learn

Training Configuration

Optimizer: Adam (Adaptive Moment Estimation)

Loss Function: CrossEntropyLoss with Label Smoothing

Model Selection: The training loop monitors validation accuracy at every epoch. The state of the model with the highest validation accuracy is automatically saved as best_model.pth.

Repository Structure
Plaintext
├── README.md               # Project documentation
├── main.ipynb              # Notebook containing training & inference logic
├── data/                   # Directory for .npy or .npz data files
│   ├── x_train.npy
│   ├── y_train.npy
│   └── ...
├── best_model.pth          # Saved weights of the highest performing model
Results
Metric: Accuracy on Clean Validation Set

Performance: ~67%

Given the constraints of noisy supervision and low-resolution input data, this performance indicates that the model successfully learned to distinguish classes without overfitting to the corrupt training labels.

Future Work
To further improve performance given more time or resources:

Noise Correction: Implement "Confident Learning" (e.g., via Cleanlab) to statistically identify and prune incorrect labels before training.

Augmentation: Introduce random rotations and flips to make the model invariant to orientation.

Ensembling: Train multiple lightweight models and average their predictions to smooth out variance.



Author
Divyansh Sharma
Submitted for the Max Healthcare Technical Assessment.
