
This project implements a **binary classification model** using **PyTorch** to predict a target variable (`sus_label`) from structured data. The workflow includes data preprocessing, model training, and evaluation using accuracy metrics.

## Project Overview

The goal of this project is to build a neural network that can classify data into two categories based on input features.

The project includes:

* Data loading and preprocessing
* Feature scaling using `StandardScaler`
* Building a neural network using PyTorch
* Training the model using gradient descent
* Evaluating performance using accuracy


## Dataset

The project uses three CSV files:

* `labelled_train.csv` – Training dataset
* `labelled_validation.csv` – Validation dataset
* `labelled_test.csv` – Testing dataset

Each dataset contains:

* Feature columns
* Target column: `sus_label` (binary: 0 or 1)


## Technologies Used

* Python 
* PyTorch 
* Pandas
* Scikit-learn
* TorchMetrics


## Data Preprocessing

1. Features and labels are separated.
2. Data is standardized using `StandardScaler`:

   * Fit on training data only
   * Transform validation and test data using the same scaler
3. Data is converted into PyTorch tensors.


## Model Architecture

A simple feedforward neural network:

* Input Layer → based on number of features
* Hidden Layer 1: 128 neurons + ReLU
* Hidden Layer 2: 64 neurons + ReLU
* Output Layer: 1 neuron + Sigmoid


## Loss Function & Optimizer

* **Loss Function:** `CrossEntropyLoss` *(Note: for binary classification, `BCEWithLogitsLoss` is usually preferred)*
* **Optimizer:** Stochastic Gradient Descent (SGD)
* Learning Rate: `1e-3`
* Weight Decay: `1e-4`


## Training Process

* Number of epochs: `10`
* Forward pass → compute predictions
* Loss calculation
* Backpropagation
* Parameter update using optimizer

## Evaluation

The model is evaluated using **accuracy** on:

* Training data
* Validation data
* Test data

Accuracy is computed using `torchmetrics.Accuracy` with a binary classification setting.


## How to Run

1. Open the Jupyter Notebook
2. Make sure required libraries are installed:

   ```bash
   pip install pandas scikit-learn torch torchmetrics
   ```
3. Place the dataset files in the working directory
4. Run all cells sequentially

## Improvements

* Consider using:

  * `BCEWithLogitsLoss` instead of `CrossEntropyLoss` for binary classification
  * `Adam` optimizer for better convergence
* Add mini-batch training using `DataLoader` for scalability
* Increase number of epochs for better performance
* Experiment with deeper architectures


## Future Work

* Hyperparameter tuning
* Adding regularization techniques
* Using more advanced architectures
* Improving generalization performance
