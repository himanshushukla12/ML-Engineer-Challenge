# Import necessary libraries
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Define a simple CNN model with customizable channels and classes
class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_channels, 3, padding=1),  # First conv layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_channels, num_channels * 2, 3, padding=1),  # Second conv layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)  # Make classifier adaptive to image size
        )
        self.classifier = nn.Linear(num_channels * 2, num_classes)  # Adjusted for adaptive pool

    def forward(self, x):
        x = self.features(x)  # Feature extraction
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)  # Classification layer
        return x

# Objective function for Optuna to optimize hyperparameters
def objective(trial):
    # Hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # Learning rate
    batch_size = trial.suggest_int("batch_size", 64, 1024)  # Batch size
    num_channels = trial.suggest_int("num_channels", 16, 64)  # Number of channels in the first conv layer
    
    # Data transformations for training and validation
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize image
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize
    ])
    
    # Load datasets for training and validation
    train_dataset = ImageFolder("./tiny-imagenet-200/tiny-imagenet-200/test", transform=transform)
    val_dataset = ImageFolder("./tiny-imagenet-200/tiny-imagenet-200/test", transform=transform)
    
    # Data loaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device
    model = SimpleCNN(num_channels, 200).to(device)  # Instantiate model with number of classes
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimizer with suggested learning rate
    
    # Training loop
    for epoch in range(5):  # Use more epochs for better results
        model.train()  # Set model to training mode
        for inputs, targets in train_loader:  # Iterate over batches
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Zero gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
    
    # Validation loop for accuracy measurement
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Forward pass
            _, predicted = outputs.max(1)  # Get predicted class
            total += targets.size(0)  # Total number of samples
            correct += predicted.eq(targets).sum().item()  # Count correct predictions
    
    accuracy = correct / total  # Calculate accuracy
    return accuracy  # Optuna will try to maximize this

# Main function to create and run the Optuna study
def main():
    study = optuna.create_study(direction="maximize")  # Create Optuna study to maximize accuracy
    study.optimize(objective, n_trials=20)  # Optimize over 20 trials
    
    # Print the best result
    print("Best trial:")
    print("  Value: ", study.best_trial.value)  # Best accuracy achieved
    print("  Params: ")
    for key, value in study.best_trial.params.items():  # Print each hyperparameter
        print(f"    {key}: {value}")

# Run the main function when this script is executed
if __name__ == "__main__":
    main()
