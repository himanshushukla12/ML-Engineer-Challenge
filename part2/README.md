
### Model Architecture
The model is a custom CNN with the following key features:
- Three convolutional blocks with ReLU activations and MaxPooling layers for feature extraction.
- An adaptive pooling layer to handle input flexibility.
- A fully connected linear layer for classification.

### Requirements
- Python 3.x
- CUDA-enabled GPU (48GB recommended for optimal batch size)
- Required packages: `torch`, `torchvision`, `optuna`, `matplotlib` (optional for visualization)

### Usage
1. **Dataset Preparation**:
    ```
     tiny-imagenet-200/
       ├── train/
       └── val/
     ```
   
2. **Run Training with Hyperparameter Optimization**:
   - Run the main script to start training with Optuna’s hyperparameter search:
   ```bash
   python train.py
   ```

### Hyperparameters from Best Trial
The best trial achieved an accuracy of **1.0** with the following parameters:
- **Learning Rate**: `0.000127`
- **Batch Size**: `996`
- **Num Channels**: `40`

### Results
The optimized model achieved high accuracy on the validation set. The batch size and number of channels were selected to maximize GPU utilization, enabling efficient training and convergence.

### Notes
- Ensure sufficient GPU memory, as the batch size is large (`996`).
- Adjust `n_trials` or the batch size if running on a different GPU.

### License
This project is open-source and licensed under the MIT License.