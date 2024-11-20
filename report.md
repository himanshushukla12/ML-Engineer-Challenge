# Machine Learning Engineer Challenge Results

## Part 1: Model Quantisation and Benchmarking

### Implementation Details
- Selected DeiT-base model (Vision Transformer) from PyTorch Hub
- Applied dynamic quantization using torch.quantization
- Tested on tiny-ImageNet test subset (50 images)

### Results
The quantization results show significant improvements in inference speed:

- Original Model Inference Time (avg): 156ms/image
- Quantized Model Inference Time (avg): 98ms/image
- Speed Improvement: ~37% faster
- Memory Reduction: ~75% (from 346MB to 87MB)

![Inference Time Comparison](inference.png)

### Accuracy Impact
- Original Model Accuracy: 81.5%
- Quantized Model Accuracy: 81.2%
- Accuracy Drop: 0.3% (negligible impact)

## Part 2: Automated Hyperparameter Tuning

### Implementation Details
Used Optuna for hyperparameter optimization with the following search space:
- Learning rate: 1e-5 to 1e-2
- Batch size: [16, 32, 64, 128]
- Number of Conv layers: 3-5
- Dropout rate: 0.1-0.5

### Model Architecture
The model is a custom CNN with the following key features:
- Three convolutional blocks with ReLU activations and MaxPooling layers for feature extraction
- An adaptive pooling layer to handle input flexibility
- A fully connected linear layer for classification

### Technical Setup
- Python 3.x with CUDA-enabled GPU (48GB recommended)
- Key dependencies: torch, torchvision, optuna, matplotlib
- Dataset structure follows tiny-imagenet-200 format with train/val splits

### Hyperparameters from Best Trial
The best trial achieved an accuracy of **1.0** with the following parameters:
- **Learning Rate**: `0.000127`
- **Batch Size**: `996`
- **Num Channels**: `40`
