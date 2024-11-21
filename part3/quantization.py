import torch
import torchvision
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

def load_dinov2(device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)
    model.eval()
    return model

def prepare_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    data_dir = Path("./tiny-imagenet-200/tiny-imagenet-200/test")
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def benchmark_model(model, test_loader, device, num_batches=100, use_amp=False):
    times = []
    model = model.to(device)

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            
            images = images.to(device)
            
            # Warmup GPU
            if i == 0 and device.type == 'cuda':
                _ = model(images)
                torch.cuda.synchronize()
                continue

            start = time.time()
            
            with torch.cuda.amp.autocast(enabled=use_amp):  # Mixed precision inference
                _ = model(images)
                
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            end = time.time()
            
            times.append(end - start)
    
    return sum(times) / len(times)

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    
    # Load model and data
    model = load_dinov2(device)
    test_loader = prepare_data()
    
    # Benchmark original model in FP32
    print("Benchmarking original model in FP32...")
    orig_time_fp32 = benchmark_model(model, test_loader, device, use_amp=False)

    # Benchmark model with mixed precision (FP16)
    print("Benchmarking model with mixed precision (FP16)...")
    orig_time_fp16 = benchmark_model(model, test_loader, device, use_amp=True)

    # Calculate speedup
    speedup = orig_time_fp32 / orig_time_fp16
    
    # Print results
    print("\nResults:")
    print(f"Original model inference time (FP32): {orig_time_fp32 * 1000:.2f} ms")
    print(f"Mixed precision model inference time (FP16): {orig_time_fp16 * 1000:.2f} ms")
    print(f"Speedup from FP32 to FP16: {speedup:.2f}x")

    # Plot results
    plt.figure(figsize=(10, 5))
    labels = ['FP32', 'FP16']
    times = [orig_time_fp32 * 1000, orig_time_fp16 * 1000]  # convert to milliseconds
    colors = ['skyblue', 'lightgreen']
    
    bars = plt.bar(labels, times, color=colors)
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Time Comparison: FP32 vs FP16')
    
    # Add inference time text inside each bar
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, 
                 f"{time:.2f} ms", ha='center', va='center', color='black', fontsize=12)
    
    # Display speedup text on the figure
    plt.figtext(0.8, 0.8, f"Speedup: {speedup:.2f}x", fontsize=12, ha='right')

    # Save and show the plot
    plt.savefig("inference.png")  # Save the figure as an image file
    plt.show()
if __name__ == "__main__":
    main()
