import os
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorRTInference:
    def __init__(self, engine_path):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(self.trt_logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
        
        # Setup stream and buffers
        self.stream = cuda.Stream()
        self.host_inputs = []
        self.host_outputs = []
        self.cuda_inputs = []
        self.cuda_outputs = []
        self.bindings = []
        
        # Allocate buffers
        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            
            # Convert TensorRT dtype to numpy dtype
            if tensor_dtype == trt.float32:
                np_dtype = np.float32
            elif tensor_dtype == trt.float16:
                np_dtype = np.float16
            else:
                np_dtype = np.float32  # Default to float32
            
            size = trt.volume(tensor_shape)
            host_mem = cuda.pagelocked_empty(size, np_dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def infer(self, input_data):
        try:
            # Set input shape
            input_name = self.engine.get_tensor_name(0)
            self.context.set_input_shape(input_name, (1, 3, 224, 224))
            
            # Copy input data to host buffer
            np.copyto(self.host_inputs[0], input_data.ravel())
            
            # Transfer to device
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            
            # Execute inference
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            # Transfer back to host
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            
            # Synchronize
            self.stream.synchronize()
            
            # Get output shape
            output_name = self.engine.get_tensor_name(1)
            output_shape = self.engine.get_tensor_shape(output_name)
            if output_shape == ():
                output_shape = (1, 384)
                
            return self.host_outputs[0].reshape(output_shape)
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise

    def __del__(self):
        try:
            del self.stream
            for cuda_mem in self.cuda_inputs + self.cuda_outputs:
                cuda_mem.free()
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

def load_dinov2(device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    return model.to(device)

def export_to_onnx(model, device, onnx_path):
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=12,
        do_constant_folding=True
    )
    logger.info(f"Model exported to ONNX at {onnx_path}")

def convert_to_trt(onnx_path, trt_path):
    try:
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)
        
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                error_msgs = []
                for idx in range(parser.num_errors):
                    error_msgs.append(str(parser.get_error(idx)))
                raise RuntimeError(f"ONNX parsing failed: {'; '.join(error_msgs)}")
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 precision")
        
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
        config.add_optimization_profile(profile)
        
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        with open(trt_path, 'wb') as f:
            f.write(engine_bytes)
            
        logger.info(f"TensorRT engine saved to {trt_path}")
        
    except Exception as e:
        logger.error(f"TensorRT conversion failed: {str(e)}")
        raise

def visualize_performance(pytorch_times, trt_times, save_path="perf_comparison.png"):
    plt.figure(figsize=(10, 6))
    
    models = ['PyTorch', 'TensorRT']
    times = [np.mean(pytorch_times), np.mean(trt_times)]
    errors = [np.std(pytorch_times), np.std(trt_times)]
    
    bars = plt.bar(models, times, yerr=errors, capsize=5)
    plt.title('Inference Time Comparison')
    plt.ylabel('Time (ms)')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms', ha='center', va='bottom')
    
    speedup = times[0] / times[1]
    plt.text(0.5, max(times) * 1.1, f'Speedup: {speedup:.2f}x',
             ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Performance plot saved to {save_path}")

def benchmark_inference(engine_path, n_warmup=50, n_iter=100):
    try:
        trt_model = TensorRTInference(engine_path)
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Warmup
        for _ in range(n_warmup):
            trt_model.infer(input_data)
        
        # Benchmark
        times = []
        for _ in range(n_iter):
            start = time.perf_counter()
            trt_model.infer(input_data)
            times.append((time.perf_counter() - start) * 1000)
        
        return times
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {str(e)}")
        raise

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT conversion")
    
    device = torch.device("cuda")
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    onnx_path = output_dir / "dinov2_vits14.onnx"
    trt_path = output_dir / "dinov2_vits14.trt"
    
    try:
        # Load and convert model
        model = load_dinov2(device)
        export_to_onnx(model, device, onnx_path)
        convert_to_trt(onnx_path, trt_path)
        
        # Benchmark PyTorch
        pytorch_times = []
        input_data = torch.randn(1, 3, 224, 224, device=device)
        
        # Warmup
        for _ in range(50):
            with torch.no_grad():
                model(input_data)
        
        # Benchmark
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                model(input_data)
            pytorch_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark TensorRT
        trt_times = benchmark_inference(trt_path)
        
        # Visualize results
        visualize_performance(pytorch_times, trt_times)
        
        print("\nPerformance Summary:")
        print(f"PyTorch mean latency: {np.mean(pytorch_times):.2f} ms")
        print(f"TensorRT mean latency: {np.mean(trt_times):.2f} ms")
        print(f"Speedup: {np.mean(pytorch_times)/np.mean(trt_times):.2f}x")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()