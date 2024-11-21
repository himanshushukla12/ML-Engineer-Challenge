import os
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import logging
from pathlib import Path

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
        
        # TensorRT to numpy dtype mapping
        dtype_mapping = {
            trt.float32: np.float32,
            trt.float16: np.float16,
            trt.int32: np.int32,
            trt.int8: np.int8,
            trt.bool: np.bool
        }
        
        # Allocate buffers for input and output
        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            
            # Get numpy dtype
            if tensor_dtype not in dtype_mapping:
                raise ValueError(f"Unsupported dtype: {tensor_dtype}")
            np_dtype = dtype_mapping[tensor_dtype]
            
            # Calculate size and allocate buffers
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
            # Set input shape using correct API
            input_name = self.engine.get_tensor_name(0)
            self.context.set_input_shape(input_name, (1, 3, 224, 224))
            
            # Copy input data to host buffer
            np.copyto(self.host_inputs[0], input_data.ravel())
            
            # Transfer input to device
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            
            # Execute inference with correct API
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            # Transfer output back to host
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            
            # Synchronize stream
            self.stream.synchronize()
            
            # Get output shape
            output_name = self.engine.get_tensor_name(1)
            output_shape = self.engine.get_tensor_shape(output_name)
            if output_shape == ():
                output_shape = (1, 384)  # DinoV2 output size
                
            return self.host_outputs[0].reshape(output_shape)
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise
    def __del__(self):
        try:
            # Clean up CUDA resources
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
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Enable FP16 if available
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 precision")
        
        # Set optimization profile
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
        config.add_optimization_profile(profile)
        
        # Build engine
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(trt_path, 'wb') as f:
            f.write(engine_bytes)
            
        logger.info(f"TensorRT engine saved to {trt_path}")
        
    except Exception as e:
        logger.error(f"TensorRT conversion failed: {str(e)}")
        raise

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
            start = cuda.Event()
            end = cuda.Event()
            
            start.record()
            trt_model.infer(input_data)
            end.record()
            
            end.synchronize()
            times.append(start.time_till(end))
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'median': np.median(times)
        }
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
        logger.info("Starting conversion pipeline...")
        
        model = load_dinov2(device)
        logger.info("Model loaded successfully")
        
        export_to_onnx(model, device, onnx_path)
        logger.info("ONNX export completed")
        
        convert_to_trt(onnx_path, trt_path)
        logger.info("TensorRT conversion completed")
        
        metrics = benchmark_inference(trt_path)
        print("\nTensorRT Inference Performance:")
        print(f"Mean latency: {metrics['mean']:.2f} ms")
        print(f"Std deviation: {metrics['std']:.2f} ms")
        print(f"Median latency: {metrics['median']:.2f} ms")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()