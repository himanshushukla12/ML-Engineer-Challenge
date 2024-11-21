import torch
import tensorrt as trt
from torch.onnx import export
import os

def load_dinov2(device):
    # Load the PyTorch model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)
    model.eval()
    return model

def export_to_onnx(model, device, onnx_file_path):
    # Define a dummy input for tracing
    dummy_input = torch.randn(1, 3, 224, 224, device=device)  # Adjust shape as needed
    # Export the model to ONNX
    export(
        model, 
        dummy_input, 
        onnx_file_path, 
        export_params=True,
        opset_version=13,  # Specify the ONNX opset version
        input_names=['input'], 
        output_names=['output']
    )
    print(f"Model exported to ONNX format at {onnx_file_path}")

def convert_to_trt(onnx_file_path, trt_file_path):
    # Set up TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GB
    
    # Parse the ONNX model
    parser = trt.OnnxParser(network, logger)
    with open(onnx_file_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse the ONNX file.")
    
    # Build the TensorRT engine
    print("Building TensorRT engine...")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build the TensorRT engine.")
    
    # Serialize the engine to file
    with open(trt_file_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved at {trt_file_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(device)

    # File paths
    onnx_file_path = "dinov2_vits14.onnx"
    trt_file_path = "dinov2_vits14.trt"

    # Export to ONNX
    export_to_onnx(model, device, onnx_file_path)
    
    # Convert ONNX to TensorRT
    convert_to_trt(onnx_file_path, trt_file_path)
