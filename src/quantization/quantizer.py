import torch
import torch.nn as nn
import time
import os
import sys
import numpy as np
import yaml

# Add src/ to path
sys.path.append(os.path.abspath("src"))
from models.mlp import LightweightMLP

def quantize_mlp_model(model_state_path, input_dim, hidden_dims=[64, 32], X_val=None):
    """
    Applies post-training dynamic quantization to convert FP32 MLP weights to INT8.
    """
    # Load configurations to match model architecture
    dropout = 0.2
    try:
        with open("configs/models.yaml", "r") as f:
            models_config = yaml.safe_load(f)
            dropout = float(models_config["mlp"]["dropout"])
            hidden_dims = models_config["mlp"]["hidden_dims"]
    except Exception as e:
        print(f"Warning: could not load configs in quantizer: {e}")
        
    # 1. Load FP32 model
    fp32_model = LightweightMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    fp32_model.load_state_dict(torch.load(model_state_path))
    fp32_model.eval()
    
    # Save temporary file to measure FP32 size
    fp32_temp_path = "artifacts/checkpoints/central/temp_fp32.pt"
    torch.save(fp32_model.state_dict(), fp32_temp_path)
    fp32_size = os.path.getsize(fp32_temp_path)
    
    # 2. Quantize model (dynamically quantize all nn.Linear layers to qint8)
    # Note: PyTorch dynamic quantization is supported on CPU out of the box
    quantized_model = torch.quantization.quantize_dynamic(
        fp32_model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save quantized model
    os.makedirs("artifacts/checkpoints/final", exist_ok=True)
    quant_path = "artifacts/checkpoints/final/mlp_quantized.pt"
    # Note: Save state_dict of quantized model requires saving the whole model or special handling,
    # let's save the whole quantized model object.
    torch.save(quantized_model, quant_path)
    quant_size = os.path.getsize(quant_path)
    
    # Clean up temp file
    if os.path.exists(fp32_temp_path):
        os.remove(fp32_temp_path)
        
    print(f"FP32 size: {fp32_size / 1024:.2f} KB | INT8 size: {quant_size / 1024:.2f} KB")
    print(f"Size reduction: {(1 - quant_size/fp32_size)*100:.2f}%")
    
    # 3. Evaluate consistency and latency
    consistency_mae = 0.0
    val_probs_fp32 = []
    val_probs_int8 = []
    
    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val)
        with torch.no_grad():
            preds_fp32 = torch.sigmoid(fp32_model(X_val_t)).squeeze().numpy()
            preds_int8 = torch.sigmoid(quantized_model(X_val_t)).squeeze().numpy()
            
            consistency_mae = float(np.mean(np.abs(preds_fp32 - preds_int8)))
            
        # 4. Measure inference latency
        dummy_input = torch.randn(100, input_dim)
        
        # FP32 Latency
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = fp32_model(dummy_input)
        latency_fp32 = ((time.time() - start) / 100.0) * 1000.0
        
        # INT8 Latency
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = quantized_model(dummy_input)
        latency_int8 = ((time.time() - start) / 100.0) * 1000.0
        
        print(f"FP32 Latency: {latency_fp32:.4f} ms | INT8 Latency: {latency_int8:.4f} ms")
    else:
        latency_fp32 = 0.0
        latency_int8 = 0.0
        
    return {
        "fp32_size_bytes": fp32_size,
        "int8_size_bytes": quant_size,
        "size_reduction_pct": (1 - quant_size/fp32_size) * 100,
        "prediction_consistency_mae": consistency_mae,
        "latency_fp32_ms": latency_fp32,
        "latency_int8_ms": latency_int8
    }

if __name__ == "__main__":
    # Test quantizer
    mlp_state = "artifacts/checkpoints/central/mlp.pt"
    if os.path.exists(mlp_state):
        print("Testing quantization on MLP model...")
        stats = quantize_mlp_model(mlp_state, input_dim=44, X_val=np.random.randn(100, 44))
        print("Quantization stats:", stats)
    else:
        print("No trained MLP model found. Train baselines first.")
