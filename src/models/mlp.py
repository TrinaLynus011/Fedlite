import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import yaml
import time
import os

try:
    import psutil
except ImportError:
    psutil = None

def load_config():
    with open("configs/models.yaml", "r") as f:
        return yaml.safe_load(f)["mlp"]

class LightweightMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def estimate_model_stats(model, input_dim):
    # Parameter count
    param_count = sum(p.numel() for p in model.parameters())
    
    # FP32 model size in KB (each parameter is 4 bytes)
    fp32_size_kb = (param_count * 4) / 1024.0
    
    # Estimated INT8 model size in KB (each parameter is 1 byte)
    int8_size_kb = (param_count * 1) / 1024.0
    
    # Measure inference latency (running 100 predictions)
    model.eval()
    dummy_input = torch.randn(100, input_dim)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    latency_ms = ((time.time() - start_time) / 100.0) * 1000.0
    
    # Measure RAM usage if psutil is available
    ram_mb = 0.0
    if psutil is not None:
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024.0 * 1024.0)
        
    return {
        "parameter_count": param_count,
        "fp32_size_kb": fp32_size_kb,
        "estimated_int8_size_kb": int8_size_kb,
        "inference_latency_ms": latency_ms,
        "ram_usage_mb": ram_mb
    }

def train_pytorch_mlp(X_train, y_train, X_val, y_val, config_override=None):
    config = load_config()
    if config_override:
        config.update(config_override)
        
    input_dim = X_train.shape[1]
    hidden_dims = config["hidden_dims"]
    dropout = float(config["dropout"])
    lr = float(config["learning_rate"])
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    weight_decay = float(config.get("weight_decay", 1e-4))
    
    print(f"Training PyTorch MLP (input_dim={input_dim}, hidden_dims={hidden_dims}, epochs={epochs})...")
    
    model = LightweightMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    pos_weight = torch.tensor(
        (y_train == 0).sum() / (y_train == 1).sum(),
        dtype=torch.float32
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Convert data to PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(X_train)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
    training_time = time.time() - start_time
    
    # Validation predictions
    model.eval()
    with torch.no_grad():
        y_val_prob = torch.sigmoid(
            model(torch.FloatTensor(X_val))
        ).squeeze().numpy()
        # Handle single element outputs if any
        if y_val_prob.ndim == 0:
            y_val_prob = np.array([y_val_prob.item()])
            
    stats = estimate_model_stats(model, input_dim)
    stats["training_time"] = training_time
    
    return model, y_val_prob, stats
