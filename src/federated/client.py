"""
FedLite-Care Federated Client
==============================

SUPPORTED DP MECHANISMS
------------------------
mechanism = "dpsgd"  (ACTIVE)
    Privacy unit : individual RECORD (patient encounter).
    Per-batch:
      1. Individual backward pass for each record i.
      2. Clip per-record gradient to L2 norm C.
      3. Aggregate clipped gradients, add Gaussian noise N(0, (σ·C)² I).
      4. Optimizer step with noisy aggregated gradient / batch_size.
    No additional noise is added to the completed model update.
    Accountant: compute_dpsgd_epsilon() from privacy.dp_mechanism.

mechanism = "client_update_gaussian"  (DIAGNOSTIC / CALIBRATION)
    Privacy unit : federated CLIENT.
    Clips and noises the *completed* model update Δθ after local training.
    See diagnostic results — utility too low at useful epsilon levels.
    Accountant: compute_client_update_epsilon().

mechanism = None / disabled
    Standard FedAvg — no privacy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import flwr as fl
import os
import sys

sys.path.append(os.path.abspath("src"))
from models.mlp import LightweightMLP
from models.logistic_regression import PyTorchLR
from privacy.dp_mechanism import clip_and_add_noise, clip_grad_tensor
from compression.sparsification import compress_updates


class FedLiteClient(fl.client.NumPyClient):
    def __init__(self, client_id, X_train, y_train, X_val, y_val, model_type="mlp",
                 input_dim=44, hidden_dims=[64, 32], local_epochs=3, batch_size=128,
                 lr=0.005, dp_config=None, compression_config=None):
        self.client_id = client_id
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.dp_config = dp_config if dp_config is not None else {"enabled": False}
        self.compression_config = compression_config if compression_config is not None else {"enabled": False}

        # Data — np.copy() ensures writable arrays (silences UserWarning).
        self.X_train_t = torch.FloatTensor(np.copy(X_train))
        self.y_train_t = torch.FloatTensor(np.copy(y_train)).unsqueeze(1)
        self.X_val_t   = torch.FloatTensor(np.copy(X_val))
        self.y_val_t   = torch.FloatTensor(np.copy(y_val)).unsqueeze(1)

        self.train_dataset = TensorDataset(self.X_train_t, self.y_train_t)
        self.val_dataset   = TensorDataset(self.X_val_t,   self.y_val_t)

        self.num_positive = int((y_train == 1).sum())
        self.num_negative = int((y_train == 0).sum())

        # Model and loss
        if self.model_type == "mlp":
            self.model = LightweightMLP(
                input_dim=self.input_dim, hidden_dims=self.hidden_dims, dropout=0.2)
            if self.num_positive > 0:
                pw = torch.tensor(
                    self.num_negative / self.num_positive if self.num_negative > 0
                    else 1.0 / 7.680996884735203, dtype=torch.float32)
            else:
                pw = torch.tensor(7.680996884735203, dtype=torch.float32)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.model = PyTorchLR(input_dim=self.input_dim)
            if self.num_positive > 0:
                _pw = torch.tensor(
                    self.num_negative / self.num_positive if self.num_negative > 0
                    else 1.0 / 7.680996884735203, dtype=torch.float32)
            else:
                _pw = torch.tensor(7.680996884735203, dtype=torch.float32)
            self._lr_pos_weight = _pw
            self.criterion = nn.BCELoss(reduction="none")

        self.bytes_sent     = 0
        self.bytes_received = 0

    # ------------------------------------------------------------------ #
    # Flower interface                                                     #
    # ------------------------------------------------------------------ #

    def get_parameters(self, config):
        return [val.cpu().detach().numpy().copy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = {k: torch.tensor(np.copy(v)) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # ── 1. Receive global model ─────────────────────────────────── #
        self.set_parameters(parameters)
        old_parameters = [np.copy(p) for p in parameters]
        param_count    = sum(p.size for p in old_parameters)
        self.bytes_received += param_count * 4

        # ── 2. Read DP config ───────────────────────────────────────── #
        dp_enabled   = self.dp_config.get("enabled", False)
        mechanism    = self.dp_config.get("mechanism", "dpsgd")
        clipping_norm = float(self.dp_config.get("clipping_norm", 1.0))
        noise_mult   = float(self.dp_config.get("noise_multiplier", 0.0))

        _VALID = ("dpsgd", "client_update_gaussian")
        if dp_enabled and mechanism not in _VALID:
            raise ValueError(
                f"Unknown DP mechanism '{mechanism}'. "
                f"Valid values: {_VALID}."
            )

        # ── 3. DataLoader (seeded) ──────────────────────────────────── #
        loader_seed = config.get("seed", None)
        if loader_seed is not None:
            try:
                seed_value = int(loader_seed)
            except (TypeError, ValueError):
                seed_value = int(np.asarray(loader_seed).item())
            g = torch.Generator()
            g.manual_seed(seed_value + self.client_id)
            loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                shuffle=True, generator=g)
        else:
            loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                shuffle=True)

        # ── 4. Local training ───────────────────────────────────────── #
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        # Telemetry accumulators (DP-SGD specific)
        all_grad_norms  = []   # per-record L2 norms before clipping
        n_clipped_total = 0
        n_records_total = 0

        if dp_enabled and mechanism == "dpsgd" and noise_mult > 0.0:
            # ── DP-SGD training loop ──────────────────────────────── #
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            self.model.train()
            for epoch in range(self.local_epochs):
                for batch_x, batch_y in loader:
                    B = batch_x.size(0)
                    # Accumulate clipped per-record gradients
                    summed_clipped = [torch.zeros_like(p) for p in model_params]

                    for i in range(B):
                        xi = batch_x[i:i+1]
                        yi = batch_y[i:i+1]

                        self.model.zero_grad()
                        out = self.model(xi)
                        # Loss for single record
                        raw = self.criterion(out, yi)
                        if self.model_type == "lr":
                            w = torch.where(
                                yi == 1, self._lr_pos_weight, torch.ones_like(yi))
                            loss_i = (raw * w).mean()
                        else:
                            loss_i = raw
                        loss_i.backward()

                        # Clip per-record gradient; accumulate
                        rec_norm_sq = 0.0
                        for p, acc in zip(model_params, summed_clipped):
                            if p.grad is not None:
                                rec_norm_sq += p.grad.data.double().pow(2).sum().item()
                        rec_norm = float(rec_norm_sq ** 0.5)
                        all_grad_norms.append(rec_norm)
                        n_records_total += 1
                        was_clipped = rec_norm > clipping_norm
                        if was_clipped:
                            n_clipped_total += 1
                        scale = min(1.0, clipping_norm / (rec_norm + 1e-8))
                        for p, acc in zip(model_params, summed_clipped):
                            if p.grad is not None:
                                acc += p.grad.data * scale

                    # Add calibrated Gaussian noise to the summed clipped gradients
                    optimizer.zero_grad()
                    noise_std = noise_mult * clipping_norm
                    for p, acc in zip(model_params, summed_clipped):
                        noise = torch.normal(
                            mean=0.0, std=noise_std,
                            size=p.shape, device=p.device)
                        p.grad = (acc + noise) / B
                    optimizer.step()

        else:
            # ── Standard training (no DP, or client_update_gaussian) ─ #
            self.model.train()
            for epoch in range(self.local_epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    raw_loss = self.criterion(outputs, batch_y)
                    if self.model_type == "lr":
                        w = torch.where(batch_y == 1,
                                        self._lr_pos_weight,
                                        torch.ones_like(batch_y))
                        loss = (raw_loss * w).mean()
                    else:
                        loss = raw_loss
                    loss.backward()
                    optimizer.step()

        # ── 5. Compute delta Δθ ─────────────────────────────────────── #
        new_parameters = self.get_parameters(config={})
        deltas = [n - o for n, o in zip(new_parameters, old_parameters)]

        # ── 6. Client-Update Gaussian noise (if that mechanism) ─────── #
        pre_clip_norm = 0.0
        if dp_enabled and mechanism == "client_update_gaussian":
            if noise_mult > 0.0:
                dp_seed = None
                if loader_seed is not None:
                    dp_seed = loader_seed + self.client_id + 99991
                deltas, pre_clip_norm = clip_and_add_noise(
                    deltas, clipping_norm, noise_mult, seed=dp_seed)
            else:
                deltas, pre_clip_norm = clip_and_add_noise(
                    deltas, clipping_norm, 0.0)

        # ── 7. Communication compression ─────────────────────────────── #
        bytes_sent = param_count * 4
        if (self.compression_config.get("enabled", False)
                and self.compression_config.get("ratio", 1.0) < 1.0):
            ratio = float(self.compression_config.get("ratio", 1.0))
            deltas, _, bytes_after = compress_updates(deltas, ratio)
            bytes_sent = bytes_after
        self.bytes_sent += bytes_sent

        # ── 8. Reconstruct parameters to return to server ────────────── #
        final_parameters = [o + d for o, d in zip(old_parameters, deltas)]
        self.set_parameters(final_parameters)

        # ── 9. Training loss (telemetry, no gradient) ────────────────── #
        self.model.eval()
        train_loss = 0.0
        with torch.no_grad():
            for bx, by in DataLoader(self.train_dataset, batch_size=256):
                out = self.model(bx)
                raw = self.criterion(out, by)
                if self.model_type == "lr":
                    w = torch.where(by == 1, self._lr_pos_weight, torch.ones_like(by))
                    lv = (raw * w).mean()
                else:
                    lv = raw
                train_loss += lv.item() * len(bx)
        train_loss /= len(self.X_train_t)

        # ── 10. Build return metrics ─────────────────────────────────── #
        client_metrics = {
            "loss":             float(train_loss),
            "bytes_sent":       int(bytes_sent),
            "bytes_received":   int(param_count * 4),
            "num_positive":     self.num_positive,
            "num_negative":     self.num_negative,
            "dp_enabled":       dp_enabled,
            "mechanism":        mechanism if dp_enabled else "none",
            "privacy_unit":     ("record" if mechanism == "dpsgd"
                                 else ("client" if mechanism == "client_update_gaussian"
                                       else "none")) if dp_enabled else "none",
            "pre_clip_norm":    float(pre_clip_norm),
        }

        # DP-SGD gradient telemetry
        if all_grad_norms:
            client_metrics["mean_grad_norm"]   = float(np.mean(all_grad_norms))
            client_metrics["median_grad_norm"] = float(np.median(all_grad_norms))
            client_metrics["frac_clipped"]     = float(n_clipped_total / n_records_total)
            client_metrics["n_records_processed"] = n_records_total
        else:
            client_metrics["mean_grad_norm"]   = None
            client_metrics["median_grad_norm"] = None
            client_metrics["frac_clipped"]     = None
            client_metrics["n_records_processed"] = 0

        return final_parameters, len(self.train_dataset), client_metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        loss, all_preds, all_targets = 0.0, [], []
        with torch.no_grad():
            for bx, by in DataLoader(self.val_dataset, batch_size=256):
                out = self.model(bx)
                raw = self.criterion(out, by)
                if self.model_type == "lr":
                    w = torch.where(by == 1, self._lr_pos_weight, torch.ones_like(by))
                    lv = (raw * w).mean()
                else:
                    lv = raw
                loss += lv.item() * len(bx)
                probs = torch.sigmoid(out) if self.model_type == "mlp" else out
                all_preds.extend(probs.squeeze(1).numpy())
                all_targets.extend(by.squeeze(1).numpy())

        loss /= len(self.val_dataset)
        y_true = np.array(all_targets)
        y_prob = np.array(all_preds)

        if len(np.unique(y_true)) > 1:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = 0.5

        return float(loss), len(self.val_dataset), {"auc": auc}
