# 05. Privacy Methodology

The privacy method is DP-SGD with a record-level privacy unit. The implementation uses per-record L2 gradient clipping and adds Gaussian noise to the aggregated update. The accountant used by the project is the tight second-order subsampled Gaussian RDP implementation via compute_dpsgd_epsilon().

The project makes a clear distinction between verified results and calibrated pending runs. Verified evidence includes the sigma=0.8 pilot result. The sigma=2.3 target is a calibrated epsilon≈5 configuration that remains pending final completion.
