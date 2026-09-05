import numpy as np

def compress_updates(updates, ratio=0.1):
    """
    Compresses model updates (list of numpy arrays) using Top-K sparsification.
    Keeps only the fraction 'ratio' of parameters with the largest absolute magnitude.
    Returns:
        - compressed_updates: list of sparsified numpy arrays (same shapes, with zeros)
        - bytes_before: raw size of parameters in bytes (FP32)
        - bytes_after: size after compression using mask-based accounting
    """
    if ratio >= 1.0 or ratio <= 0.0:
        # No compression
        flat_params = np.concatenate([u.flatten() for u in updates])
        total_bytes = len(flat_params) * 4
        return updates, total_bytes, total_bytes
        
    shapes = [u.shape for u in updates]
    flat_updates = [u.flatten() for u in updates]
    concat_updates = np.concatenate(flat_updates)
    
    total_params = len(concat_updates)
    k = int(total_params * ratio)
    if k == 0:
        k = 1  # Keep at least one parameter
        
    # Get absolute values
    abs_updates = np.abs(concat_updates)
    
    # Find threshold for Top-K
    threshold = np.partition(abs_updates, -k)[-k]
    
    # Zero out elements below the threshold
    mask = abs_updates >= threshold
    compressed_concat = np.zeros_like(concat_updates)
    compressed_concat[mask] = concat_updates[mask]
    
    # Reconstruct original shapes
    compressed_updates = []
    curr_idx = 0
    for shape in shapes:
        size = np.prod(shape)
        flat_seg = compressed_concat[curr_idx:curr_idx + size]
        compressed_updates.append(flat_seg.reshape(shape))
        curr_idx += size
        
    # Calculate bytes
    num_non_zeros = int(np.sum(mask))
    bytes_before = total_params * 4
    
    # Mask-based communication cost:
    # 4 bytes per non-zero value, plus 1 bit per total parameters for the bitmask index
    bitmask_bytes = int(np.ceil(total_params / 8.0))
    bytes_after = num_non_zeros * 4 + bitmask_bytes
    
    # Ensure bytes_after does not exceed bytes_before in reports
    if bytes_after > bytes_before:
        bytes_after = bytes_before
        
    return compressed_updates, bytes_before, bytes_after
