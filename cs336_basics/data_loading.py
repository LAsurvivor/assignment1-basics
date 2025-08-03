import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str = 'mps',
):
    num_samples = len(x) - context_length
    indices = np.random.choice(num_samples, size=batch_size, replace=False)
    
    input_sequences = []
    target_sequences = []
    for idx in indices:
        input_seq = x[idx:idx + context_length]
        target_seq = x[idx + 1:idx + context_length + 1]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

    input_sequences = torch.tensor(input_sequences, device=device)
    target_sequences = torch.tensor(target_sequences, device=device)
    
    return input_sequences, target_sequences
    

