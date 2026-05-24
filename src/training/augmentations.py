import torch
import numpy as np

def mixup_data(x_seq, x_glob, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x_seq.size(0)
    index = torch.randperm(batch_size).to(x_seq.device)
    mixed_seq = lam * x_seq + (1 - lam) * x_seq[index]
    mixed_glob = lam * x_glob + (1 - lam) * x_glob[index]
    y_a, y_b = y, y[index]
    return mixed_seq, mixed_glob, y_a, y_b, lam

def cutmix_data(x_seq, x_glob, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x_seq.size(0)
    index = torch.randperm(batch_size).to(x_seq.device)
    
    seq_len = x_seq.size(1)
    cut_len = int(seq_len * (1 - lam))
    cut_start = np.random.randint(0, seq_len - cut_len + 1)
    
    x_seq_mixed = x_seq.clone()
    x_seq_mixed[:, cut_start:cut_start+cut_len, :] = x_seq[index, cut_start:cut_start+cut_len, :]
    
    glob_dim = x_glob.size(1)
    cut_glob = int(glob_dim * (1 - lam))
    cut_glob_start = np.random.randint(0, glob_dim - cut_glob + 1)
    
    x_glob_mixed = x_glob.clone()
    x_glob_mixed[:, cut_glob_start:cut_glob_start+cut_glob] = x_glob[index, cut_glob_start:cut_glob_start+cut_glob]
    
    y_a, y_b = y, y[index]
    return x_seq_mixed, x_glob_mixed, y_a, y_b, lam
