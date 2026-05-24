import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class HybridDataset(Dataset):
    def __init__(self, seq, glob, labels):
        self.seq = torch.FloatTensor(seq)
        self.glob = torch.FloatTensor(glob)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.seq[idx], self.glob[idx], self.labels[idx]

def safe_stat(tensor, func, default=0.0):
    try:
        val = func(tensor)
        if isinstance(val, torch.Tensor):
            val = val.item()
        return val if not (np.isnan(val) or np.isinf(val)) else default
    except:
        return default

def get_sampled_layer_indices(n_total_layers: int, n_sample: int) -> list[int]:
    if n_total_layers <= 0:
        raise ValueError(f"Model reported {n_total_layers} transformer layers")

    if n_total_layers <= n_sample:
        indices = list(range(1, n_total_layers + 1))
        while len(indices) < n_sample:
            indices.append(indices[-1])
        return indices
    else:
        return [
            max(1, min(n_total_layers,
                       round(1 + (n_total_layers - 1) * i / (n_sample - 1))))
            for i in range(n_sample)
        ]

def get_anchor_stats(seq_feats: list, sampled_indices: list[int],
                     n_total_layers: int, anchor_fractions: list[float]) -> dict:
    sampled_arr = np.array(sampled_indices)
    anchor_stats = {}
    for pos, frac in enumerate(anchor_fractions):
        target_layer = max(1, min(n_total_layers, round(frac * n_total_layers)))
        closest_rank = int(np.argmin(np.abs(sampled_arr - target_layer)))
        anchor_stats[pos] = seq_feats[closest_rank]
    return anchor_stats

def compute_kurtosis(x):
    mu = x.mean()
    std = x.std()
    if std < 1e-9:
        return 0.0
    return torch.mean(((x - mu) / std) ** 4)

def compute_mad(x):
    med = torch.median(x)
    return torch.median(torch.abs(x - med))

def extract_features(question, answer, tokenizer, model_llm, config):
    prompt = f"Question: {question}\nAnswer: {answer}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=256).to(model_llm.device)

    with torch.no_grad():
        outputs = model_llm(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    logits        = outputs.logits[0, -1].float()

    n_total_layers  = len(hidden_states) - 1
    sampled_indices = get_sampled_layer_indices(n_total_layers,
                                                config.n_sample_layers)

    seq_feats = []

    for layer_idx in sampled_indices:
        hs      = hidden_states[layer_idx][0].float()
        last_hs = hs[-1]

        f = [
            safe_stat(last_hs, lambda x: torch.norm(x)),
            safe_stat(last_hs, lambda x: x.mean()),
            safe_stat(last_hs, lambda x: x.std()),
            safe_stat(last_hs, lambda x: x.min()),
            safe_stat(last_hs, lambda x: x.max()),
            safe_stat(last_hs, lambda x: (x > 0).float().mean()),
            safe_stat(last_hs, lambda x: (x.abs() < 0.1).float().mean()),
            safe_stat(last_hs, lambda x: -(F.softmax(x, dim=0) *
                                           torch.log(F.softmax(x, dim=0) + 1e-9)).sum()),
            safe_stat(last_hs, compute_kurtosis),
            safe_stat(last_hs, compute_mad)
        ]

        mean_hs = hs.mean(dim=0)
        f.extend([
            safe_stat(mean_hs, lambda x: torch.norm(x)),
            safe_stat(mean_hs, lambda x: x.std())
        ])

        seq_feats.append(f)

    anchor_stats = get_anchor_stats(seq_feats, sampled_indices,
                                    n_total_layers, config.anchor_fractions)

    probs = F.softmax(logits, dim=-1)
    top_k = torch.topk(probs, k=min(10, len(probs)))

    logit_entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    logit_std     = logits.std().item()
    logit_max     = logits.max().item()

    glob_feats = [
        top_k.values[0].item(),
        top_k.values[1].item() if len(top_k.values) > 1 else 0,
        (top_k.values[0] - top_k.values[1]).item() if len(top_k.values) > 1
            else top_k.values[0].item(),
        logit_entropy,
        logit_std,
        logit_max
    ]

    if len(top_k.values) >= 3:
        glob_feats.extend([
            top_k.values[2].item(),
            (top_k.values[0] - top_k.values[2]).item()
        ])
    else:
        glob_feats.extend([0, 0])

    norms      = [f[0] for f in seq_feats]
    norm_diffs = np.diff(norms)

    glob_feats.extend([
        np.mean(norm_diffs),
        np.std(norm_diffs),
        np.max(norm_diffs) if len(norm_diffs) > 0 else 0,
        np.min(norm_diffs) if len(norm_diffs) > 0 else 0,
        norms[-1] / (norms[0] + 1e-6),
        norms[-1] - norms[0]
    ])

    for pos in range(len(config.anchor_fractions)):
        glob_feats.extend(anchor_stats[pos][:3])

    glob_feats.append(anchor_stats[3][0] - anchor_stats[1][0])
    glob_feats.append(anchor_stats[3][1] * logit_entropy)

    glob_feats.extend([
        logit_std * np.mean(norm_diffs) if len(norm_diffs) > 0 else 0,
        logit_entropy * logit_std
    ])

    return np.array(seq_feats, dtype=np.float32), np.array(glob_feats, dtype=np.float32)
