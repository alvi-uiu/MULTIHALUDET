import torch

class Config:
    seed = 42
    n_inner_folds = 5
    test_size = 0.20
    
    hidden_dim = 384
    num_heads = 8
    num_layers = 6
    scales = [1, 2, 4, 8]
    mlp_hidden_dim = 288
    dropout = 0.25
    
    batch_size = 28
    epochs = 45
    learning_rate = 2e-4
    weight_decay = 6e-5
    patience = 15
    min_lr = 1e-7
    warmup_epochs = 5
    
    use_mixup = True
    mixup_alpha = 0.15
    use_cutmix = True
    cutmix_alpha = 0.15
    label_smoothing = 0.02
    contrastive_weight = 0.20
    contrastive_temp = 0.04
    grad_clip = 0.5
    
    use_ema = True
    ema_decay = 0.999
    use_swa = True
    swa_start = 30
    swa_lr = 5e-5
    
    use_class_weights = False
    minority_oversample = 1.0
    decision_bias = 0.0
    
    focal_alpha_pos = 0.5
    focal_alpha_neg = 0.5
    focal_gamma = 2.0

    n_sample_layers  = 32
    anchor_fractions = [0.25, 0.50, 0.75, 1.00]
    model_dtype      = torch.float16

MODEL_REGISTRY = {
    "mistral-7b" : "mistralai/Mistral-7B-Instruct-v0.2",
    "llama2-7b"  : "meta-llama/Llama-2-7b-hf",
    "qwen-3b"    : "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b"    : "Qwen/Qwen2.5-7B-Instruct",
}

def get_config():
    return Config()
