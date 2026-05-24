import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import random
from src.data.feature_extractor import HybridDataset
from src.models.multihaludet import HybridDeepLAP
from src.models.losses import FocalLoss, AsymmetricLoss, ContrastiveLoss
from src.training.augmentations import mixup_data, cutmix_data
from sklearn.metrics import roc_auc_score

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def train_deep_model_fold(X_seq_tr, X_glob_tr, y_tr, X_seq_val, X_glob_val, y_val, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_pos = sum(y_tr)
    n_neg = len(y_tr) - n_pos
    
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)]).to(device)
    
    train_ds = HybridDataset(X_seq_tr, X_glob_tr, y_tr)
    val_ds = HybridDataset(X_seq_val, X_glob_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    
    model = HybridDeepLAP(
        seq_dim=X_seq_tr.shape[2],
        global_dim=X_glob_tr.shape[1],
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_llm_layers=X_seq_tr.shape[1],
        scales=config.scales,
        dropout=config.dropout
    ).to(device)
    
    bce_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    focal_crit = FocalLoss(alpha_pos=config.focal_alpha_pos, alpha_neg=config.focal_alpha_neg, gamma=config.focal_gamma)
    asym_crit = AsymmetricLoss(gamma_neg=3, gamma_pos=1, clip=0.03)
    cont_crit = ContrastiveLoss(config.contrastive_temp)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=config.min_lr)
    
    if config.use_ema:
        ema = EMA(model, decay=config.ema_decay)
    
    best_auc = 0
    best_model = None
    patience_counter = 0
    
    for epoch in range(config.epochs):
        model.train()
        
        if epoch < config.warmup_epochs:
            warmup_lr = config.learning_rate * (epoch + 1) / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        for s, g, t in train_loader:
            s, g, t = s.to(device), g.to(device), t.to(device).unsqueeze(1)
            optimizer.zero_grad()
            
            aug_choice = random.random()
            if aug_choice < 0.33 and config.use_mixup:
                s, g, t_a, t_b, lam = mixup_data(s, g, t, config.mixup_alpha)
                out, emb = model(s, g, return_embedding=True)
                loss = lam * bce_crit(out, t_a) + (1 - lam) * bce_crit(out, t_b)
            elif aug_choice < 0.66 and config.use_cutmix:
                s, g, t_a, t_b, lam = cutmix_data(s, g, t, config.cutmix_alpha)
                out, emb = model(s, g, return_embedding=True)
                loss = lam * bce_crit(out, t_a) + (1 - lam) * bce_crit(out, t_b)
            else:
                out, emb = model(s, g, return_embedding=True)
                t_smooth = t * (1 - config.label_smoothing) + 0.5 * config.label_smoothing
                loss = 0.45 * bce_crit(out, t_smooth) + 0.35 * focal_crit(out, t) + 0.20 * asym_crit(out, t)
                loss += config.contrastive_weight * cont_crit(emb, t)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if config.use_ema:
                ema.update()
        
        if config.use_ema:
            ema.apply_shadow()
        
        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for s, g, t in val_loader:
                s, g = s.to(device), g.to(device)
                out = model(s, g)
                probs.extend(torch.sigmoid(out).cpu().numpy())
                targets.extend(t.numpy())
        
        if config.use_ema:
            ema.restore()
        
        auc_score = roc_auc_score(targets, probs)
        if epoch >= config.warmup_epochs:
            scheduler.step(auc_score)
        
        if auc_score > best_auc:
            best_auc = auc_score
            if config.use_ema:
                ema.apply_shadow()
                best_model = copy.deepcopy(model.state_dict())
                ema.restore()
            else:
                best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            break
    
    model.load_state_dict(best_model)
    return model, best_auc

def extract_features_batch(model, X_seq, X_glob):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    dataset = HybridDataset(X_seq, X_glob, np.zeros(len(X_seq)))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    features = []
    with torch.no_grad():
        for s, g, _ in loader:
            s, g = s.to(device), g.to(device)
            feats = model(s, g, return_features=True)
            features.append(feats.cpu().numpy())
    return np.concatenate(features, axis=0)
