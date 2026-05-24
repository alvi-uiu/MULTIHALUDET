import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha_pos=0.5, alpha_neg=0.5, gamma=2.0):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha_pos * targets + self.alpha_neg * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=3, gamma_pos=1, clip=0.03):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        
        xs_pos = probs
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        if self.gamma_pos > 0:
            los_pos = los_pos * (1 - xs_pos) ** self.gamma_pos
        
        xs_neg = 1 - probs
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        if self.gamma_neg > 0:
            los_neg = los_neg * (probs ** self.gamma_neg)
        
        loss = -los_pos - los_neg
        return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        
    def forward(self, emb, labels):
        if emb.size(0) <= 1:
            return torch.tensor(0.0, device=emb.device, requires_grad=True)
        
        emb = F.normalize(emb, dim=1)
        sim = torch.matmul(emb, emb.T) / self.temp
        
        labels = labels.squeeze()
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_pos = mask_pos - torch.eye(mask_pos.size(0), device=mask_pos.device)
        
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        
        pos_loss = -(mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-9)
        
        return pos_loss.mean()
