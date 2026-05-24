import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleAttention(nn.Module):
    def __init__(self, hidden_dim, scales):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in scales
        ])
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim, self.num_scales),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        B, L, H = x.shape
        scale_feats = []
        
        for scale, proj in zip(self.scales, self.projections):
            if scale == 1:
                feat = proj(x)
            else:
                padding = (scale - L % scale) % scale
                if padding > 0:
                    x_pad = F.pad(x, (0, 0, 0, padding))
                else:
                    x_pad = x
                pooled = x_pad.reshape(B, -1, scale, H).mean(dim=2)
                feat = proj(pooled)
                feat = feat.repeat_interleave(scale, dim=1)[:, :L, :]
            scale_feats.append(feat)
        
        scale_feats = torch.stack(scale_feats, dim=-2)
        attn_weights = self.scale_attention(x).unsqueeze(-1)
        weighted_feats = (scale_feats * attn_weights).sum(dim=-2)
        return weighted_feats

class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        attn_scores = self.attention(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled, attn_weights

class HybridDeepLAP(nn.Module):
    def __init__(self, seq_dim, global_dim, hidden_dim, num_heads, num_layers, num_llm_layers, scales, dropout=0.3):
        super().__init__()
        
        self.seq_project = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.multi_scale = MultiScaleAttention(hidden_dim, scales)
        
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.layer_weights = nn.Parameter(torch.ones(num_llm_layers) / num_llm_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_llm_layers, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attn_pool = SelfAttentionPooling(hidden_dim)
        
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion_dim = hidden_dim + (hidden_dim // 2)
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128)
        )
        
    def forward(self, x_seq, x_global, return_attn=False, return_embedding=False, return_features=False):
        h_seq = self.seq_project(x_seq)
        
        h_multi = self.multi_scale(h_seq)
        h_seq = h_seq + self.scale_fusion(h_multi)
        
        layer_weights_norm = F.softmax(self.layer_weights, dim=0)
        h_seq = h_seq * layer_weights_norm.unsqueeze(0).unsqueeze(-1)
        h_seq = h_seq + self.pos_embedding
        
        h_seq = self.transformer(h_seq)
        
        seq_pooled, attn_weights = self.attn_pool(h_seq)
        
        global_encoded = self.global_mlp(x_global)
        
        combined = torch.cat([seq_pooled, global_encoded], dim=1)
        
        gate = self.fusion_gate(combined)
        combined = combined * gate
        
        if return_features:
            return combined
        
        logits = self.classifier(combined)
        
        if return_embedding:
            embedding = self.contrastive_head(combined)
            if return_attn:
                return logits, attn_weights.squeeze(-1), embedding
            return logits, embedding
        
        if return_attn:
            return logits, attn_weights.squeeze(-1)
        
        return logits
