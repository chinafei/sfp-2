"""
sfp-2 轻量化调频预测模型

与 sfp 的区别：
  - 不做两阶段预测，直接从特征预测调频目标（5个时段×3指标）
  - 不拆分三分支 Transformer，使用单 Transformer 编码器 + MLP 解码头
  - 输入维度可配置，默认使用 sfp/datasets/npy 中已有特征

架构：
  输入: [batch, seq_len, input_dim]  (seq_len=96, input_dim 由配置决定)
  → Linear(input_dim → d_model)
  → TransformerEncoder(n_layers, n_heads, d_ff, dropout)
  → 96×d_model 池化为 fm_seq_len(5) × d_model
  → Linear(d_model → n_targets=3) 每段
  输出: [batch, 5, 3]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_dim: int = 130        # D-day 96pt 特征维度（可改为任意值）
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 2
    dropout: float = 0.2
    seq_len: int = 96           # 输入序列长度
    fm_seq_len: int = 5         # 调频时段数
    n_targets: int = 3          # 市场出清价/容量/边际排序价


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 96):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FMPredictor(nn.Module):
    """
    轻量化调频预测模型

    输入:  x96  [B, 96, input_dim]  当日96点特征
    可选:  x_day [B, daily_dim]      日级标量特征 (仅做简单线性融合)
    输出:  [B, 5, 3]
    """

    def __init__(self, cfg: ModelConfig, daily_dim: int = 0):
        super().__init__()
        self.cfg = cfg
        self.daily_dim = daily_dim

        # 输入投影
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)

        # 位置编码
        self.pe = SinusoidalPE(cfg.d_model, cfg.seq_len)

        # Transformer 编码器
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        # 日级特征融合（可选）
        if daily_dim > 0:
            self.daily_proj = nn.Linear(daily_dim, cfg.d_model)
        else:
            self.daily_proj = None

        # 96 → 5 时段池化（可学习权重）
        self.pool_query = nn.Parameter(torch.randn(cfg.fm_seq_len, cfg.d_model))

        # 解码头
        self.decoder = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.n_targets),
        )

    def forward(
        self,
        x96: torch.Tensor,
        x_day: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x96:   [B, 96, input_dim]
        x_day: [B, daily_dim] 或 None
        return: [B, 5, 3]
        """
        B = x96.size(0)

        # 投影 + 位置编码
        h = self.pe(self.input_proj(x96))           # [B, 96, d_model]

        # 日级特征融合（作为额外 token 拼接）
        if self.daily_proj is not None and x_day is not None:
            day_tok = self.daily_proj(x_day).unsqueeze(1)  # [B, 1, d_model]
            h = torch.cat([h, day_tok], dim=1)             # [B, 97, d_model]

        h = self.encoder(h)                         # [B, 96(+1), d_model]

        # 仅取 96pt 部分
        h96 = h[:, : self.cfg.seq_len]              # [B, 96, d_model]

        # 可学习注意力池化：query[5, d_model] × h96[B, 96, d_model]
        q = self.pool_query.unsqueeze(0).expand(B, -1, -1)  # [B, 5, d_model]
        attn = torch.softmax(
            torch.bmm(q, h96.transpose(1, 2)) / math.sqrt(self.cfg.d_model),
            dim=-1,
        )
        pooled = torch.bmm(attn, h96)               # [B, 5, d_model]

        return self.decoder(pooled)                  # [B, 5, 3]
