import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dual_tower_model import DualTowerBiddingModel, PenaltyBiddingLoss

class MultiDayFusionModel(nn.Module):
    def __init__(self, seq_input_dim, static_input_dim, hidden_dim=64, weight_strategy='learnable'):
        """
        处理带有时间间隔的多日预测数据（D日，D-1日，D-2日）。
        weight_strategy: 
            'decay': 专家设定的经验衰减权重（例如 D日 0.7, D-1日 0.2, D-2日 0.1）
            'learnable': 通过网络自带的 Attention 自动学习历史预测数据的参考权重
        """
        super(MultiDayFusionModel, self).__init__()
        self.weight_strategy = weight_strategy
        
        # 基础双塔提取器，共享权重（抽取每一天的特征向量）
        self.base_model = DualTowerBiddingModel(seq_input_dim, static_input_dim, hidden_dim)
        
        # 摘除 base_model 的最后一层，只用它提取 128 维的特征向量
        self.feature_extractor = nn.Sequential(
            *list(self.base_model.fc_fusion.children())[:-1] 
        )
        
        # 机制 A: 自定义可学习的时序注意力权重（Learnable Weights）
        if self.weight_strategy == 'learnable':
            # 针对 3 天的特征向量进行自适应打分
            self.day_attention = nn.Linear(64, 1)
            
        # 融合后的最终输出头
        self.final_head = nn.Linear(64, 5 * 3)

    def forward(self, xs_d0, xs_d1, xs_d2, xst_d0, xst_d1, xst_d2):
        # 分别抽取 D, D-1, D-2 的隐层特征
        # 由于我们拦截了 base_model 的前向，需要重写以获取融合中间层
        def extract_features(xs, xst):
            gru_out, _ = self.base_model.gru(xs)
            context_vec, _ = self.base_model.attention(gru_out, self.base_model.time_prior.unsqueeze(0))
            static_vec = self.base_model.static_net(xst)
            fused = torch.cat([context_vec, static_vec], dim=1)
            return self.feature_extractor(fused) # (batch, 64)

        feat_d0 = extract_features(xs_d0, xst_d0)
        feat_d1 = extract_features(xs_d1, xst_d1)
        feat_d2 = extract_features(xs_d2, xst_d2)
        
        # ---------- 两种权重设置策略 ----------
        if self.weight_strategy == 'decay':
            # 策略 1：固定指数衰减权重 (经验主义，时间越近权重越高)
            # 例如 D日占70%，D-1占20%，D-2占10%
            w0, w1, w2 = 0.7, 0.2, 0.1
            fused_feat = w0 * feat_d0 + w1 * feat_d1 + w2 * feat_d2
            
        elif self.weight_strategy == 'learnable':
            # 策略 2：自适应注意力权重 (数据驱动)
            # feats: (batch, 3, 64)
            feats = torch.stack([feat_d0, feat_d1, feat_d2], dim=1) 
            attn_scores = self.day_attention(feats).squeeze(-1) # (batch, 3)
            attn_weights = torch.softmax(attn_scores, dim=-1)   # 自动归一化到 0~1
            
            # 打印监控权重学习情况 (选第一个batch打印监控)
            if not self.training:
                print(f"[监控] 当前学习出的分配权重 - D日: {attn_weights[0,0]:.3f}, D-1: {attn_weights[0,1]:.3f}, D-2: {attn_weights[0,2]:.3f}")
                
            # 加权求和 (batch, 1, 3) * (batch, 3, 64) => (batch, 1, 64)
            fused_feat = torch.bmm(attn_weights.unsqueeze(1), feats).squeeze(1)
            
        # 最终输出头
        out_flat = self.final_head(fused_feat)
        out_raw = out_flat.view(-1, 5, 3)
        return out_raw


class MultiDayDataset(Dataset):
    def __init__(self, num_samples=100, seq_dim=12, static_dim=4):
        # 模拟生成 3 天的数据。通常前两天的数据会更平滑或有一定的 noise
        self.xs_d0 = torch.randn(num_samples, 96, seq_dim)
        self.xs_d1 = self.xs_d0 + torch.randn_like(self.xs_d0) * 0.1
        self.xs_d2 = self.xs_d0 + torch.randn_like(self.xs_d0) * 0.2
        
        self.xst_d0 = torch.randn(num_samples, static_dim)
        self.xst_d1 = self.xst_d0 + 0.1
        self.xst_d2 = self.xst_d0 + 0.2
        
        # 模拟目标价格与容量
        # index 0: 调频需求容量 (真实量级位于 2500 ~ 4800，均值约 3750) -> 3000-4000
        # index 1: 边际排序价 (真实量级位于 6 ~ 75，均值 15.4) -> 模拟在限价周围波动
        # index 2: 市场出清均价 (真实量级位于 5 ~ 15，均值 8.9) -> 5-15之间
        self.y_targets = torch.zeros(num_samples, 5, 3)
        self.y_targets[..., 0] = torch.rand(num_samples, 5) * 1000 + 3000 # 容量: 3000-4000
        self.y_targets[..., 1] = torch.rand(num_samples, 5) * 20 + 5      # 边际价
        self.y_targets[..., 2] = torch.rand(num_samples, 5) * 10 + 5      # 出清价
        
        self.p_min = torch.tensor([[5., 5., 10., 10., 5.]] * num_samples)
        self.p_max = torch.tensor([[10., 10., 15., 15., 10.]] * num_samples)
        
    def __len__(self): return len(self.y_targets)
    def __getitem__(self, idx):
        return (self.xs_d0[idx], self.xs_d1[idx], self.xs_d2[idx],
                self.xst_d0[idx], self.xst_d1[idx], self.xst_d2[idx],
                self.y_targets[idx], self.p_min[idx], self.p_max[idx])

def train_multi_day_model(strategy='learnable'):
    print(f"\n>>> 启动多日联合训练，采用的融合权重分配策略为：{strategy}")
    dataset = MultiDayDataset(num_samples=100)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = MultiDayFusionModel(seq_input_dim=12, static_input_dim=4, weight_strategy=strategy)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = PenaltyBiddingLoss(penalty_weight=10.0)
    
    model.train()
    for epoch in range(5):
        epoch_loss = 0
        for xs0, xs1, xs2, xst0, xst1, xst2, y, pmin, pmax in loader:
            optimizer.zero_grad()
            preds = model(xs0, xs1, xs2, xst0, xst1, xst2)
            loss = criterion(preds, y, pmin, pmax)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(loader):.4f}")
        
    # 测试评估时触发监控打印
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(loader))
        print(">> 预测一次输出以查看权重或形状：")
        _ = model(*test_batch[:6])

if __name__ == '__main__':
    # 分别测试两种权重配置的训练机制
    train_multi_day_model(strategy='decay')
    train_multi_day_model(strategy='learnable')