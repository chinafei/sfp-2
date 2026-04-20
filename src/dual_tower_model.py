import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out, time_prior_weights=None):
        """
        rnn_out: [batch_size, seq_len, hidden_dim]
        time_prior_weights: [seq_len] 10:00-16:00 (40-64) 和 16:00-21:00 (64-84) 高峰时段的注意力先验
        """
        # [batch_size, seq_len]
        attn_weights = self.attention(rnn_out).squeeze(-1)
        
        # 融入先验权重指导模型在特定时段聚焦
        if time_prior_weights is not None:
            # Broadcast 到 batch
            attn_weights = attn_weights + time_prior_weights
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # [batch_size, seq_len, 1] * [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        context = torch.sum(attn_weights.unsqueeze(-1) * rnn_out, dim=1)
        return context, attn_weights

class DualTowerBiddingModel(nn.Module):
    def __init__(self, seq_input_dim, static_input_dim, hidden_dim=64):
        """
        双塔结构：
        - 路径A: GRU 处理 15 分钟级的序列数据 (96点)
        - 路径B: 全连接层处理日单点静态数据
        """
        super(DualTowerBiddingModel, self).__init__()
        
        # =========== 路径 A (时间序列特征) ===========
        # 双向 GRU 出来的维度是 hidden_dim * 2
        self.gru = nn.GRU(seq_input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = TemporalAttention(hidden_dim * 2)
        
        # 注意力先验 (96点)，根据要求，权重聚焦在 10:00-16:00 (40-64)，16:00-21:00 (64-84)
        self.register_buffer('time_prior', self._build_time_prior())
        
        # =========== 路径 B (单点/日粒度静态特征) ===========
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
        
        # =========== 融合与输出层 ===========
        # 合并路径A和B的输出，映射到 5 个时段 (T1-T5)
        # 每个时段预测 3 个值：容量, 边际价, 出清价 -> [batch, 5*3]
        concat_dim = (hidden_dim * 2) + (hidden_dim * 2)
        self.fc_fusion = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5 * 3) 
        )
        
    def _build_time_prior(self):
        # 长度为 96, 在关键时段施加强烈正偏置
        prior = torch.zeros(96)
        prior[40:64] += 2.0  # 光伏高峰 (10:00-16:00)
        prior[64:84] += 2.0  # 晚高峰 (16:00-21:00)
        return prior

    def forward(self, x_seq, x_static):
        """
        x_seq: (batch, 96, seq_input_dim)
        x_static: (batch, static_input_dim)
        返回: (未截断预测值, 注意力权重) 
        由于 PyTorch 自动回传梯度问题，不建议在算 loss 之前直接用 clamped/clipped 
        结果输出，因此这里返回未截断结果，用在 Loss 和后续的手动 clip 中。
        """
        # ================= 塔 A 计算 =================
        gru_out, _ = self.gru(x_seq)
        context_vec, attn_weights = self.attention(gru_out, self.time_prior.unsqueeze(0))
        
        # ================= 塔 B 计算 =================
        static_vec = self.static_net(x_static)
        
        # ================= 融合计算 =================
        fused = torch.cat([context_vec, static_vec], dim=1)
        out_flat = self.fc_fusion(fused)  # (batch, 15)
        out_raw = out_flat.view(-1, 5, 3) # (batch, 5, 3) 原生未截断预测：容量，边际价，出清价
            
        return out_raw, attn_weights
        
    def predict(self, x_seq, x_static, price_min, price_max):
        """
        推理/应用阶段使用：自动将价格相关的预测（指标1边际价、指标2出清价）clip 到所属时段的限价区间内。
        而容量(指标0)不受价格限价的影响。
        price_min/max: [batch, 5]
        """
        self.eval()
        with torch.no_grad():
            out_raw, _ = self.forward(x_seq, x_static)
            # out_raw shape: (batch, 5, 3)
            
            out_clipped = out_raw.clone()
            
            p_min_exp = price_min.unsqueeze(-1) # (batch, 5, 1)
            p_max_exp = price_max.unsqueeze(-1)
            
            # 对边际排序价和市场出清价进行截断 (index 1和2)
            out_clipped[..., 1:] = torch.max(torch.min(out_raw[..., 1:], p_max_exp), p_min_exp)
            
        return out_clipped


class PenaltyBiddingLoss(nn.Module):
    def __init__(self, penalty_weight=10.0):
        super(PenaltyBiddingLoss, self).__init__()
        # 对 T3 和 T4 时段的预测误差赋予 1.5 倍的权重 (即索引 2 和 3 为 1.5倍，其余为 1.0)
        self.register_buffer('segment_weights', torch.tensor([1.0, 1.0, 1.5, 1.5, 1.0]))
        self.penalty_weight = penalty_weight

    def forward(self, preds, targets, price_min, price_max):
        """
        preds: (batch, 5, 3) 未截断的模型输出：[容量, 边际价, 出清均价]
        targets: (batch, 5, 3) 真实标签
        price_min: (batch, 5) 时段下限
        price_max: (batch, 5) 时段上限 (T3/T4: 15, 其余: 10)
        """
        # 1. 带时段权重的基本 MSE Loss (对容量、边际价、出清价做整体 MSE)
        base_squared_error = (preds - targets) ** 2
        weighted_mse = (base_squared_error * self.segment_weights.view(1, 5, 1)).mean()
        
        # 2. 超出限价区间的严厉惩罚 (Penalty 项) - 仅对价格相关的特征(index 1 和 2)触发惩罚
        price_preds = preds[..., 1:] # 截取 边际价和出清价 (batch, 5, 2)
        p_min_exp = price_min.unsqueeze(2) # (batch, 5, 1)
        p_max_exp = price_max.unsqueeze(2) # (batch, 5, 1)
        
        # 预测价低于最低价的情况
        under_penalty = F.relu(p_min_exp - price_preds)
        # 预测价高于最高价的情况
        over_penalty = F.relu(price_preds - p_max_exp)
        
        # 按照要求：带惩罚项的损失函数，给予 10 倍惩罚
        penalty_loss = (under_penalty + over_penalty).mean() * self.penalty_weight
        
        # 最终自定义 Loss
        total_loss = weighted_mse + penalty_loss
        return total_loss

if __name__ == '__main__':
    # ================= 简易功能自测 =================
    batch_size = 8
    seq_features = 12       # 每日15分钟分辨率特征(负荷、新能源等)
    static_features = 4     # 每日单点约束特征(日检修量、备用等)
    
    # 初始化双塔模型和损失
    model = DualTowerBiddingModel(seq_input_dim=seq_features, static_input_dim=static_features)
    criterion = PenaltyBiddingLoss(penalty_weight=10.0)
    
    # 模拟输入
    x_s = torch.randn(batch_size, 96, seq_features)
    x_t = torch.randn(batch_size, static_features)
    y_true = torch.rand(batch_size, 5) * 10 + 5 # 标签
    
    # 限定 T3/T4 为 [10.0, 15.0]，其余 T1/T2/T5 为 [5.0, 10.0]
    p_min = torch.tensor([[5., 5., 10., 10., 5.]] * batch_size)
    p_max = torch.tensor([[10., 10., 15., 15., 10.]] * batch_size)
    
    # 训练场景 Forward 及 Loss 计算
    model.train()
    preds_raw, attns = model(x_s, x_t)
    loss = criterion(preds_raw, y_true, p_min, p_max)
    
    print("Train 原生预测值(带越界可能):\n", preds_raw.detach())
    print(f"Loss值: {loss.item():.4f}")
    print("注意力权重(截取最后几个步长评估):\n", attns[0, -10:].detach())
    
    # 推理场景 预测 (限制于 price_min 和 price_max)
    clamped_preds = model.predict(x_s, x_t, p_min, p_max)
    print("Inference 截断限制后的最终输出:\n", clamped_preds)