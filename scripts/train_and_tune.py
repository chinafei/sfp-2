import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import logging

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dual_tower_model import DualTowerBiddingModel, PenaltyBiddingLoss
from src.real_data_loader import load_real_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BiddingDataset(Dataset):
    def __init__(self, x_seq, x_static, y_targets, p_min, p_max):
        """
        x_seq: (N, 96, seq_dim) - 每日15分钟序列特征
        x_static: (N, static_dim) - 每日单点约束特征
        y_targets: (N, 5) - T1~T5 出清目标
        p_min, p_max: (N, 5) - 对应限价区间
        """
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.x_static = torch.tensor(x_static, dtype=torch.float32)
        self.y_targets = torch.tensor(y_targets, dtype=torch.float32)
        self.p_min = torch.tensor(p_min, dtype=torch.float32)
        self.p_max = torch.tensor(p_max, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y_targets)
        
    def __getitem__(self, idx):
        return self.x_seq[idx], self.x_static[idx], self.y_targets[idx], self.p_min[idx], self.p_max[idx]

def load_data():
    """
    根据实际数据情况加载数据。这里使用模拟数据演示工程全流程。
    实际使用时请替换为通过 np.load('.../datasets/npy/x_seq.npy') 加载真实工程特征。
    """
    logging.info("Loading dataset...")
    # 模拟 1000 天的数据
    num_samples = 1000
    seq_dim = 12       # 每日序列特征厚度
    static_dim = 4     # 每日静态特征厚度
    
    x_seq = np.random.randn(num_samples, 96, seq_dim)
    x_static = np.random.randn(num_samples, static_dim)
    
    # 模拟目标价格和容量 [容量, 边际价, 出清均价]
    y_targets = np.zeros((num_samples, 5, 3))
    y_targets[..., 0] = np.random.rand(num_samples, 5) * 1000 + 3000   # 容量约 3000-4000
    y_targets[..., 1] = np.random.rand(num_samples, 5) * 20 + 5        # 边际价约 5-25
    y_targets[..., 2] = np.random.rand(num_samples, 5) * 10 + 5        # 出清价约 5-15
    
    # 构造限价约束 T3,T4:[10,15], 其余:[5,10]
    p_min = np.tile([5., 5., 10., 10., 5.], (num_samples, 1))
    p_max = np.tile([10., 10., 15., 15., 10.], (num_samples, 1))
    
    # 简易划分 Train / Val (80:20)
    split_idx = int(num_samples * 0.8)
    
    train_dataset = BiddingDataset(x_seq[:split_idx], x_static[:split_idx], y_targets[:split_idx], p_min[:split_idx], p_max[:split_idx])
    val_dataset = BiddingDataset(x_seq[split_idx:], x_static[split_idx:], y_targets[split_idx:], p_min[split_idx:], p_max[split_idx:])
    
    return train_dataset, val_dataset, seq_dim, static_dim

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs=50, device='cpu'):
    best_val_loss = float('inf')
    early_stop_patience = 5
    patience_counter = 0
    target_scales = torch.tensor([4000.0, 10.0, 10.0], dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xs, xst, yt, pmin, pmax in train_loader:
            xs, xst = xs.to(device), xst.to(device)
            yt, pmin, pmax = yt.to(device), pmin.to(device), pmax.to(device)
            
            optimizer.zero_grad()
            preds_scaled, _ = model(xs, xst)
            
            yt_scaled = yt / target_scales.view(1, 1, 3)
            pmin_scaled = pmin / 10.0
            pmax_scaled = pmax / 10.0
            
            loss = criterion(preds_scaled, yt_scaled, pmin_scaled, pmax_scaled)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xs, xst, yt, pmin, pmax in val_loader:
                xs, xst = xs.to(device), xst.to(device)
                yt, pmin, pmax = yt.to(device), pmin.to(device), pmax.to(device)
                
                preds_scaled, _ = model(xs, xst)
                
                yt_scaled = yt / target_scales.view(1, 1, 3)
                pmin_scaled = pmin / 10.0
                pmax_scaled = pmax / 10.0
                
                loss = criterion(preds_scaled, yt_scaled, pmin_scaled, pmax_scaled)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break
                
    return best_val_loss

def objective(trial):
    """ Optuna Hyperparameter Tuning Objective """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    train_dataset, val_dataset, seq_dim, static_dim = load_real_data(workspace_dir)
    
    if train_dataset is None:
        train_dataset, val_dataset, seq_dim, static_dim = load_data()
    
    # 1. 自动调优超参数定义
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    penalty_weight = trial.suggest_float("penalty_weight", 1.0, 20.0)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. 实例化模型与 Loss
    model = DualTowerBiddingModel(seq_input_dim=seq_dim, static_input_dim=static_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = PenaltyBiddingLoss(penalty_weight=penalty_weight).to(device)
    
    # 3. 执行训练验证 (设定少量 epoch 以加速调参搜索)
    val_loss = train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs=20, device=device)
    
    return val_loss

if __name__ == '__main__':
    logging.info("Starting Automatic Hyperparameter Tuning with Optuna...")
    # 创建 Optuna 学习器
    study = optuna.create_study(direction="minimize", study_name="sfp-2_dual_tower_tuning")
    
    # 执行 15 次试验寻找最佳超参数
    study.optimize(objective, n_trials=15)
    
    logging.info("=== Tuning Completed ===")
    logging.info(f"Best Trial Score (Val Loss): {study.best_value:.4f}")
    logging.info(f"Best Hyperparameters: {study.best_params}")
    
    # 这里可以利用 study.best_params 开启最终的大 Epoch 训练全量保存模型
    best_params = study.best_params
    logging.info("Proceed to Full Training with Best Parameters...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    train_dataset, val_dataset, seq_dim, static_dim = load_real_data(workspace_dir)
    
    if train_dataset is None:
        train_dataset, val_dataset, seq_dim, static_dim = load_data()
    
    # 将训练集与验证集合并为全量数据
    from torch.utils.data import ConcatDataset
    full_dataset = ConcatDataset([train_dataset, val_dataset])
    full_loader = DataLoader(full_dataset, batch_size=best_params['batch_size'], shuffle=True)
    
    # 实例化最优模型
    final_model = DualTowerBiddingModel(seq_input_dim=seq_dim, static_input_dim=static_dim, hidden_dim=best_params['hidden_dim']).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    criterion = PenaltyBiddingLoss(penalty_weight=best_params['penalty_weight']).to(device)
    
    target_scales = torch.tensor([4000.0, 10.0, 10.0], dtype=torch.float32).to(device)
    final_epochs = 50
    best_loss = float('inf')
    
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_dual_tower_model.pt')
    
    final_model.train()
    for epoch in range(final_epochs):
        epoch_loss = 0.0
        for xs, xst, yt, pmin, pmax in full_loader:
            xs, xst = xs.to(device), xst.to(device)
            yt, pmin, pmax = yt.to(device), pmin.to(device), pmax.to(device)
            
            optimizer.zero_grad()
            preds_scaled, _ = final_model(xs, xst)
            
            yt_scaled = yt / target_scales.view(1, 1, 3)
            pmin_scaled = pmin / 10.0
            pmax_scaled = pmax / 10.0
            
            loss = criterion(preds_scaled, yt_scaled, pmin_scaled, pmax_scaled)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(full_loader)
        
        # 简单保存最优的全量训练状态
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(final_model.state_dict(), best_model_path)
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(f"Full Train Epoch {epoch+1:02d}/{final_epochs} - Loss: {epoch_loss:.4f}")
            
    logging.info(f"Full training finished. Best model saved to {best_model_path}")
