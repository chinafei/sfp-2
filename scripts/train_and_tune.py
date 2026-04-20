import os
import sys
import argparse
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import optuna

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dual_tower_model import DualTowerBiddingModel, PenaltyBiddingLoss
from src.data_loader import NPYDataLoader, save_model_params, save_normalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data():
    """从 NPY 数据集加载训练数据"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sfp2_dir   = os.path.dirname(script_dir)
    npy_base   = os.path.join(sfp2_dir, "datasets", "npy")

    loader = NPYDataLoader(npy_base)
    train_ds, val_ds, seq_dim, static_dim = loader.load_training_data()
    if train_ds is None:
        logging.error("无法加载真实数据，请先运行 --build 构建 NPY 数据集")
        sys.exit(1)
    return loader, train_ds, val_ds, seq_dim, static_dim

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

def objective(trial, train_ds, val_ds, seq_dim, static_dim):
    """Optuna 超参数调参目标函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    penalty_weight = trial.suggest_float("penalty_weight", 1.0, 20.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = DualTowerBiddingModel(
        seq_input_dim=seq_dim, static_input_dim=static_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = PenaltyBiddingLoss(penalty_weight=penalty_weight).to(device)

    val_loss = train_and_validate(
        model, train_loader, val_loader, optimizer, criterion,
        epochs=20, device=device,
    )
    return val_loss


def main():
    parser = argparse.ArgumentParser(description="SFP-2 模型训练")
    parser.add_argument("--epochs",      type=int, default=50, help="全量训练 epoch 数")
    parser.add_argument("--patience",    type=int, default=5,  help="Early stopping patience")
    parser.add_argument("--tune_trials", type=int, default=15, help="Optuna 调参 trial 数")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 加载数据
    logging.info("加载训练数据...")
    loader, train_ds, val_ds, seq_dim, static_dim = get_data()

    # ── Optuna 调参 ──────────────────────────────────────
    logging.info(f"开始 Optuna 调参 ({args.tune_trials} trials)...")
    study = optuna.create_study(
        direction="minimize",
        study_name="sfp-2_dual_tower_tuning",
    )
    study.optimize(
        lambda t: objective(t, train_ds, val_ds, seq_dim, static_dim),
        n_trials=args.tune_trials,
    )

    logging.info("=== 调参完成 ===")
    logging.info(f"Best Val Loss: {study.best_value:.4f}")
    logging.info(f"Best Params: {study.best_params}")
    best_params = study.best_params

    # 保存 best_params（含 hidden_dim）
    save_model_params(results_dir, best_params)
    save_normalizer(results_dir, loader)

    # ── 全量训练 ─────────────────────────────────────────
    logging.info("开始全量训练...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = ConcatDataset([train_ds, val_ds])
    full_loader = DataLoader(
        full_dataset,
        batch_size=best_params["batch_size"],
        shuffle=True,
    )

    final_model = DualTowerBiddingModel(
        seq_input_dim=seq_dim, static_input_dim=static_dim,
        hidden_dim=best_params["hidden_dim"],
    ).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params["lr"])
    criterion = PenaltyBiddingLoss(
        penalty_weight=best_params["penalty_weight"],
    ).to(device)

    target_scales = torch.tensor(
        [4000.0, 10.0, 10.0], dtype=torch.float32,
    ).to(device)

    best_loss = float("inf")
    best_model_path = os.path.join(results_dir, "best_dual_tower_model.pt")
    patience_counter = 0

    final_model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for xs, xst, yt, pmin, pmax in full_loader:
            xs, xst = xs.to(device), xst.to(device)
            yt, pmin, pmax = yt.to(device), pmin.to(device), pmax.to(device)

            optimizer.zero_grad()
            preds_scaled, _ = final_model(xs, xst)

            yt_scaled    = yt / target_scales.view(1, 1, 3)
            pmin_scaled  = pmin / 10.0
            pmax_scaled  = pmax / 10.0

            loss = criterion(preds_scaled, yt_scaled, pmin_scaled, pmax_scaled)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(full_loader)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(final_model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(
                f"Epoch {epoch+1:02d}/{args.epochs} - Loss: {epoch_loss:.4f}"
            )

        if patience_counter >= args.patience:
            logging.info(
                f"Early stopping at epoch {epoch+1} (patience={args.patience})"
            )
            break

    logging.info(f"全量训练完成。最佳模型存至: {best_model_path}")


if __name__ == "__main__":
    main()
