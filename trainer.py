"""
训练模块 - 包含模型训练的主要逻辑
支持不确定性估计、早停、学习率调度等功能
"""

import os
import torch
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from loss import StableMSELoss, StableUncertaintyLoss
import pickle
from dataset import MultiResistivityDataset


def train_model(model, train_loader, val_loader, epochs=200, patience=20, learning_rate=1e-3):
    """优化的训练函数 - 修复NaN问题"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)

    # 确定模型是否估计不确定性
    uncertainty_mode = hasattr(model, 'estimate_uncertainty') and model.estimate_uncertainty

    # 选择损失函数
    if uncertainty_mode:
        criterion = StableUncertaintyLoss()
    else:
        criterion = StableMSELoss()

    # 优化器 - 使用更保守的学习率和权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 学习率调度器 - 使用更温和的调度策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience // 2,
        verbose=True,
        min_lr=1e-6
    )

    # 初始化训练记录
    best_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_uncertainty': [] if uncertainty_mode else None,
        'val_uncertainty': [] if uncertainty_mode else None,
        'learning_rates': []
    }

    # 训练循环
    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        train_uncertainty = 0.0 if uncertainty_mode else None

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for inputs, targets in pbar:
            try:
                inputs, targets = inputs.to(device), targets.to(device)

                # 前向传播
                if uncertainty_mode:
                    mean, var = model(inputs)
                    loss = criterion(mean, var, targets)
                    if not torch.isnan(var).all():  # 只在值不全是NaN时累加
                        train_uncertainty += torch.mean(var).detach().cpu().item()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # 如果损失是NaN，跳过此批次
                if torch.isnan(loss).any():
                    print("警告: 发现NaN损失，跳过此批次")
                    continue

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪（重要！防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 记录损失
                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            except Exception as e:
                print(f"训练批次处理出错: {str(e)}")
                continue

        # 计算平均损失
        train_loss /= len(train_loader)
        if uncertainty_mode and train_uncertainty is not None:
            train_uncertainty /= len(train_loader)

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        val_uncertainty = 0.0 if uncertainty_mode else None

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Valid]"):
                try:
                    inputs, targets = inputs.to(device), targets.to(device)

                    if uncertainty_mode:
                        mean, var = model(inputs)
                        loss = criterion(mean, var, targets)
                        if not torch.isnan(var).all():
                            val_uncertainty += torch.mean(var).detach().cpu().item()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    # 记录损失
                    if not torch.isnan(loss).any():
                        val_loss += loss.item()

                except Exception as e:
                    print(f"验证批次处理出错: {str(e)}")
                    continue

        # 计算平均验证损失
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        if uncertainty_mode and val_uncertainty is not None:
            val_uncertainty /= len(val_loader) if len(val_loader) > 0 else 1

        # 记录损失
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        if uncertainty_mode:
            if train_uncertainty is not None:
                history['train_uncertainty'].append(train_uncertainty)
            if val_uncertainty is not None:
                history['val_uncertainty'].append(val_uncertainty)

        # 输出当前训练状态
        epoch_msg = f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        if uncertainty_mode:
            if train_uncertainty is not None and val_uncertainty is not None:
                epoch_msg += f" | Train Uncertainty: {train_uncertainty:.4f} | Val Uncertainty: {val_uncertainty:.4f}"
        print(epoch_msg)

        # 更新学习率
        scheduler.step(val_loss)

        # 早停机制 - 只在验证损失是有效值时才启用
        if not np.isnan(val_loss) and not np.isinf(val_loss):
            if val_loss < best_loss:
                best_loss = val_loss
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': history,
                }, 'best_model.pth')

                patience_counter = 0
                print(f"模型已保存! 验证损失改进至 {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"验证损失未改进。耐心计数: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"早停! 最佳验证损失: {best_loss:.4f} (Epoch {epoch + 1 - patience})")
                    break

    # 加载最佳模型 (如果存在)
    best_model_path = 'best_model.pth'
    if os.path.exists(best_model_path):
        print(f"加载最佳模型: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"未找到最佳模型文件，使用当前模型状态")
        # 保存最终模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'history': history,
        }, 'final_model.pth')

    return model, history


class Trainer:
    """训练器类 - 封装训练逻辑的面向对象版本"""

    def __init__(self, model, train_loader, val_loader, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # 确定模型是否估计不确定性
        self.uncertainty_mode = hasattr(model, 'estimate_uncertainty') and model.estimate_uncertainty

    def setup_training(self, learning_rate=1e-3, weight_decay=1e-4, loss_type='mse'):
        """设置训练组件"""
        # 选择损失函数
        if self.uncertainty_mode:
            self.criterion = StableUncertaintyLoss()
        else:
            self.criterion = StableMSELoss()

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
        )

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_uncertainty = 0.0 if self.uncertainty_mode else None

        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, targets in pbar:
            try:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 前向传播
                if self.uncertainty_mode:
                    mean, var = self.model(inputs)
                    loss = self.criterion(mean, var, targets)
                    if not torch.isnan(var).all():
                        total_uncertainty += torch.mean(var).detach().cpu().item()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # 跳过NaN损失
                if torch.isnan(loss).any():
                    continue

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            except Exception as e:
                print(f"训练批次错误: {str(e)}")
                continue

        avg_loss = total_loss / len(self.train_loader)
        avg_uncertainty = total_uncertainty / len(self.train_loader) if self.uncertainty_mode else None

        return avg_loss, avg_uncertainty

    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_uncertainty = 0.0 if self.uncertainty_mode else None

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                try:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    if self.uncertainty_mode:
                        mean, var = self.model(inputs)
                        loss = self.criterion(mean, var, targets)
                        if not torch.isnan(var).all():
                            total_uncertainty += torch.mean(var).detach().cpu().item()
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)

                    if not torch.isnan(loss).any():
                        total_loss += loss.item()

                except Exception as e:
                    print(f"验证批次错误: {str(e)}")
                    continue

        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else float('inf')
        avg_uncertainty = total_uncertainty / len(self.val_loader) if self.uncertainty_mode else None

        return avg_loss, avg_uncertainty

    def train(self, epochs=200, patience=20, save_path='best_model.pth'):
        """完整的训练流程"""
        best_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_uncertainty': [] if self.uncertainty_mode else None,
            'val_uncertainty': [] if self.uncertainty_mode else None,
            'learning_rates': []
        }

        for epoch in range(epochs):
            # 训练
            train_loss, train_uncertainty = self.train_epoch()

            # 验证
            val_loss, val_uncertainty = self.validate_epoch()

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            if self.uncertainty_mode:
                if train_uncertainty is not None:
                    history['train_uncertainty'].append(train_uncertainty)
                if val_uncertainty is not None:
                    history['val_uncertainty'].append(val_uncertainty)

            # 打印进度
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 学习率调整
            self.scheduler.step(val_loss)

            # 早停检查
            if not np.isnan(val_loss) and not np.isinf(val_loss):
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint(save_path, epoch, val_loss, history)
                    patience_counter = 0
                    print(f"模型保存! 验证损失: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停! 最佳损失: {best_loss:.4f}")
                        break

        # 加载最佳模型
        if os.path.exists(save_path):
            self.load_checkpoint(save_path)

        return history

    def save_checkpoint(self, path, epoch, val_loss, history):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': history,
        }, path)




    def load_checkpoint(self, path):
        """Load checkpoint safely"""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None

    def safe_load_checkpoint(path, device=None):
        """Safe load checkpoint with proper error handling"""
        try:
            # For newer PyTorch versions
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            return checkpoint
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None