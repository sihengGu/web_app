"""
增强训练器模块 - 专门处理超级模型的训练
包含混合精度训练、梯度累积、TensorBoard集成等功能
"""

import os
import torch
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from loss import StableMSELoss, StableUncertaintyLoss
import logging


class UltraTrainer:
    """超级模型专用训练器"""

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

        # 训练相关属性
        self.scaler = None  # 混合精度训练
        self.tb_writer = None  # TensorBoard
        self.logger = logging.getLogger(__name__)

    def setup_training(self, learning_rate=1e-3, weight_decay=1e-4, loss_type='mse'):
        """设置训练组件"""
        # 选择损失函数
        if self.uncertainty_mode and loss_type == 'uncertainty':
            self.criterion = StableUncertaintyLoss()
        elif loss_type == 'stable_uncertainty':
            # 使用改进的稳定不确定性损失
            try:
                from improved_simple_model import StableUncertaintyLoss as ImprovedUncertaintyLoss
                self.criterion = ImprovedUncertaintyLoss(beta=1.0)
            except ImportError:
                self.criterion = StableUncertaintyLoss()
        elif loss_type == 'mse' or loss_type == 'stable_mse':
            # 使用稳定的MSE损失
            try:
                from improved_simple_model import StableMSELoss as ImprovedMSELoss
                self.criterion = ImprovedMSELoss()
            except ImportError:
                self.criterion = StableMSELoss()
        else:
            self.criterion = StableMSELoss()

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
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

    def setup_mixed_precision(self):
        """设置混合精度训练"""
        if self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("混合精度训练已启用")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_uncertainty = 0.0 if self.uncertainty_mode else None
        valid_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                # 混合精度前向传播
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        if self.uncertainty_mode:
                            mean, var = self.model(inputs)
                            loss = self.criterion(mean, var, targets)
                            if not torch.isnan(var).all():
                                total_uncertainty += torch.mean(var).detach().cpu().item()
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)

                    # 检查损失是否有效
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        self.logger.warning(f"跳过无效损失批次 {batch_idx}")
                        continue

                    # 混合精度反向传播
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 普通精度训练
                    if self.uncertainty_mode:
                        mean, var = self.model(inputs)
                        loss = self.criterion(mean, var, targets)
                        if not torch.isnan(var).all():
                            total_uncertainty += torch.mean(var).detach().cpu().item()
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)

                    # 检查损失是否有效
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        self.logger.warning(f"跳过无效损失批次 {batch_idx}")
                        continue

                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # 记录损失
                total_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

                # TensorBoard记录
                if self.tb_writer:
                    global_step = epoch * len(self.train_loader) + batch_idx
                    self.tb_writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.error(f"GPU内存不足，跳过批次 {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    self.logger.error(f"训练批次错误: {str(e)}")
                    continue

        # 计算平均损失
        avg_loss = total_loss / max(valid_batches, 1)
        avg_uncertainty = total_uncertainty / max(valid_batches, 1) if self.uncertainty_mode else None

        return avg_loss, avg_uncertainty

    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_uncertainty = 0.0 if self.uncertainty_mode else None
        valid_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Valid]")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                try:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # 混合精度前向传播
                    if self.scaler is not None:
                        with torch.cuda.amp.autocast():
                            if self.uncertainty_mode:
                                mean, var = self.model(inputs)
                                loss = self.criterion(mean, var, targets)
                                if not torch.isnan(var).all():
                                    total_uncertainty += torch.mean(var).detach().cpu().item()
                            else:
                                outputs = self.model(inputs)
                                loss = self.criterion(outputs, targets)
                    else:
                        if self.uncertainty_mode:
                            mean, var = self.model(inputs)
                            loss = self.criterion(mean, var, targets)
                            if not torch.isnan(var).all():
                                total_uncertainty += torch.mean(var).detach().cpu().item()
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)

                    # 检查损失是否有效
                    if not torch.isnan(loss).any() and not torch.isinf(loss).any():
                        total_loss += loss.item()
                        valid_batches += 1

                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger.error(f"GPU内存不足，跳过验证批次 {batch_idx}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        self.logger.error(f"验证批次错误: {str(e)}")
                        continue

        # 计算平均损失
        avg_loss = total_loss / max(valid_batches, 1)
        avg_uncertainty = total_uncertainty / max(valid_batches, 1) if self.uncertainty_mode else None

        return avg_loss, avg_uncertainty

    def train(self, epochs=500, patience=50, save_path='best_model.pth'):
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

        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_uncertainty = self.train_epoch(epoch)

            # 验证
            val_loss, val_uncertainty = self.validate_epoch(epoch)

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
            epoch_msg = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            if self.uncertainty_mode and train_uncertainty is not None and val_uncertainty is not None:
                epoch_msg += f" | Train Unc: {train_uncertainty:.4f} | Val Unc: {val_uncertainty:.4f}"
            print(epoch_msg)

            # TensorBoard记录
            if self.tb_writer:
                self.tb_writer.add_scalar('Train/EpochLoss', train_loss, epoch)
                self.tb_writer.add_scalar('Validation/EpochLoss', val_loss, epoch)
                self.tb_writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
                if self.uncertainty_mode:
                    if train_uncertainty is not None:
                        self.tb_writer.add_scalar('Train/Uncertainty', train_uncertainty, epoch)
                    if val_uncertainty is not None:
                        self.tb_writer.add_scalar('Validation/Uncertainty', val_uncertainty, epoch)

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
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


# 替换原有的train_model函数
def train_ultra_model(model, train_loader, val_loader, epochs=500, patience=50, learning_rate=1e-5, use_mixed_precision=True):
    """训练超级模型的便捷函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建训练器
    trainer = UltraTrainer(model, train_loader, val_loader, device)
    trainer.setup_training(learning_rate=learning_rate)

    # 设置混合精度
    if use_mixed_precision:
        trainer.setup_mixed_precision()

    # 开始训练
    history = trainer.train(epochs=epochs, patience=patience)

    return model, history