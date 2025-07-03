"""
损失函数模块 - 定义训练过程中使用的各种损失函数
包含数值稳定的MSE损失和不确定性感知损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableMSELoss(nn.Module):
    """数值稳定的MSE损失函数"""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # 检查并处理NaN值
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        # 计算MSE
        loss = (pred - target) ** 2

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class StableUncertaintyLoss(nn.Module):
    """数值稳定的不确定性感知损失函数"""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, mean, var, target):
        """数值稳定版本"""
        # 处理可能的NaN值
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        var = torch.where(torch.isnan(var), torch.ones_like(var), var)
        var = torch.where(var < 1e-6, torch.ones_like(var) * 1e-6, var)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        # 计算损失
        precision = 1 / (var)
        precision_errors = precision * (mean - target) ** 2

        # 负对数似然损失 + 方差正则化
        loss = 0.5 * (torch.log(var) + precision_errors)

        # 剪裁极端损失值
        loss = torch.clamp(loss, -1e3, 1e3)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class HuberLoss(nn.Module):
    """Huber损失 - 对异常值较为鲁棒"""

    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred, target):
        # 处理NaN值
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        # 计算误差
        error = pred - target
        abs_error = torch.abs(error)

        # Huber损失
        quadratic = torch.min(abs_error, torch.full_like(abs_error, self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class LogCoshLoss(nn.Module):
    """Log-Cosh损失 - 结合MSE和MAE的优点"""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # 处理NaN值
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        # 计算错误
        error = pred - target

        # Log-Cosh损失
        loss = torch.log(torch.cosh(error + 1e-12))

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class QuantileLoss(nn.Module):
    """分位数损失 - 用于分位数回归"""

    def __init__(self, quantile=0.5, reduction='mean'):
        super().__init__()
        self.quantile = quantile
        self.reduction = reduction

    def forward(self, pred, target):
        # 处理NaN值
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        # 计算误差
        error = target - pred

        # 分位数损失
        loss = torch.where(error >= 0,
                           self.quantile * error,
                           (self.quantile - 1) * error)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class FocalLoss(nn.Module):
    """Focal Loss - 处理不平衡回归问题"""

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        # 处理NaN值
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        # 计算基础损失
        mse_loss = (pred - target) ** 2

        # 计算调制因子
        pt = torch.exp(-mse_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # 应用focal权重
        loss = focal_weight * mse_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class CombinedLoss(nn.Module):
    """组合损失函数 - 结合多种损失的优点"""

    def __init__(self, losses_weights=None):
        super().__init__()

        # 默认损失函数和权重
        if losses_weights is None:
            losses_weights = {
                'mse': 1.0,
                'huber': 0.5,
                'logcosh': 0.3
            }

        self.losses = nn.ModuleDict()
        self.weights = {}

        for loss_name, weight in losses_weights.items():
            if loss_name == 'mse':
                self.losses[loss_name] = StableMSELoss()
            elif loss_name == 'huber':
                self.losses[loss_name] = HuberLoss()
            elif loss_name == 'logcosh':
                self.losses[loss_name] = LogCoshLoss()
            elif loss_name == 'quantile':
                self.losses[loss_name] = QuantileLoss()

            self.weights[loss_name] = weight

    def forward(self, pred, target):
        total_loss = 0
        for loss_name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            total_loss += self.weights[loss_name] * loss_value

        return total_loss


def get_loss_function(loss_type='mse', **kwargs):
    """
    损失函数工厂函数

    Args:
        loss_type: 损失函数类型
        **kwargs: 损失函数参数

    Returns:
        对应的损失函数实例
    """
    loss_functions = {
        'mse': StableMSELoss,
        'uncertainty': StableUncertaintyLoss,
        'huber': HuberLoss,
        'logcosh': LogCoshLoss,
        'quantile': QuantileLoss,
        'focal': FocalLoss,
        'combined': CombinedLoss
    }

    if loss_type not in loss_functions:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

    return loss_functions[loss_type](**kwargs)