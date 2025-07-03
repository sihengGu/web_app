"""
 基于原始SimpleMultiScaleModel的渐进式优化
只添加最有效的改进，保持模型简单性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedSimpleModel(nn.Module):
    """简单模型 - 渐进式优化"""

    def __init__(self,
                 input_dim=7,
                 hidden_dim=64,
                 dropout=0.1,
                 estimate_uncertainty=False,
                 use_attention=False,
                 use_residual=False):
        super().__init__()

        self.estimate_uncertainty = estimate_uncertainty
        self.use_attention = use_attention
        self.use_residual = use_residual

        # 输入映射层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

        # 双向LSTM层 - 保持原来的架构
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 可选的简单注意力机制
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            )

        # 全局上下文表示
        context_input_dim = hidden_dim * 2
        if use_residual:
            # 添加残差连接
            self.residual_layer = nn.Linear(hidden_dim, hidden_dim * 2)
            context_input_dim = hidden_dim * 2

        self.context_layer = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )

        # 输出层
        output_dim = 2 if estimate_uncertainty else 1
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        # 输入映射
        x_proj = self.input_projection(x)
        x_proj = self.activation(x_proj)
        x_proj = self.dropout(x_proj)

        # 残差连接（可选）
        if self.use_residual:
            residual = self.residual_layer(x_proj)

        # LSTM处理
        lstm_out, _ = self.lstm(x_proj)

        # 注意力机制（可选）
        if self.use_attention:
            attention_weights = self.attention(lstm_out)
            global_context = torch.sum(lstm_out * attention_weights, dim=1)
        else:
            # 全局上下文 - 使用平均池化
            global_context = torch.mean(lstm_out, dim=1)

        # 添加残差连接
        if self.use_residual:
            global_context = global_context + torch.mean(residual, dim=1)

        # 上下文处理
        global_context = self.context_layer(global_context)

        # 回归输出
        output = self.regression_head(global_context)

        if self.estimate_uncertainty:
            # 输出均值和方差估计
            mean, log_var = output.chunk(2, dim=1)
            # 确保方差为正 - 使用softplus但限制范围
            var = F.softplus(log_var) + 1e-6
            return mean, var
        else:
            # 仅输出均值
            return output


class EnhancedSimpleModel(nn.Module):
    """添加更多有效改进"""

    def __init__(self,
                 input_dim=7,
                 hidden_dim=64,
                 dropout=0.1,
                 estimate_uncertainty=False):
        super().__init__()

        self.estimate_uncertainty = estimate_uncertainty

        # 输入投影 - 添加层归一化
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )

        # 多尺度LSTM - 使用不同的隐藏维度
        self.lstm_small = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.lstm_large = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 特征融合
        fusion_dim = hidden_dim + hidden_dim * 2  # small + large LSTM outputs
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )

        # 简单的自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 最终分类器
        final_input_dim = hidden_dim * 2  # avg + max pooling
        output_dim = 2 if estimate_uncertainty else 1

        self.final_classifier = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)

        # 多尺度LSTM
        lstm_small_out, _ = self.lstm_small(x)  # (batch, seq_len, hidden_dim)
        lstm_large_out, _ = self.lstm_large(x)  # (batch, seq_len, hidden_dim*2)

        # 特征融合
        fused_features = torch.cat([lstm_small_out, lstm_large_out], dim=-1)
        fused_features = self.feature_fusion(fused_features)

        # 自注意力
        attn_out, _ = self.self_attention(fused_features, fused_features, fused_features)
        attn_out = attn_out + fused_features  # 残差连接

        # 全局池化
        # 转置以适配池化层
        attn_out_t = attn_out.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        avg_pool = self.global_avg_pool(attn_out_t).squeeze(-1)  # (batch, hidden_dim)
        max_pool = self.global_max_pool(attn_out_t).squeeze(-1)  # (batch, hidden_dim)

        # 拼接池化特征
        global_features = torch.cat([avg_pool, max_pool], dim=1)

        # 最终输出
        output = self.final_classifier(global_features)

        if self.estimate_uncertainty:
            mean, log_var = output.chunk(2, dim=1)
            var = F.softplus(log_var) + 1e-6
            return mean, var
        else:
            return output


# 稳定的损失函数
class StableMSELoss(nn.Module):
    """完全稳定的MSE损失函数"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # 确保输入是有效的
        pred = torch.clamp(pred, -1e6, 1e6)
        target = torch.clamp(target, -1e6, 1e6)

        # 处理NaN值
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        # 计算MSE
        loss = F.mse_loss(pred, target)

        # 确保损失是正数
        return torch.abs(loss)


class StableUncertaintyLoss(nn.Module):
    """改进的不确定性损失函数"""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, mean, var, target):
        # 清理输入
        mean = torch.clamp(mean, -1e6, 1e6)
        var = torch.clamp(var, 1e-6, 1e6)
        target = torch.clamp(target, -1e6, 1e6)

        # 处理NaN
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        var = torch.where(torch.isnan(var), torch.ones_like(var), var)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        # 计算损失
        precision = 1.0 / var
        mse_term = precision * (mean - target) ** 2
        regularization_term = torch.log(var)

        # 组合损失
        loss = 0.5 * (mse_term + self.beta * regularization_term)

        # 取平均并确保为正
        return torch.mean(torch.abs(loss))


# 工厂函数
def create_improved_simple_model(model_type='improved', **kwargs):
    """创建改进的简单模型"""
    if model_type == 'improved':
        return ImprovedSimpleModel(**kwargs)
    elif model_type == 'enhanced':
        return EnhancedSimpleModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_improved_loss_function(loss_type='mse', **kwargs):
    """获取改进的损失函数"""
    if loss_type == 'mse':
        return StableMSELoss()
    elif loss_type == 'uncertainty':
        return StableUncertaintyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# 模型配置
IMPROVED_MODEL_CONFIGS = {
    'baseline': {
        'hidden_dim': 64,
        'dropout': 0.1,
        'estimate_uncertainty': False,
        'use_attention': False,
        'use_residual': False
    },
    'with_attention': {
        'hidden_dim': 64,
        'dropout': 0.1,
        'estimate_uncertainty': False,
        'use_attention': True,
        'use_residual': False
    },
    'with_residual': {
        'hidden_dim': 64,
        'dropout': 0.1,
        'estimate_uncertainty': False,
        'use_attention': False,
        'use_residual': True
    },
    'full_improved': {
        'hidden_dim': 64,
        'dropout': 0.1,
        'estimate_uncertainty': True,
        'use_attention': True,
        'use_residual': True
    },
    'enhanced': {
        'hidden_dim': 64,
        'dropout': 0.1,
        'estimate_uncertainty': True
    }
}