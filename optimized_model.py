"""
优化版模型架构 - 适中复杂度的电阻率预测模型
在简单模型基础上适度增加先进技术，避免过拟合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock(nn.Module):
    """轻量级Squeeze-and-Excitation块"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, max(channels // reduction, 4))
        self.fc2 = nn.Linear(max(channels // reduction, 4), channels)

    def forward(self, x):
        b, c, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = torch.sigmoid(self.fc2(F.relu(self.fc1(y)))).view(b, c, 1)
        return x * y


class MultiScaleConvBlock(nn.Module):
    """多尺度卷积块 - 轻量级版本"""

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()

        # 三个不同卷积核的分支
        self.conv3 = nn.Conv1d(in_channels, out_channels // 3, 3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 3, 5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels // 3, 7, padding=3)

        # 批标准化和激活
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # SE注意力
        self.se = SEBlock(out_channels)

        # 残差连接
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        # 多尺度卷积
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)

        # 拼接
        out = torch.cat([out3, out5, out7], dim=1)
        out = F.relu(self.bn(out))
        out = self.dropout(out)

        # SE注意力
        out = self.se(out)

        # 残差连接
        out = out + residual

        return F.relu(out)


class LightAttentionBlock(nn.Module):
    """轻量级注意力块"""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        residual = x

        # Multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        # Output projection and residual connection
        output = self.w_o(attn_output)
        output = self.norm(output + residual)

        return output


class OptimizedResistivityModel(nn.Module):
    """优化版电阻率预测模型 - 适中复杂度"""

    def __init__(self,
                 input_dim=7,
                 hidden_dim=128,
                 num_conv_blocks=2,
                 num_attention_blocks=2,
                 n_heads=4,
                 dropout=0.1,
                 estimate_uncertainty=True):
        super().__init__()

        self.estimate_uncertainty = estimate_uncertainty

        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 多尺度卷积块
        self.conv_blocks = nn.ModuleList([
            MultiScaleConvBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(num_conv_blocks)
        ])

        # 位置编码（简化版）
        self.pos_embedding = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)

        # 轻量级注意力块
        self.attention_blocks = nn.ModuleList([
            LightAttentionBlock(hidden_dim, n_heads)
            for _ in range(num_attention_blocks)
        ])

        # 双向LSTM（单层）
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # 单层不需要dropout
        )

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 特征融合
        fusion_dim = hidden_dim * 3  # conv + attention + lstm
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2)
        )

        # 输出头
        if estimate_uncertainty:
            self.mean_head = nn.Linear(hidden_dim // 2, 1)
            self.var_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()
            )
        else:
            self.output_head = nn.Linear(hidden_dim // 2, 1)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 1. 输入投影
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)

        # 2. 卷积特征提取
        x_conv = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        for conv_block in self.conv_blocks:
            x_conv = conv_block(x_conv)

        # 全局池化得到卷积特征
        conv_features = torch.cat([
            self.global_avg_pool(x_conv).squeeze(-1),
            self.global_max_pool(x_conv).squeeze(-1)
        ], dim=1)  # (batch, hidden_dim * 2)

        # 3. 注意力特征提取
        x_attn = x_conv.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        # 加入位置编码
        if x_attn.size(1) <= self.pos_embedding.size(1):
            x_attn = x_attn + self.pos_embedding[:, :x_attn.size(1), :]

        # 应用注意力块
        for attn_block in self.attention_blocks:
            x_attn = attn_block(x_attn)

        # 全局平均池化得到注意力特征
        attn_features = torch.mean(x_attn, dim=1)  # (batch, hidden_dim)

        # 4. LSTM特征提取
        lstm_out, _ = self.lstm(x_attn)
        lstm_features = torch.mean(lstm_out, dim=1)  # (batch, hidden_dim)

        # 5. 特征融合
        # 将卷积特征降维后拼接
        conv_reduced = conv_features[:, :attn_features.size(1)]  # 取前hidden_dim维
        fused_features = torch.cat([conv_reduced, attn_features, lstm_features], dim=1)

        # 6. 最终预测
        features = self.feature_fusion(fused_features)

        if self.estimate_uncertainty:
            mean = self.mean_head(features)
            var = self.var_head(features) + 1e-6
            return mean, var
        else:
            output = self.output_head(features)
            return output


class EnsembleOptimizedModel(nn.Module):
    """优化模型的集成版本"""

    def __init__(self, model_configs, ensemble_method='mean'):
        super().__init__()
        self.models = nn.ModuleList([
            OptimizedResistivityModel(**config) for config in model_configs
        ])
        self.ensemble_method = ensemble_method

        # 可学习的集成权重
        if ensemble_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(len(model_configs)))

    def forward(self, x):
        outputs = []
        uncertainties = []

        for model in self.models:
            if hasattr(model, 'estimate_uncertainty') and model.estimate_uncertainty:
                mean, var = model(x)
                outputs.append(mean)
                uncertainties.append(var)
            else:
                output = model(x)
                outputs.append(output)

        # 集成预测
        if self.ensemble_method == 'mean':
            final_output = torch.stack(outputs).mean(dim=0)
        elif self.ensemble_method == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            final_output = sum(w * out for w, out in zip(weights, outputs))

        if uncertainties:
            if self.ensemble_method == 'mean':
                final_uncertainty = torch.stack(uncertainties).mean(dim=0)
            else:
                weights = F.softmax(self.weights, dim=0)
                final_uncertainty = sum(w * unc for w, unc in zip(weights, uncertainties))
            return final_output, final_uncertainty

        return final_output


# 预定义配置
def get_optimized_configs():
    """获取优化模型的不同配置"""
    configs = {
        'light': {
            'input_dim': 7,
            'hidden_dim': 64,
            'num_conv_blocks': 1,
            'num_attention_blocks': 1,
            'n_heads': 4,
            'dropout': 0.1,
            'estimate_uncertainty': True
        },
        'medium': {
            'input_dim': 7,
            'hidden_dim': 96,
            'num_conv_blocks': 2,
            'num_attention_blocks': 2,
            'n_heads': 4,
            'dropout': 0.1,
            'estimate_uncertainty': True
        },
        'strong': {
            'input_dim': 7,
            'hidden_dim': 128,
            'num_conv_blocks': 3,
            'num_attention_blocks': 2,
            'n_heads': 8,
            'dropout': 0.15,
            'estimate_uncertainty': True
        }
    }
    return configs


def create_optimized_model(complexity='medium', ensemble=False):
    """创建优化模型"""
    configs = get_optimized_configs()

    if ensemble:
        # 创建不同配置的集成模型
        ensemble_configs = [configs['light'], configs['medium']]
        return EnsembleOptimizedModel(ensemble_configs)
    else:
        return OptimizedResistivityModel(**configs[complexity])


# 模型分析函数
def analyze_optimized_model_complexity(model):
    """分析优化模型的复杂度"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024,
        'complexity_level': 'Low' if total_params < 1e5 else 'Medium' if total_params < 1e6 else 'High'
    }