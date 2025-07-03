"""
模型架构 - 基于深度学习的电阻率预测模型
包含注意力机制、残差连接、特征金字塔、Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码 - 用于Transformer架构"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2) 计算注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 3) 拼接多头并进行线性变换
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        return self.w_o(attn_output), attn_weights

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class SpatialAttention(nn.Module):
    """空间注意力机制 - 关注重要的深度位置"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv1d(in_channels // 8, 1, 1)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))
        return x * attention


class ChannelAttention(nn.Module):
    """通道注意力机制 - 关注重要的特征通道"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        attention = torch.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * attention


class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """增强的残差块，包含多种注意力机制"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 注意力机制
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)
        self.se_block = SEBlock(out_channels)

        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # 应用多种注意力机制
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        out = self.se_block(out)

        out += residual
        out = F.relu(out)

        return out


class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络 - 多尺度特征融合"""

    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            inner_block = nn.Conv1d(in_channels, out_channels, 1)
            layer_block = nn.Conv1d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

    def forward(self, x_list):
        last_inner = self.inner_blocks[-1](x_list[-1])
        results = [self.layer_blocks[-1](last_inner)]

        for idx in range(len(x_list) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x_list[idx])
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-1], mode='linear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        return results


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class WaveletTransform(nn.Module):
    """小波变换模块 - 提取多频率特征"""

    def __init__(self, in_channels, wavelet_levels=3):
        super().__init__()
        self.wavelet_levels = wavelet_levels
        self.conv_layers = nn.ModuleList()

        for level in range(wavelet_levels):
            # 高频和低频分解
            self.conv_layers.append(nn.Conv1d(in_channels, in_channels, 3, stride=2, padding=1))

    def forward(self, x):
        # 保存原始序列长度
        original_length = x.shape[-1]
        current_x = x
        features = [x]  # 原始特征

        for level in range(self.wavelet_levels):
            # 模拟小波分解 - 使用下采样
            low_freq = F.avg_pool1d(current_x, 2, stride=2)

            # 对低频部分进行上采样以计算高频
            low_freq_upsampled = F.interpolate(low_freq, size=current_x.shape[-1], mode='linear', align_corners=False)
            high_freq = current_x - low_freq_upsampled

            # 将高频部分也上采样到原始长度以保持一致性
            high_freq_full = F.interpolate(high_freq, size=original_length, mode='linear', align_corners=False)
            low_freq_full = F.interpolate(low_freq, size=original_length, mode='linear', align_corners=False)

            features.append(low_freq_full)
            features.append(high_freq_full)
            current_x = low_freq

        return features


class UltraAdvancedResistivityModel(nn.Module):
    """电阻率预测模型 - 集成多种模块"""

    def __init__(self,
                 input_dim=7,
                 hidden_dim=128,
                 n_heads=8,
                 n_transformer_layers=4,
                 dropout=0.1,
                 estimate_uncertainty=True,
                 use_wavelet=True,
                 use_fpn=True):
        super().__init__()

        self.estimate_uncertainty = estimate_uncertainty
        self.use_wavelet = use_wavelet
        self.use_fpn = use_fpn

        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # 小波变换（可选）
        if use_wavelet:
            self.wavelet_transform = WaveletTransform(hidden_dim)
            self.wavelet_fusion = nn.Conv1d(hidden_dim * 7, hidden_dim, 1)  # 7个小波特征

        # 多尺度残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=5, dropout=dropout),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=7, dropout=dropout),
        ])

        # 特征金字塔网络（可选）
        if use_fpn:
            self.fpn = FeaturePyramidNetwork([hidden_dim] * 3, hidden_dim)
            self.fpn_fusion = nn.Conv1d(hidden_dim * 3, hidden_dim, 1)

        # Transformer编码器
        self.transformer = TransformerEncoder(
            d_model=hidden_dim,
            n_heads=n_heads,
            d_ff=hidden_dim * 4,
            n_layers=n_transformer_layers,
            dropout=dropout
        )

        # 双向LSTM（增强序列建模）
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        # 融合层
        fusion_dim = hidden_dim * 4  # avg_pool + max_pool + transformer + lstm
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出头
        if estimate_uncertainty:
            # 分别预测均值和方差
            self.mean_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.var_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()  # 确保方差为正
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
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

        # 2. 位置编码
        x = self.pos_encoding(x)

        # 转换为卷积格式进行卷积操作
        x_conv = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)

        # 3. 小波变换（可选）
        wavelet_features = []
        if self.use_wavelet:
            wavelet_features = self.wavelet_transform(x_conv)
            wavelet_concat = torch.cat(wavelet_features, dim=1)
            x_conv = self.wavelet_fusion(wavelet_concat)

        # 4. 多尺度残差块
        res_features = []
        for res_block in self.res_blocks:
            res_features.append(res_block(x_conv))

        # 5. 特征金字塔网络（可选）
        if self.use_fpn:
            fpn_features = self.fpn(res_features)
            fpn_concat = torch.cat(fpn_features, dim=1)
            x_conv = self.fpn_fusion(fpn_concat)
        else:
            x_conv = sum(res_features) / len(res_features)

        # 转换回序列格式进行Transformer处理
        x_seq = x_conv.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        # 6. Transformer编码器
        transformer_out = self.transformer(x_seq)

        # 7. 双向LSTM
        lstm_out, _ = self.lstm(transformer_out)

        # 8. 全局特征提取
        # 平均池化和最大池化
        avg_pool_conv = self.adaptive_pool(x_conv).squeeze(-1)  # (batch, hidden_dim)
        max_pool_conv = self.adaptive_max_pool(x_conv).squeeze(-1)  # (batch, hidden_dim)

        # Transformer全局特征
        transformer_global = torch.mean(transformer_out, dim=1)  # (batch, hidden_dim)

        # LSTM全局特征
        lstm_global = torch.mean(lstm_out, dim=1)  # (batch, hidden_dim*2)
        lstm_global = lstm_global[:, :transformer_global.size(1)]  # 截取匹配维度

        # 9. 特征融合
        global_features = torch.cat([
            avg_pool_conv,
            max_pool_conv,
            transformer_global,
            lstm_global
        ], dim=1)

        fused_features = self.fusion_layers(global_features)

        # 10. 输出预测
        if self.estimate_uncertainty:
            mean = self.mean_head(fused_features)
            var = self.var_head(fused_features) + 1e-6  # 添加小常数避免数值问题
            return mean, var
        else:
            output = self.output_head(fused_features)
            return output


class EnsembleUltraModel(nn.Module):
    """集成多个超级增强模型的集成模型"""

    def __init__(self, model_configs):
        super().__init__()
        self.models = nn.ModuleList([
            UltraAdvancedResistivityModel(**config) for config in model_configs
        ])
        self.num_models = len(self.models)

        # 学习权重（可选）
        self.ensemble_weights = nn.Parameter(torch.ones(self.num_models))

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

        # 加权平均
        weights = F.softmax(self.ensemble_weights, dim=0)

        if uncertainties:
            # 贝叶斯模型平均
            weighted_mean = sum(w * out for w, out in zip(weights, outputs))
            weighted_var = sum(w * (var + out ** 2) for w, var, out in zip(weights, uncertainties, outputs))
            weighted_var = weighted_var - weighted_mean ** 2
            return weighted_mean, weighted_var
        else:
            # 简单加权平均
            ensemble_output = sum(w * out for w, out in zip(weights, outputs))
            return ensemble_output


# 预定义配置
def get_ultra_model_configs():
    """获取不同配置的超级模型"""
    configs = [
        {
            'input_dim': 7,
            'hidden_dim': 128,
            'n_heads': 8,
            'n_transformer_layers': 4,
            'dropout': 0.1,
            'estimate_uncertainty': True,
            'use_wavelet': True,
            'use_fpn': True
        },
        {
            'input_dim': 7,
            'hidden_dim': 96,
            'n_heads': 6,
            'n_transformer_layers': 3,
            'dropout': 0.15,
            'estimate_uncertainty': True,
            'use_wavelet': False,
            'use_fpn': True
        },
        {
            'input_dim': 7,
            'hidden_dim': 160,
            'n_heads': 10,
            'n_transformer_layers': 5,
            'dropout': 0.05,
            'estimate_uncertainty': True,
            'use_wavelet': True,
            'use_fpn': False
        }
    ]
    return configs


# 创建模型的工厂函数
def create_ultra_model(config_name='ultra_single', **kwargs):
    """创建超级增强模型"""
    if config_name == 'ultra_single':
        return UltraAdvancedResistivityModel(**kwargs)
    elif config_name == 'ultra_ensemble':
        configs = get_ultra_model_configs()
        return EnsembleUltraModel(configs)
    else:
        raise ValueError(f"Unknown config: {config_name}")


# 模型复杂度分析
def analyze_model_complexity(model, input_shape=(32, 128, 7)):
    """分析模型复杂度"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 估算FLOPs（浮点运算数）
    dummy_input = torch.randn(input_shape)
    flops = 0

    # 简单估算（实际计算会更复杂）
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            flops += module.in_features * module.out_features
        elif isinstance(module, nn.Conv1d):
            flops += module.in_channels * module.out_channels * module.kernel_size[0]

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'estimated_flops': flops,
        'model_size_mb': total_params * 4 / 1024 / 1024  # 假设float32
    }