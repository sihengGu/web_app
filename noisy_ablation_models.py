"""
带噪声的消融实验模型设计 - 增强泛化能力和可解释性
包含多种噪声类型：输入噪声、特征噪声、权重噪声等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from rich import inspect
import inspect


# 导入基础组件（从原始代码复制）
class PositionalEncoding(nn.Module):
    """位置编码"""

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
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return x


# ================================ 噪声层定义 ================================

class GaussianNoise(nn.Module):
    """高斯噪声层"""

    def __init__(self, noise_std=0.1, noise_type='additive'):
        super().__init__()
        self.noise_std = noise_std
        self.noise_type = noise_type  # 'additive' or 'multiplicative'

    def forward(self, x):
        if not self.training and not hasattr(self, 'force_noise'):
            return x

        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std

            if self.noise_type == 'additive':
                return x + noise
            elif self.noise_type == 'multiplicative':
                return x * (1 + noise)
        return x


class UniformNoise(nn.Module):
    """均匀噪声层"""

    def __init__(self, noise_range=0.1):
        super().__init__()
        self.noise_range = noise_range

    def forward(self, x):
        if not self.training and not hasattr(self, 'force_noise'):
            return x

        if self.noise_range > 0:
            noise = (torch.rand_like(x) - 0.5) * 2 * self.noise_range
            return x + noise
        return x


class FeatureNoise(nn.Module):
    """特征级噪声 - 随机遮蔽或扰动某些特征"""

    def __init__(self, noise_prob=0.1, noise_std=0.1, mask_prob=0.05):
        super().__init__()
        self.noise_prob = noise_prob  # 添加噪声的概率
        self.noise_std = noise_std  # 噪声标准差
        self.mask_prob = mask_prob  # 特征遮蔽概率

    def forward(self, x):
        if not self.training and not hasattr(self, 'force_noise'):
            return x

        # 随机添加噪声
        if self.noise_prob > 0 and torch.rand(1) < self.noise_prob:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # 随机遮蔽特征
        if self.mask_prob > 0:
            mask = torch.rand_like(x) > self.mask_prob
            x = x * mask.float()

        return x


class WeightNoise(nn.Module):
    """权重噪声 - 为线性层添加权重扰动"""

    def __init__(self, module, noise_std=0.01):
        super().__init__()
        self.module = module
        self.noise_std = noise_std

    def forward(self, x):
        if self.training and self.noise_std > 0:
            # 为权重添加临时噪声
            if hasattr(self.module, 'weight'):
                original_weight = self.module.weight.data.clone()
                noise = torch.randn_like(self.module.weight) * self.noise_std
                self.module.weight.data += noise

                output = self.module(x)

                # 恢复原始权重
                self.module.weight.data = original_weight
                return output

        return self.module(x)


class AdaptiveNoise(nn.Module):
    """自适应噪声 - 根据输入动态调整噪声强度"""

    def __init__(self, base_noise_std=0.1, adaptive_factor=0.1):
        super().__init__()
        self.base_noise_std = base_noise_std
        self.adaptive_factor = adaptive_factor

    def forward(self, x):
        if not self.training and not hasattr(self, 'force_noise'):
            return x

        if self.base_noise_std > 0:
            # 根据输入的方差自适应调整噪声
            input_std = torch.std(x, dim=-1, keepdim=True)
            adaptive_noise_std = self.base_noise_std * (1 + self.adaptive_factor * input_std)
            noise = torch.randn_like(x) * adaptive_noise_std
            return x + noise
        return x


# ================================ 基础组件（带噪声版本） ================================

class NoisyMultiHeadAttention(nn.Module):
    """带噪声的多头注意力机制"""

    def __init__(self, d_model, n_heads, dropout=0.1, attention_noise_std=0.01):
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
        self.attention_noise = GaussianNoise(attention_noise_std)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 为输入添加噪声
        query = self.attention_noise(query)
        key = self.attention_noise(key)
        value = self.attention_noise(value)

        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = self.attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(attn_output), attn_weights

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class NoisyResidualBlock(nn.Module):
    """带噪声的残差块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1,
                 input_noise_std=0.05, feature_noise_std=0.02, use_attention=True):
        super().__init__()

        # 输入噪声
        self.input_noise = GaussianNoise(input_noise_std)

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 特征噪声
        self.feature_noise = GaussianNoise(feature_noise_std)

        # 注意力机制（可选）
        self.use_attention = use_attention
        if use_attention:
            from ablation_models import ChannelAttention, SpatialAttention, SEBlock
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
        # 添加输入噪声
        x_noisy = self.input_noise(x)
        residual = self.shortcut(x_noisy)

        # 前向传播
        out = F.relu(self.bn1(self.conv1(x_noisy)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # 添加特征噪声
        out = self.feature_noise(out)

        # 应用注意力机制
        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)
            out = self.se_block(out)

        # 残差连接
        out += residual
        out = F.relu(out)

        return out


class NoisyLSTM(nn.Module):
    """带噪声的LSTM层"""

    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True,
                 dropout=0.1, input_noise_std=0.05, hidden_noise_std=0.02):
        super().__init__()

        self.input_noise = GaussianNoise(input_noise_std)
        self.hidden_noise = GaussianNoise(hidden_noise_std)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        # 添加输入噪声
        x_noisy = self.input_noise(x)

        # LSTM前向传播
        output, (hidden, cell) = self.lstm(x_noisy)

        # 添加隐状态噪声
        output = self.hidden_noise(output)

        return output, (hidden, cell)


# ================================ 带噪声的消融实验模型 ================================

class NoisyBaselineModel(nn.Module):
    """带噪声的基线模型"""

    def __init__(self, input_dim=7, hidden_dim=64, dropout=0.1, estimate_uncertainty=True,
                 input_noise_std=0.1, feature_noise_std=0.05):
        super().__init__()
        self.estimate_uncertainty = estimate_uncertainty

        # 输入噪声
        self.input_noise = GaussianNoise(input_noise_std)

        # 网络层
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            GaussianNoise(feature_noise_std),  # 特征噪声
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            GaussianNoise(feature_noise_std),  # 特征噪声
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if estimate_uncertainty:
            self.mean_head = nn.Linear(hidden_dim // 2, 1)
            self.var_head = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.Softplus())
        else:
            self.output_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        # 添加输入噪声
        x = self.input_noise(x)

        # 简单地对序列取平均
        x = torch.mean(x, dim=1)  # (batch, seq_len, features) -> (batch, features)

        features = self.layers(x)

        if self.estimate_uncertainty:
            mean = self.mean_head(features)
            var = self.var_head(features) + 1e-6
            return mean, var
        else:
            output = self.output_head(features)
            return output


class NoisySimpleLSTMModel(nn.Module):
    """带噪声的简单LSTM模型"""

    def __init__(self, input_dim=7, hidden_dim=128, dropout=0.1, estimate_uncertainty=True,
                 input_noise_std=0.1, lstm_noise_std=0.05, feature_noise_std=0.02):
        super().__init__()
        self.estimate_uncertainty = estimate_uncertainty

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_noise = GaussianNoise(input_noise_std)

        # 带噪声的LSTM
        self.lstm = NoisyLSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            input_noise_std=0.02,  # LSTM内部输入噪声
            hidden_noise_std=lstm_noise_std
        )

        # 特征噪声
        self.feature_noise = GaussianNoise(feature_noise_std)

        # 融合层
        fusion_dim = hidden_dim * 2  # 双向LSTM
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if estimate_uncertainty:
            self.mean_head = nn.Linear(hidden_dim, 1)
            self.var_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        else:
            self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 输入投影和噪声
        x = self.input_projection(x)
        x = self.input_noise(x)

        # LSTM处理
        lstm_out, _ = self.lstm(x)
        lstm_global = torch.mean(lstm_out, dim=1)

        # 添加特征噪声
        lstm_global = self.feature_noise(lstm_global)

        fused_features = self.fusion_layers(lstm_global)

        if self.estimate_uncertainty:
            mean = self.mean_head(fused_features)
            var = self.var_head(fused_features) + 1e-6
            return mean, var
        else:
            output = self.output_head(fused_features)
            return output


class NoisyBasicResNetModel(nn.Module):
    """带噪声的基础残差网络模型"""

    def __init__(self, input_dim=7, hidden_dim=128, dropout=0.1, estimate_uncertainty=True,
                 input_noise_std=0.1, conv_noise_std=0.05, feature_noise_std=0.02):
        super().__init__()
        self.estimate_uncertainty = estimate_uncertainty

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_noise = GaussianNoise(input_noise_std)

        # 带噪声的残差块
        self.res_blocks = nn.ModuleList([
            NoisyResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout,
                               input_noise_std=conv_noise_std, feature_noise_std=feature_noise_std,
                               use_attention=False),
            NoisyResidualBlock(hidden_dim, hidden_dim, kernel_size=5, dropout=dropout,
                               input_noise_std=conv_noise_std, feature_noise_std=feature_noise_std,
                               use_attention=False),
            NoisyResidualBlock(hidden_dim, hidden_dim, kernel_size=7, dropout=dropout,
                               input_noise_std=conv_noise_std, feature_noise_std=feature_noise_std,
                               use_attention=False),
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        # 特征融合
        fusion_dim = hidden_dim * 2  # avg_pool + max_pool
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            GaussianNoise(feature_noise_std),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if estimate_uncertainty:
            self.mean_head = nn.Linear(hidden_dim, 1)
            self.var_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        else:
            self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 输入投影和噪声
        x = self.input_projection(x)
        x = self.input_noise(x)
        x_conv = x.transpose(1, 2)

        # 残差块处理
        res_features = [block(x_conv) for block in self.res_blocks]
        x_conv = sum(res_features) / len(res_features)

        # 全局池化
        avg_pool_conv = self.adaptive_pool(x_conv).squeeze(-1)
        max_pool_conv = self.adaptive_max_pool(x_conv).squeeze(-1)

        global_features = torch.cat([avg_pool_conv, max_pool_conv], dim=1)
        fused_features = self.fusion_layers(global_features)

        if self.estimate_uncertainty:
            mean = self.mean_head(fused_features)
            var = self.var_head(fused_features) + 1e-6
            return mean, var
        else:
            output = self.output_head(fused_features)
            return output


class NoisyUltraOriginalModel(nn.Module):
    """带噪声的完整超级模型"""

    def __init__(self, input_dim=7, hidden_dim=128, n_heads=8, n_transformer_layers=4,
                 dropout=0.1, estimate_uncertainty=True, use_wavelet=True, use_fpn=True,
                 input_noise_std=0.1, feature_noise_std=0.02, attention_noise_std=0.01,
                 conv_noise_std=0.03, lstm_noise_std=0.02):
        super().__init__()

        self.estimate_uncertainty = estimate_uncertainty
        self.use_wavelet = use_wavelet
        self.use_fpn = use_fpn

        # 输入噪声
        self.input_noise = GaussianNoise(input_noise_std)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # 小波变换（可选）
        if use_wavelet:
            from ablation_models import WaveletTransform
            self.wavelet_transform = WaveletTransform(hidden_dim)
            self.wavelet_fusion = nn.Conv1d(hidden_dim * 7, hidden_dim, 1)
            self.wavelet_noise = GaussianNoise(feature_noise_std)

        # 带噪声的多尺度残差块
        self.res_blocks = nn.ModuleList([
            NoisyResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout,
                               input_noise_std=conv_noise_std, feature_noise_std=feature_noise_std, use_attention=True),
            NoisyResidualBlock(hidden_dim, hidden_dim, kernel_size=5, dropout=dropout,
                               input_noise_std=conv_noise_std, feature_noise_std=feature_noise_std, use_attention=True),
            NoisyResidualBlock(hidden_dim, hidden_dim, kernel_size=7, dropout=dropout,
                               input_noise_std=conv_noise_std, feature_noise_std=feature_noise_std, use_attention=True),
        ])

        # 特征金字塔网络（可选）
        if use_fpn:
            from ablation_models import FeaturePyramidNetwork
            self.fpn = FeaturePyramidNetwork([hidden_dim] * 3, hidden_dim)
            self.fpn_fusion = nn.Conv1d(hidden_dim * 3, hidden_dim, 1)
            self.fpn_noise = GaussianNoise(feature_noise_std)

        # 带噪声的Transformer编码器
        self.transformer_noise = GaussianNoise(attention_noise_std)
        from ablation_models import TransformerEncoder
        self.transformer = TransformerEncoder(hidden_dim, n_heads, hidden_dim * 4, n_transformer_layers, dropout)

        # 带噪声的LSTM
        self.lstm = NoisyLSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            input_noise_std=0.01,
            hidden_noise_std=lstm_noise_std
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        # 特征融合
        fusion_dim = hidden_dim * 4  # avg_pool + max_pool + transformer + lstm
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            GaussianNoise(feature_noise_std),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            GaussianNoise(feature_noise_std),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出头
        if estimate_uncertainty:
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
                nn.Softplus()
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 1. 输入噪声和投影
        x = self.input_noise(x)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x_conv = x.transpose(1, 2)

        # 2. 小波变换（可选）
        if self.use_wavelet:
            wavelet_features = self.wavelet_transform(x_conv)
            wavelet_concat = torch.cat(wavelet_features, dim=1)
            x_conv = self.wavelet_fusion(wavelet_concat)
            x_conv = self.wavelet_noise(x_conv)

        # 3. 多尺度残差块
        res_features = [block(x_conv) for block in self.res_blocks]

        # 4. 特征金字塔网络（可选）
        if self.use_fpn:
            fpn_features = self.fpn(res_features)
            fpn_concat = torch.cat(fpn_features, dim=1)
            x_conv = self.fpn_fusion(fpn_concat)
            x_conv = self.fpn_noise(x_conv)
        else:
            x_conv = sum(res_features) / len(res_features)

        # 5. Transformer编码器
        x_seq = x_conv.transpose(1, 2)
        x_seq = self.transformer_noise(x_seq)
        transformer_out = self.transformer(x_seq)

        # 6. LSTM
        lstm_out, _ = self.lstm(transformer_out)

        # 7. 全局特征提取
        avg_pool_conv = self.adaptive_pool(x_conv).squeeze(-1)
        max_pool_conv = self.adaptive_max_pool(x_conv).squeeze(-1)
        transformer_global = torch.mean(transformer_out, dim=1)
        lstm_global = torch.mean(lstm_out, dim=1)[:, :transformer_global.size(1)]

        # 8. 特征融合
        global_features = torch.cat([avg_pool_conv, max_pool_conv, transformer_global, lstm_global], dim=1)
        fused_features = self.fusion_layers(global_features)

        # 9. 输出预测
        if self.estimate_uncertainty:
            mean = self.mean_head(fused_features)
            var = self.var_head(fused_features) + 1e-6
            return mean, var
        else:
            output = self.output_head(fused_features)
            return output


# ================================ 噪声配置和工厂函数 ================================

class NoiseConfig:
    """噪声配置类"""

    def __init__(self,
                 input_noise_std=0.1,  # 输入噪声标准差
                 feature_noise_std=0.02,  # 特征噪声标准差
                 conv_noise_std=0.03,  # 卷积噪声标准差
                 attention_noise_std=0.01,  # 注意力噪声标准差
                 lstm_noise_std=0.02,  # LSTM噪声标准差
                 weight_noise_std=0.005,  # 权重噪声标准差
                 adaptive_noise=False,  # 是否使用自适应噪声
                 noise_schedule='constant'  # 噪声调度策略
                 ):
        self.input_noise_std = input_noise_std
        self.feature_noise_std = feature_noise_std
        self.conv_noise_std = conv_noise_std
        self.attention_noise_std = attention_noise_std
        self.lstm_noise_std = lstm_noise_std
        self.weight_noise_std = weight_noise_std
        self.adaptive_noise = adaptive_noise
        self.noise_schedule = noise_schedule


def create_noisy_ablation_model(model_name, noise_config=None, **kwargs):
    """创建带噪声的消融实验模型 (最终修正版)"""

    # 默认噪声配置
    if noise_config is None:
        noise_config = NoiseConfig()

    # 模型工厂字典
    noisy_models = {
        'noisy_baseline': NoisyBaselineModel,
        'noisy_simple_lstm': NoisySimpleLSTMModel,
        'noisy_basic_resnet': NoisyBasicResNetModel,
        'noisy_ultra_original': NoisyUltraOriginalModel,
    }

    if model_name not in noisy_models:
        raise ValueError(f"Unknown noisy model: {model_name}. Available: {list(noisy_models.keys())}")

    # 准备传递给模型构造函数的参数字典
    model_params = kwargs.copy()

    # 根据模型类型，从noise_config中提取并添加相关的噪声参数
    if model_name == 'noisy_baseline':
        model_params.update({
            'input_noise_std': noise_config.input_noise_std,
            'feature_noise_std': noise_config.feature_noise_std,
        })
    elif model_name == 'noisy_simple_lstm':
        model_params.update({
            'input_noise_std': noise_config.input_noise_std,
            'lstm_noise_std': noise_config.lstm_noise_std,
            'feature_noise_std': noise_config.feature_noise_std,
        })
    elif model_name == 'noisy_basic_resnet':
        model_params.update({
            'input_noise_std': noise_config.input_noise_std,
            'conv_noise_std': noise_config.conv_noise_std,
            'feature_noise_std': noise_config.feature_noise_std,
        })
    elif model_name == 'noisy_ultra_original':
        model_params.update({
            'input_noise_std': noise_config.input_noise_std,
            'feature_noise_std': noise_config.feature_noise_std,
            'attention_noise_std': noise_config.attention_noise_std,
            'conv_noise_std': noise_config.conv_noise_std,
            'lstm_noise_std': noise_config.lstm_noise_std,
        })

    # --- 【核心修正】---
    # 在创建模型实例前，过滤掉所有目标模型构造函数不接受的参数

    # 1. 获取目标模型类
    target_class = noisy_models[model_name]

    # 2. 使用inspect库获取该类__init__方法所有合法的参数名
    valid_keys = inspect.signature(target_class.__init__).parameters.keys()

    # 3. 从我们准备好的参数字典中，只保留那些合法的参数
    filtered_params = {key: model_params[key] for key in model_params if key in valid_keys}

    # 4. 使用过滤后的、干净的参数字典来创建模型实例
    return target_class(**filtered_params)


def get_noisy_ablation_configs():
    """获取带噪声的消融实验配置"""

    # 定义不同强度的噪声配置
    noise_configs = {
        'low_noise': NoiseConfig(
            input_noise_std=0.05,
            feature_noise_std=0.01,
            conv_noise_std=0.015,
            attention_noise_std=0.005,
            lstm_noise_std=0.01
        ),
        'medium_noise': NoiseConfig(
            input_noise_std=0.1,
            feature_noise_std=0.02,
            conv_noise_std=0.03,
            attention_noise_std=0.01,
            lstm_noise_std=0.02
        ),
        'high_noise': NoiseConfig(
            input_noise_std=0.2,
            feature_noise_std=0.05,
            conv_noise_std=0.06,
            attention_noise_std=0.02,
            lstm_noise_std=0.04
        )
    }

    # 基础模型配置
    base_config = {
        'input_dim': 7,
        'hidden_dim': 128,
        'dropout': 0.1,
        'estimate_uncertainty': True
    }

    full_config = {
        **base_config,
        'n_heads': 8,
        'n_transformer_layers': 4,
        'use_wavelet': True,
        'use_fpn': True
    }

    # 生成所有配置组合
    configs = {}

    for noise_level, noise_config in noise_configs.items():
        configs[f'noisy_baseline_{noise_level}'] = {
            **base_config,
            'noise_config': noise_config
        }

        configs[f'noisy_simple_lstm_{noise_level}'] = {
            **base_config,
            'noise_config': noise_config
        }

        configs[f'noisy_basic_resnet_{noise_level}'] = {
            **base_config,
            'noise_config': noise_config
        }

        configs[f'noisy_ultra_original_{noise_level}'] = {
            **full_config,
            'noise_config': noise_config
        }

    return configs


def analyze_noise_impact(model_with_noise, model_without_noise, test_loader, device):
    """分析噪声对模型性能的影响"""

    model_with_noise.eval()
    model_without_noise.eval()

    predictions_with_noise = []
    predictions_without_noise = []
    targets = []

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)

            # 带噪声模型预测
            if hasattr(model_with_noise, 'estimate_uncertainty') and model_with_noise.estimate_uncertainty:
                mean_noise, _ = model_with_noise(inputs)
                predictions_with_noise.append(mean_noise.cpu())
            else:
                pred_noise = model_with_noise(inputs)
                predictions_with_noise.append(pred_noise.cpu())

            # 无噪声模型预测
            if hasattr(model_without_noise, 'estimate_uncertainty') and model_without_noise.estimate_uncertainty:
                mean_clean, _ = model_without_noise(inputs)
                predictions_without_noise.append(mean_clean.cpu())
            else:
                pred_clean = model_without_noise(inputs)
                predictions_without_noise.append(pred_clean.cpu())

            targets.append(target.cpu())

    # 合并预测结果
    predictions_with_noise = torch.cat(predictions_with_noise, dim=0).numpy()
    predictions_without_noise = torch.cat(predictions_without_noise, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    # 计算性能指标
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    r2_noise = r2_score(targets, predictions_with_noise)
    r2_clean = r2_score(targets, predictions_without_noise)

    mae_noise = mean_absolute_error(targets, predictions_with_noise)
    mae_clean = mean_absolute_error(targets, predictions_without_noise)

    # 计算鲁棒性指标
    prediction_variance = np.var(predictions_with_noise - predictions_without_noise)

    return {
        'r2_with_noise': r2_noise,
        'r2_without_noise': r2_clean,
        'r2_degradation': r2_clean - r2_noise,
        'mae_with_noise': mae_noise,
        'mae_without_noise': mae_clean,
        'mae_increase': mae_noise - mae_clean,
        'prediction_variance': prediction_variance,
        'robustness_score': 1 - (prediction_variance / np.var(targets))
    }


if __name__ == "__main__":
    # 示例：创建带噪声的模型
    noise_config = NoiseConfig(
        input_noise_std=0.1,
        feature_noise_std=0.02,
        conv_noise_std=0.03,
        attention_noise_std=0.01,
        lstm_noise_std=0.02
    )

    # 测试模型创建
    models_to_test = ['noisy_baseline', 'noisy_simple_lstm', 'noisy_basic_resnet', 'noisy_ultra_original']

    print("带噪声的消融实验模型测试:")
    print("=" * 60)

    for model_name in models_to_test:
        try:
            # 根据模型类型准备不同的基础配置
            if model_name == 'noisy_baseline':
                base_config = {
                    'input_dim': 7,
                    'hidden_dim': 64,
                    'dropout': 0.1,
                    'estimate_uncertainty': True
                }
            elif model_name == 'noisy_simple_lstm':
                base_config = {
                    'input_dim': 7,
                    'hidden_dim': 128,
                    'dropout': 0.1,
                    'estimate_uncertainty': True
                }
            elif model_name == 'noisy_basic_resnet':
                base_config = {
                    'input_dim': 7,
                    'hidden_dim': 128,
                    'dropout': 0.1,
                    'estimate_uncertainty': True
                }
            elif model_name == 'noisy_ultra_original':
                base_config = {
                    'input_dim': 7,
                    'hidden_dim': 128,
                    'n_heads': 8,
                    'n_transformer_layers': 4,
                    'dropout': 0.1,
                    'estimate_uncertainty': True,
                    'use_wavelet': True,
                    'use_fpn': True
                }
            else:
                base_config = {
                    'input_dim': 7,
                    'hidden_dim': 128,
                    'dropout': 0.1,
                    'estimate_uncertainty': True
                }

            # 创建模型
            model = create_noisy_ablation_model(
                model_name,
                noise_config=noise_config,
                **base_config
            )

            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())

            print(f"{model_name:25} | 参数量: {total_params:>8,}")

            # 测试前向传播
            x = torch.randn(4, 128, 7)
            with torch.no_grad():
                output = model(x)
                if isinstance(output, tuple):
                    print(f"{'':25} | 输出维度: mean{output[0].shape}, var{output[1].shape}")
                else:
                    print(f"{'':25} | 输出维度: {output.shape}")

        except Exception as e:
            print(f"{model_name:25} | 错误: {str(e)}")

        print("-" * 60)

    print("\n噪声配置选项:")
    configs = get_noisy_ablation_configs()
    for config_name in list(configs.keys())[:6]:  # 显示前6个配置
        print(f"  - {config_name}")
    print(f"  ... 共 {len(configs)} 个配置")