"""
消融实验模型设计 - 基于UltraAdvancedResistivityModel的弱化版本
用于验证各个组件的贡献度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

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


class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv1d(in_channels // 8, 1, 1)

    def forward(self, x):
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))
        return x * attention


class ChannelAttention(nn.Module):
    """通道注意力机制"""
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
    """残差块（带注意力）"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1, use_attention=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention(out_channels)
            self.spatial_attention = SpatialAttention(out_channels)
            self.se_block = SEBlock(out_channels)

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

        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)
            out = self.se_block(out)

        out += residual
        out = F.relu(out)
        return out


class SimpleResidualBlock(nn.Module):
    """简化的残差块（无注意力）"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

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
        out += residual
        out = F.relu(out)
        return out


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
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


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


class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络"""
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


class WaveletTransform(nn.Module):
    """小波变换模块"""
    def __init__(self, in_channels, wavelet_levels=3):
        super().__init__()
        self.wavelet_levels = wavelet_levels

    def forward(self, x):
        original_length = x.shape[-1]
        current_x = x
        features = [x]

        for level in range(self.wavelet_levels):
            low_freq = F.avg_pool1d(current_x, 2, stride=2)
            low_freq_upsampled = F.interpolate(low_freq, size=current_x.shape[-1], mode='linear', align_corners=False)
            high_freq = current_x - low_freq_upsampled
            high_freq_full = F.interpolate(high_freq, size=original_length, mode='linear', align_corners=False)
            low_freq_full = F.interpolate(low_freq, size=original_length, mode='linear', align_corners=False)
            features.append(low_freq_full)
            features.append(high_freq_full)
            current_x = low_freq

        return features


# ================================ 消融实验模型 ================================

class UltraAdvancedResistivityModel_Original(nn.Module):
    """原始完整版模型 - 包含所有组件"""
    def __init__(self, input_dim=7, hidden_dim=128, n_heads=8, n_transformer_layers=4,
                 dropout=0.1, estimate_uncertainty=True, use_wavelet=True, use_fpn=True):
        super().__init__()
        self.estimate_uncertainty = estimate_uncertainty
        self.use_wavelet = use_wavelet
        self.use_fpn = use_fpn

        # 所有组件都保留
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        if use_wavelet:
            self.wavelet_transform = WaveletTransform(hidden_dim)
            self.wavelet_fusion = nn.Conv1d(hidden_dim * 7, hidden_dim, 1)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout, use_attention=True),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=5, dropout=dropout, use_attention=True),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=7, dropout=dropout, use_attention=True),
        ])

        if use_fpn:
            self.fpn = FeaturePyramidNetwork([hidden_dim] * 3, hidden_dim)
            self.fpn_fusion = nn.Conv1d(hidden_dim * 3, hidden_dim, 1)

        self.transformer = TransformerEncoder(hidden_dim, n_heads, hidden_dim * 4, n_transformer_layers, dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        fusion_dim = hidden_dim * 4
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if estimate_uncertainty:
            self.mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
            self.var_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Softplus())
        else:
            self.output_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x_conv = x.transpose(1, 2)

        if self.use_wavelet:
            wavelet_features = self.wavelet_transform(x_conv)
            wavelet_concat = torch.cat(wavelet_features, dim=1)
            x_conv = self.wavelet_fusion(wavelet_concat)

        res_features = [block(x_conv) for block in self.res_blocks]

        if self.use_fpn:
            fpn_features = self.fpn(res_features)
            fpn_concat = torch.cat(fpn_features, dim=1)
            x_conv = self.fpn_fusion(fpn_concat)
        else:
            x_conv = sum(res_features) / len(res_features)

        x_seq = x_conv.transpose(1, 2)
        transformer_out = self.transformer(x_seq)
        lstm_out, _ = self.lstm(transformer_out)

        avg_pool_conv = self.adaptive_pool(x_conv).squeeze(-1)
        max_pool_conv = self.adaptive_max_pool(x_conv).squeeze(-1)
        transformer_global = torch.mean(transformer_out, dim=1)
        lstm_global = torch.mean(lstm_out, dim=1)[:, :transformer_global.size(1)]

        global_features = torch.cat([avg_pool_conv, max_pool_conv, transformer_global, lstm_global], dim=1)
        fused_features = self.fusion_layers(global_features)

        if self.estimate_uncertainty:
            mean = self.mean_head(fused_features)
            var = self.var_head(fused_features) + 1e-6
            return mean, var
        else:
            output = self.output_head(fused_features)
            return output


class UltraAdvancedResistivityModel_NoWavelet(nn.Module):
    """移除小波变换的版本"""
    def __init__(self, input_dim=7, hidden_dim=128, n_heads=8, n_transformer_layers=4,
                 dropout=0.1, estimate_uncertainty=True, use_fpn=True):
        super().__init__()
        self.estimate_uncertainty = estimate_uncertainty
        self.use_fpn = use_fpn

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # 移除小波变换

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout, use_attention=True),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=5, dropout=dropout, use_attention=True),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=7, dropout=dropout, use_attention=True),
        ])

        if use_fpn:
            self.fpn = FeaturePyramidNetwork([hidden_dim] * 3, hidden_dim)
            self.fpn_fusion = nn.Conv1d(hidden_dim * 3, hidden_dim, 1)

        self.transformer = TransformerEncoder(hidden_dim, n_heads, hidden_dim * 4, n_transformer_layers, dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        fusion_dim = hidden_dim * 4
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if estimate_uncertainty:
            self.mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
            self.var_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Softplus())
        else:
            self.output_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x_conv = x.transpose(1, 2)  # 直接使用原始投影，无小波变换

        res_features = [block(x_conv) for block in self.res_blocks]

        if self.use_fpn:
            fpn_features = self.fpn(res_features)
            fpn_concat = torch.cat(fpn_features, dim=1)
            x_conv = self.fpn_fusion(fpn_concat)
        else:
            x_conv = sum(res_features) / len(res_features)

        x_seq = x_conv.transpose(1, 2)
        transformer_out = self.transformer(x_seq)
        lstm_out, _ = self.lstm(transformer_out)

        avg_pool_conv = self.adaptive_pool(x_conv).squeeze(-1)
        max_pool_conv = self.adaptive_max_pool(x_conv).squeeze(-1)
        transformer_global = torch.mean(transformer_out, dim=1)
        lstm_global = torch.mean(lstm_out, dim=1)[:, :transformer_global.size(1)]

        global_features = torch.cat([avg_pool_conv, max_pool_conv, transformer_global, lstm_global], dim=1)
        fused_features = self.fusion_layers(global_features)

        if self.estimate_uncertainty:
            mean = self.mean_head(fused_features)
            var = self.var_head(fused_features) + 1e-6
            return mean, var
        else:
            output = self.output_head(fused_features)
            return output


class UltraAdvancedResistivityModel_NoFPN(nn.Module):
    """移除特征金字塔网络的版本"""
    def __init__(self, input_dim=7, hidden_dim=128, n_heads=8, n_transformer_layers=4,
                 dropout=0.1, estimate_uncertainty=True, use_wavelet=True):
        super().__init__()
        self.estimate_uncertainty = estimate_uncertainty
        self.use_wavelet = use_wavelet

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        if use_wavelet:
            self.wavelet_transform = WaveletTransform(hidden_dim)
            self.wavelet_fusion = nn.Conv1d(hidden_dim * 7, hidden_dim, 1)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout, use_attention=True),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=5, dropout=dropout, use_attention=True),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=7, dropout=dropout, use_attention=True),
        ])

        # 移除FPN，直接使用平均

        self.transformer = TransformerEncoder(hidden_dim, n_heads, hidden_dim * 4, n_transformer_layers, dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        fusion_dim = hidden_dim * 4
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if estimate_uncertainty:
            self.mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
            self.var_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Softplus())
        else:
            self.output_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x_conv = x.transpose(1, 2)

        if self.use_wavelet:
            wavelet_features = self.wavelet_transform(x_conv)
            wavelet_concat = torch.cat(wavelet_features, dim=1)
            x_conv = self.wavelet_fusion(wavelet_concat)

        res_features = [block(x_conv) for block in self.res_blocks]
        x_conv = sum(res_features) / len(res_features)  # 简单平均，不使用FPN

        x_seq = x_conv.transpose(1, 2)
        transformer_out = self.transformer(x_seq)
        lstm_out, _ = self.lstm(transformer_out)

        avg_pool_conv = self.adaptive_pool(x_conv).squeeze(-1)
        max_pool_conv = self.adaptive_max_pool(x_conv).squeeze(-1)
        transformer_global = torch.mean(transformer_out, dim=1)
        lstm_global = torch.mean(lstm_out, dim=1)[:, :transformer_global.size(1)]

        global_features = torch.cat([avg_pool_conv, max_pool_conv, transformer_global, lstm_global], dim=1)
        fused_features = self.fusion_layers(global_features)

        if self.estimate_uncertainty:
            mean = self.mean_head(fused_features)
            var = self.var_head(fused_features) + 1e-6
            return mean, var
        else:
            output = self.output_head(fused_features)
            return output


class UltraAdvancedResistivityModel_NoAttention(nn.Module):
    """移除注意力机制的版本"""
    def __init__(self, input_dim=7, hidden_dim=128, n_heads=8, n_transformer_layers=4,
                 dropout=0.1, estimate_uncertainty=True, use_wavelet=True, use_fpn=True):
        super().__init__()
        self.estimate_uncertainty = estimate_uncertainty
        self.use_wavelet = use_wavelet
        self.use_fpn = use_fpn

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        if use_wavelet:
            self.wavelet_transform = WaveletTransform(hidden_dim)
            self.wavelet_fusion = nn.Conv1d(hidden_dim * 7, hidden_dim, 1)

        # 使用简化的残差块（无注意力）
        self.res_blocks = nn.ModuleList([
            SimpleResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout),
            SimpleResidualBlock(hidden_dim, hidden_dim, kernel_size=5, dropout=dropout),
            SimpleResidualBlock(hidden_dim, hidden_dim, kernel_size=7, dropout=dropout),
        ])

        if use_fpn:
            self.fpn = FeaturePyramidNetwork([hidden_dim] * 3, hidden_dim)
            self.fpn_fusion = nn.Conv1d(hidden_dim * 3, hidden_dim, 1)

        self.transformer = TransformerEncoder(hidden_dim, n_heads, hidden_dim * 4, n_transformer_layers, dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        fusion_dim = hidden_dim * 4
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if estimate_uncertainty:
            self.mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
            self.var_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Softplus())
        else:
            self.output_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        # 与原始模型相同的前向传播，但残差块中没有注意力机制
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x_conv = x.transpose(1, 2)

        if self.use_wavelet:
            wavelet_features = self.wavelet_transform(x_conv)
            wavelet_concat = torch.cat(wavelet_features, dim=1)
            x_conv = self.wavelet_fusion(wavelet_concat)

        res_features = [block(x_conv) for block in self.res_blocks]

        if self.use_fpn:
            fpn_features = self.fpn(res_features)
            fpn_concat = torch.cat(fpn_features, dim=1)
            x_conv = self.fpn_fusion(fpn_concat)
        else:
            x_conv = sum(res_features) / len(res_features)

        x_seq = x_conv.transpose(1, 2)
        transformer_out = self.transformer(x_seq)
        lstm_out, _ = self.lstm(transformer_out)

        avg_pool_conv = self.adaptive_pool(x_conv).squeeze(-1)
        max_pool_conv = self.adaptive_max_pool(x_conv).squeeze(-1)
        transformer_global = torch.mean(transformer_out, dim=1)
        lstm_global = torch.mean(lstm_out, dim=1)[:, :transformer_global.size(1)]

        global_features = torch.cat([avg_pool_conv, max_pool_conv, transformer_global, lstm_global], dim=1)
        fused_features = self.fusion_layers(global_features)

        if self.estimate_uncertainty:
            mean = self.mean_head(fused_features)
            var = self.var_head(fused_features) + 1e-6
            return mean, var
        else:
            output = self.output_head(fused_features)
            return output


class UltraAdvancedResistivityModel_NoTransformer(nn.Module):
    """移除Transformer编码器的版本"""
    def __init__(self, input_dim=7, hidden_dim=128, dropout=0.1, estimate_uncertainty=True,
                 use_wavelet=True, use_fpn=True):
        super().__init__()
        self.estimate_uncertainty = estimate_uncertainty
        self.use_wavelet = use_wavelet
        self.use_fpn = use_fpn

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # 移除位置编码（因为没有Transformer）

        if use_wavelet:
            self.wavelet_transform = WaveletTransform(hidden_dim)
            self.wavelet_fusion = nn.Conv1d(hidden_dim * 7, hidden_dim, 1)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout, use_attention=True),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=5, dropout=dropout, use_attention=True),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=7, dropout=dropout, use_attention=True),
        ])

        if use_fpn:
            self.fpn = FeaturePyramidNetwork([hidden_dim] * 3, hidden_dim)
            self.fpn_fusion = nn.Conv1d(hidden_dim * 3, hidden_dim, 1)

        # 移除Transformer，只保留LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        # 调整融合维度（减去Transformer的贡献）
        fusion_dim = hidden_dim * 3  # avg_pool + max_pool + lstm
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if estimate_uncertainty:
            self.mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
            self.var_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Softplus())
        else:
            self.output_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        # 不使用位置编码
        x_conv = x.transpose(1, 2)

        if self.use_wavelet:
            wavelet_features = self.wavelet_transform(x_conv)
            wavelet_concat = torch.cat(wavelet_features, dim=1)
            x_conv = self.wavelet_fusion(wavelet_concat)

        res_features = [block(x_conv) for block in self.res_blocks]

        if self.use_fpn:
            fpn_features = self.fpn(res_features)
            fpn_concat = torch.cat(fpn_features, dim=1)
            x_conv = self.fpn_fusion(fpn_concat)
        else:
            x_conv = sum(res_features) / len(res_features)

        x_seq = x_conv.transpose(1, 2)
        # 只使用LSTM，不使用Transformer
        lstm_out, _ = self.lstm(x_seq)

        avg_pool_conv = self.adaptive_pool(x_conv).squeeze(-1)
        max_pool_conv = self.adaptive_max_pool(x_conv).squeeze(-1)
        lstm_global = torch.mean(lstm_out, dim=1)[:, :avg_pool_conv.size(1)]

        global_features = torch.cat([avg_pool_conv, max_pool_conv, lstm_global], dim=1)
        fused_features = self.fusion_layers(global_features)

        if self.estimate_uncertainty:
            mean = self.mean_head(fused_features)
            var = self.var_head(fused_features) + 1e-6
            return mean, var
        else:
            output = self.output_head(fused_features)
            return output


class UltraAdvancedResistivityModel_BasicResNet(nn.Module):
    """只保留基本残差网络的版本"""
    def __init__(self, input_dim=7, hidden_dim=128, dropout=0.1, estimate_uncertainty=True):
        super().__init__()
        self.estimate_uncertainty = estimate_uncertainty

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 只保留简化的残差块
        self.res_blocks = nn.ModuleList([
            SimpleResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout),
            SimpleResidualBlock(hidden_dim, hidden_dim, kernel_size=5, dropout=dropout),
            SimpleResidualBlock(hidden_dim, hidden_dim, kernel_size=7, dropout=dropout),
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        # 简化的融合层
        fusion_dim = hidden_dim * 2  # avg_pool + max_pool
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if estimate_uncertainty:
            self.mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
            self.var_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Softplus())
        else:
            self.output_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        x_conv = x.transpose(1, 2)

        # 只使用残差块
        res_features = [block(x_conv) for block in self.res_blocks]
        x_conv = sum(res_features) / len(res_features)

        # 简单的全局池化
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


class UltraAdvancedResistivityModel_SimpleLSTM(nn.Module):
    """只保留LSTM的版本"""
    def __init__(self, input_dim=7, hidden_dim=128, dropout=0.1, estimate_uncertainty=True):
        super().__init__()
        self.estimate_uncertainty = estimate_uncertainty

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 只保留LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)

        # 简化的融合层
        fusion_dim = hidden_dim * 2  # 双向LSTM
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if estimate_uncertainty:
            self.mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
            self.var_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Softplus())
        else:
            self.output_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)

        # 只使用LSTM
        lstm_out, _ = self.lstm(x)
        lstm_global = torch.mean(lstm_out, dim=1)

        fused_features = self.fusion_layers(lstm_global)

        if self.estimate_uncertainty:
            mean = self.mean_head(fused_features)
            var = self.var_head(fused_features) + 1e-6
            return mean, var
        else:
            output = self.output_head(fused_features)
            return output


class BaselineModel(nn.Module):
    """最简单的基线模型"""
    def __init__(self, input_dim=7, hidden_dim=64, dropout=0.1, estimate_uncertainty=True):
        super().__init__()
        self.estimate_uncertainty = estimate_uncertainty

        # 最简单的架构
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if estimate_uncertainty:
            self.mean_head = nn.Linear(hidden_dim // 2, 1)
            self.var_head = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.Softplus())
        else:
            self.output_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
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


# ================================ 工厂函数 ================================

def create_ablation_model(model_name, **kwargs):
    """创建消融实验模型的工厂函数"""
    models = {
        'ultra_original': UltraAdvancedResistivityModel_Original,
        'ultra_no_wavelet': UltraAdvancedResistivityModel_NoWavelet,
        'ultra_no_fpn': UltraAdvancedResistivityModel_NoFPN,
        'ultra_no_attention': UltraAdvancedResistivityModel_NoAttention,
        'ultra_no_transformer': UltraAdvancedResistivityModel_NoTransformer,
        'ultra_basic_resnet': UltraAdvancedResistivityModel_BasicResNet,
        'ultra_simple_lstm': UltraAdvancedResistivityModel_SimpleLSTM,
        'baseline': BaselineModel,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name](**kwargs)


def get_ablation_configs():
    """获取所有消融实验模型的配置"""
    # 完整配置（用于完整模型）
    full_config = {
        'input_dim': 7,
        'hidden_dim': 128,
        'n_heads': 8,
        'n_transformer_layers': 4,
        'dropout': 0.1,
        'estimate_uncertainty': True
    }

    # 基础配置（不包含Transformer相关参数）
    base_config = {
        'input_dim': 7,
        'hidden_dim': 128,
        'dropout': 0.1,
        'estimate_uncertainty': True
    }

    configs = {
        'ultra_original': {**full_config, 'use_wavelet': True, 'use_fpn': True},
        'ultra_no_wavelet': {**full_config, 'use_fpn': True},
        'ultra_no_fpn': {**full_config, 'use_wavelet': True},
        'ultra_no_attention': {**full_config, 'use_wavelet': True, 'use_fpn': True},
        'ultra_no_transformer': {**base_config, 'use_wavelet': True, 'use_fpn': True},  # 不需要transformer参数
        'ultra_basic_resnet': base_config,
        'ultra_simple_lstm': base_config,
        'baseline': {'input_dim': 7, 'hidden_dim': 64, 'dropout': 0.1, 'estimate_uncertainty': True},
    }

    return configs


def analyze_ablation_model_complexity(model):
    """分析消融模型的复杂度"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'model_name': model.__class__.__name__,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024
    }


if __name__ == "__main__":
    # 示例：创建所有消融模型并分析复杂度
    configs = get_ablation_configs()

    print("消融实验模型复杂度分析:")
    print("=" * 80)

    for model_name, config in configs.items():
        try:
            model = create_ablation_model(model_name, **config)
            complexity = analyze_ablation_model_complexity(model)

            print(f"{model_name:20} | 参数量: {complexity['total_params']:>8,} | 大小: {complexity['model_size_mb']:>6.2f}MB")

            # 测试前向传播
            x = torch.randn(4, 128, 7)  # batch_size=4, seq_len=128, features=7
            with torch.no_grad():
                output = model(x)
                if isinstance(output, tuple):
                    print(f"{'':20} | 输出维度: mean{output[0].shape}, var{output[1].shape}")
                else:
                    print(f"{'':20} | 输出维度: {output.shape}")

        except Exception as e:
            print(f"{model_name:20} | 错误: {str(e)}")

        print("-" * 80)