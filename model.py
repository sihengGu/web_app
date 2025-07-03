"""
模型架构模块 - 三维感应测井电阻率反演深度学习模型
包含主要的神经网络模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMultiScaleModel(nn.Module):
    """简化版多尺度电阻率预测模型 - 提高数值稳定性"""

    def __init__(self,
                 input_dim=7,
                 hidden_dim=64,
                 dropout=0.1,
                 estimate_uncertainty=False):
        super().__init__()

        self.estimate_uncertainty = estimate_uncertainty

        # 输入映射层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 全局上下文表示
        self.context_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
        """初始化模型权重 - 修复版"""
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
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.dropout(x)

        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 全局上下文 - 使用平均池化
        global_context = torch.mean(lstm_out, dim=1)
        global_context = self.context_layer(global_context)

        # 回归输出
        output = self.regression_head(global_context)

        if self.estimate_uncertainty:
            # 输出均值和方差估计
            mean, log_var = output.chunk(2, dim=1)
            # 确保方差为正 - 使用softplus保持数值稳定性
            var = F.softplus(log_var) + 1e-6
            return mean, var
        else:
            # 仅输出均值
            return output


class ResNet1DBlock(nn.Module):
    """一维残差块 - 用于更复杂的模型"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AdvancedResistivityModel(nn.Module):
    """基于残差网络的更复杂模型 - 可选使用"""

    def __init__(self, input_dim=7, hidden_dim=64, num_blocks=3, dropout=0.1):
        super().__init__()

        # 将序列数据转换为卷积格式
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 残差块序列
        self.res_blocks = nn.ModuleList([
            ResNet1DBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # 输入投影
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)

        # 转置为卷积格式
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)

        # 通过残差块
        for block in self.res_blocks:
            x = block(x)

        # 全局平均池化
        x = self.global_pool(x).squeeze(-1)  # (batch, hidden_dim)

        # 最终预测
        output = self.classifier(x)
        return output


class EnsembleModel(nn.Module):
    """集成模型 - 结合多个子模型的预测"""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # 平均集成
        ensemble_output = torch.stack(outputs, dim=0).mean(dim=0)
        return ensemble_output

    def forward_with_uncertainty(self, x):
        """返回预测均值和不确定性"""
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        outputs = torch.stack(outputs, dim=0)  # (num_models, batch_size, 1)

        # 计算均值和方差
        mean = outputs.mean(dim=0)
        var = outputs.var(dim=0)

        return mean, var