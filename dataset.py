"""
数据预处理模块 - 三维感应测井电阻率数据处理
包含数据加载、预处理、标准化等功能
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


class MultiResistivityDataset(Dataset):
    """多电阻率测井数据集加载器"""

    def __init__(self,
                 data_folder,
                 seq_length=128,
                 interpolation='linear',
                 augmentation=True,
                 validation_mode=False):
        self.seq_length = seq_length
        self.interpolation = interpolation
        self.augmentation = augmentation and not validation_mode
        self.validation_mode = validation_mode

        # 加载数据文件
        self.file_list = glob.glob(os.path.join(data_folder, "*.csv"))
        if len(self.file_list) == 0:
            raise ValueError(f"没有在{data_folder}中找到CSV文件")

        print(f"找到{len(self.file_list)}个CSV文件")

        # 初始化存储
        self.features = []  # 三维电场数据
        self.labels = []  # 电阻率值
        self.original_lengths = []  # 记录原始序列长度

        # 加载和处理数据
        self._load_data()
        self._process_data()

        print(f"数据集初始化完成，共{len(self.labels)}个样本")

    def _load_data(self):
        """加载所有CSV文件并提取特征"""
        for file_path in self.file_list:
            try:
                df = pd.read_csv(file_path)
                # 检查必要的列是否存在
                required_cols = ['depth', 'Ex_real', 'Ex_imag', 'Ey_real', 'Ey_imag', 'Ez_real', 'Ez_imag', 'Resistivity']
                if not all(col in df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in df.columns]
                    print(f"警告: {os.path.basename(file_path)}缺少列: {missing}，跳过此文件")
                    continue

                # 按深度排序
                df = df.sort_values('depth')

                # 提取特征：6个分量(Ex_real, Ex_imag, Ey_real, Ey_imag, Ez_real, Ez_imag)
                features = df[['Ex_real', 'Ex_imag', 'Ey_real', 'Ey_imag', 'Ez_real', 'Ez_imag']].values

                # 添加深度列
                depth_column = df['depth'].values.reshape(-1, 1)
                features = np.hstack((features, depth_column))

                # 获取电阻率标签
                resistivity = df['Resistivity'].iloc[0]  # 使用第一个值，避免潜在问题

                # 检查数据是否包含NaN
                if np.isnan(features).any() or np.isnan(resistivity):
                    print(f"警告: {os.path.basename(file_path)}包含NaN值，跳过此文件")
                    continue

                # 检查数据是否包含无限值
                if np.isinf(features).any() or np.isinf(resistivity):
                    print(f"警告: {os.path.basename(file_path)}包含无限值，跳过此文件")
                    continue

                # 记录原始序列长度
                self.original_lengths.append(len(features))

                self.features.append(features)
                self.labels.append(resistivity)

            except Exception as e:
                print(f"处理文件{file_path}时出错: {str(e)}")

    def _process_data(self):
        """数据处理流程"""
        if not self.features:
            raise ValueError("没有成功加载任何数据")

        # 1. 统一序列长度
        self.features = [self._resample(f) for f in self.features]

        # 2. 转换为numpy数组
        self.features = np.stack(self.features)  # (num_samples, seq_length, features)
        self.labels = np.array(self.labels).reshape(-1, 1)

        # 检查并移除异常值
        self._clean_data()

        # 打印数据形状
        print(f"特征形状: {self.features.shape}, 标签形状: {self.labels.shape}")

        # 3. 数据标准化
        self._setup_scalers()

        # 4. 分析特征重要性 (可选)
        self._analyze_feature_importance()

    def _clean_data(self):
        """清理异常值"""
        # 检查特征中的极端值
        feature_means = np.mean(self.features, axis=(0, 1))
        feature_stds = np.std(self.features, axis=(0, 1))

        # 检测并替换极端值 (超过5个标准差)
        for i in range(self.features.shape[2]):
            mask = np.abs(self.features[:, :, i] - feature_means[i]) > 5 * feature_stds[i]
            self.features[:, :, i][mask] = feature_means[i]

        # 检查标签中的极端值
        label_mean = np.mean(self.labels)
        label_std = np.std(self.labels)

        # 打印标签分布信息
        print(f"标签分布 - 均值: {label_mean:.2f}, 标准差: {label_std:.2f}, 最小值: {np.min(self.labels):.2f}, 最大值: {np.max(self.labels):.2f}")

    def _resample(self, data):
        """使用线性插值将数据重采样到统一长度"""
        original_length = data.shape[0]
        x_orig = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, self.seq_length)

        resampled = np.zeros((self.seq_length, data.shape[1]))

        # 使用简单线性插值，更稳定
        for col in range(data.shape[1]):
            resampled[:, col] = np.interp(x_new, x_orig, data[:, col])

        return resampled

    def _setup_scalers(self):
        """初始化标准化器"""
        # 特征标准化
        num_features = self.features.shape[2]
        self.feat_scalers = []

        for i in range(num_features):
            # 使用标准缩放器，简单稳定
            scaler = StandardScaler()
            # 展平所有样本的该分量进行拟合
            scaler.fit(self.features[:, :, i].reshape(-1, 1))
            # 应用标准化
            self.features[:, :, i] = scaler.transform(
                self.features[:, :, i].reshape(-1, 1)
            ).reshape(self.features.shape[0], self.seq_length)
            self.feat_scalers.append(scaler)

        # 电阻率标准化 (对数变换)
        self.log_transform = True
        if self.log_transform:
            # 确保所有值为正
            min_resistivity = np.min(self.labels)
            if min_resistivity <= 0:
                self.offset = abs(min_resistivity) + 0.1
                self.labels = self.labels + self.offset
            else:
                self.offset = 0
            # 应用对数变换
            self.labels = np.log(self.labels)

        # 标准化
        self.resistivity_scaler = StandardScaler()
        self.labels = self.resistivity_scaler.fit_transform(self.labels)

    def _analyze_feature_importance(self):
        """分析特征重要性 """
        if len(self.features) < 10:
            return

        # 简单相关性分析
        feature_corrs = []
        for i in range(self.features.shape[2]):
            feature_means = np.nanmean(self.features[:, :, i], axis=1)
            corr = np.ma.corrcoef(
                np.ma.masked_invalid(feature_means),
                np.ma.masked_invalid(self.labels.flatten())
            )[0, 1]
            if not np.isnan(corr):
                feature_corrs.append((i, abs(corr)))
            else:
                feature_corrs.append((i, float('nan')))

        # 输出结果
        print("特征重要性排序 (索引, 相关性):")
        for idx, corr in feature_corrs:
            if not np.isnan(corr):
                print(f"  特征 {idx}: {corr:.4f}")
            else:
                print(f"  特征 {idx}: nan")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """返回标准化后的数据"""
        # 直接返回特征和标签，不进行数据增强
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])

    def get_inverse_transforms(self):
        """提供逆变换函数供评估时使用"""
        def inverse_transform_y(y):
            # 反标准化
            y_std = self.resistivity_scaler.inverse_transform(y)
            # 反对数变换
            if self.log_transform:
                y_exp = np.exp(y_std)
                # 如果有偏移量，减去它
                if hasattr(self, 'offset') and self.offset > 0:
                    y_exp = y_exp - self.offset
                return y_exp
            return y_std

        return inverse_transform_y