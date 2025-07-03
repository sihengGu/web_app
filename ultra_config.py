"""
增强配置模块 - 支持超级增强版模型的配置
添加了对新模型架构和前沿技术的配置支持
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class DataConfig:
    """数据相关配置"""
    data_path: str = "./dataset"
    seq_length: int = 128
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 16
    num_workers: int = 0
    shuffle: bool = True
    augmentation: bool = False
    interpolation: str = 'linear'

    # 新增数据增强选项
    noise_level: float = 0.01
    scale_range: tuple = (0.9, 1.1)
    time_shift_range: int = 5


@dataclass
class ModelConfig:
    """模型相关配置"""
    model_type: str = 'OptimizedResistivityModel'
    input_dim: int = 7
    hidden_dim: int = 96
    dropout: float = 0.1
    estimate_uncertainty: bool = True

    # 原有模型配置
    num_blocks: int = 3

    # 优化模型配置
    complexity_level: str = 'medium'
    num_conv_blocks: int = 2
    num_attention_blocks: int = 2
    n_heads: int = 4
    use_ensemble: bool = False
    ensemble_method: str = 'mean'

    # 超级模型配置
    n_transformer_layers: int = 4
    use_wavelet: bool = False
    use_fpn: bool = False

    # 注意力机制配置
    attention_dropout: float = 0.1
    use_channel_attention: bool = True
    use_spatial_attention: bool = True
    use_se_block: bool = True

    # Transformer配置
    transformer_dim_feedforward: int = 512
    transformer_activation: str = 'relu'
    transformer_norm_first: bool = False

    # 小波变换配置
    wavelet_levels: int = 3
    wavelet_type: str = 'db4'

    # 特征金字塔配置
    fpn_out_channels: int = 64
    fpn_num_levels: int = 3

    # 【新增】为带噪模型的配置增加一个可选字段
    noise_config: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfig:
    """训练相关配置"""
    epochs: int = 200  # 适中的训练轮次
    learning_rate: float = 1e-3  # 提高学习率
    weight_decay: float = 1e-5  # 降低正则化
    patience: int = 25  # 适中的早停耐心值
    optimizer: str = 'AdamW'
    scheduler: str = 'ReduceLROnPlateau'
    loss_type: str = 'uncertainty'
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # 训练策略
    warmup_epochs: int = 5  # 减少热身轮次
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0

    # 学习率调度器参数
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'mode': 'min',
        'factor': 0.5,  # 更明显的学习率衰减
        'patience': 10,  # 减少调度器耐心值
        'verbose': True,
        'min_lr': 1e-6  # 提高最小学习率
    })

    # 优化器配置
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'amsgrad': False
    })

    # 损失函数权重
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'mse': 1.0,
        'uncertainty': 1.0,
        'regularization': 0.001  # 减少正则化权重
    })


@dataclass
class ExperimentConfig:
    """实验相关配置"""
    experiment_name: str = "ultra_resistivity_inversion"
    save_dir: str = "./experiments"
    save_best_model: bool = True
    save_final_model: bool = True
    save_history: bool = True
    save_predictions: bool = True
    save_attention_maps: bool = True  # 保存注意力图
    plot_results: bool = True
    print_detailed_results: bool = True

    # 随机种子设置
    seed: int = 42
    deterministic: bool = True

    # 新增实验跟踪
    use_tensorboard: bool = True
    use_wandb: bool = False  # Weights & Biases实验跟踪
    wandb_project: str = "resistivity_inversion"

    # 模型检查点
    save_checkpoint_every: int = 50  # 每N个epoch保存检查点
    max_checkpoints: int = 5  # 最多保留的检查点数量

    # 验证和测试
    validate_every: int = 1  # 每N个epoch验证一次
    test_during_training: bool = False

    # GPU相关
    mixed_precision: bool = True  # 使用混合精度训练
    gradient_accumulation_steps: int = 1  # 梯度累积步数


@dataclass
class Config:
    """主配置类，整合所有配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'experiment': self.experiment.__dict__
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置对象"""
        config = cls()

        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'experiment' in config_dict:
            config.experiment = ExperimentConfig(**config_dict['experiment'])

        return config

    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        import json

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

        print(f"配置已保存至: {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        """从文件加载配置"""
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def update_with_args(self, args):
        """使用命令行参数更新配置"""
        # 数据配置
        if hasattr(args, 'data_path') and args.data_path:
            self.data.data_path = args.data_path
        if hasattr(args, 'batch_size') and args.batch_size:
            self.data.batch_size = args.batch_size
        if hasattr(args, 'seq_length') and args.seq_length:
            self.data.seq_length = args.seq_length

        # 模型配置
        if hasattr(args, 'model_type') and args.model_type:
            self.model.model_type = args.model_type
        if hasattr(args, 'hidden_dim') and args.hidden_dim:
            self.model.hidden_dim = args.hidden_dim
        if hasattr(args, 'dropout') and args.dropout:
            self.model.dropout = args.dropout
        if hasattr(args, 'n_heads') and args.n_heads:
            self.model.n_heads = args.n_heads
        if hasattr(args, 'n_transformer_layers') and args.n_transformer_layers:
            self.model.n_transformer_layers = args.n_transformer_layers

        # 训练配置
        if hasattr(args, 'epochs') and args.epochs:
            self.training.epochs = args.epochs
        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.training.learning_rate = args.learning_rate
        if hasattr(args, 'patience') and args.patience:
            self.training.patience = args.patience

        # 实验配置
        if hasattr(args, 'experiment_name') and args.experiment_name:
            self.experiment.experiment_name = args.experiment_name
        if hasattr(args, 'seed') and args.seed:
            self.experiment.seed = args.seed


# 预定义的配置模板
def get_improved_simple_config() -> Config:
    """改进版简单模型配置 - 渐进式优化"""
    config = Config()
    config.model.model_type = 'ImprovedSimpleModel'
    config.model.hidden_dim = 64
    config.model.dropout = 0.1
    config.model.estimate_uncertainty = False  # 先不用不确定性
    config.training.epochs = 200
    config.training.learning_rate = 1e-3
    config.training.weight_decay = 1e-5
    config.training.patience = 25
    config.training.loss_type = 'mse'  # 使用稳定的MSE
    config.data.batch_size = 32
    return config


def get_enhanced_simple_config() -> Config:
    """增强版简单模型配置"""
    config = Config()
    config.model.model_type = 'EnhancedSimpleModel'
    config.model.hidden_dim = 64
    config.model.dropout = 0.1
    config.model.estimate_uncertainty = False  # 先不用不确定性
    config.training.epochs = 200
    config.training.learning_rate = 8e-4
    config.training.weight_decay = 1e-5
    config.training.patience = 30
    config.training.loss_type = 'mse'  # 使用稳定的MSE
    config.data.batch_size = 32
    return config


def get_stable_uncertainty_config() -> Config:
    """稳定的不确定性配置"""
    config = Config()
    config.model.model_type = 'ImprovedSimpleModel'
    config.model.hidden_dim = 64
    config.model.dropout = 0.1
    config.model.estimate_uncertainty = True
    config.training.epochs = 200
    config.training.learning_rate = 8e-4
    config.training.weight_decay = 1e-5
    config.training.patience = 25
    config.training.loss_type = 'stable_uncertainty'  # 使用稳定的不确定性损失
    config.data.batch_size = 32
    return config


def get_original_simple_config() -> Config:
    """原始简单模型配置 - 作为基准对比"""
    config = Config()
    config.model.model_type = 'SimpleMultiScaleModel'
    config.model.hidden_dim = 64
    config.model.dropout = 0.2
    config.model.estimate_uncertainty = True
    config.training.epochs = 200
    config.training.learning_rate = 1e-4
    config.training.weight_decay = 1e-4
    config.training.patience = 20
    config.training.loss_type = 'uncertainty'
    config.data.batch_size = 16
    return config


def get_ultra_fast_config() -> Config:
    """超级快速测试配置"""
    config = Config()
    config.model.model_type = 'UltraAdvancedResistivityModel'
    config.model.hidden_dim = 64
    config.model.n_heads = 4
    config.model.n_transformer_layers = 2
    config.model.use_wavelet = False
    config.model.use_fpn = False
    config.training.epochs = 30
    config.training.patience = 10
    config.data.batch_size = 8
    return config


def get_ultra_production_config() -> Config:
    """超级生产环境配置"""
    config = Config()
    config.model.model_type = 'UltraAdvancedResistivityModel'
    config.model.hidden_dim = 256
    config.model.n_heads = 16
    config.model.n_transformer_layers = 6
    config.model.use_wavelet = True
    config.model.use_fpn = True
    config.training.epochs = 500
    config.training.patience = 50
    config.training.learning_rate = 1e-5
    config.data.batch_size = 32
    config.experiment.mixed_precision = True
    return config


def get_ultra_ensemble_config() -> Config:
    """超级集成模型配置"""
    config = Config()
    config.model.model_type = 'EnsembleUltraModel'
    config.model.use_ensemble = True
    config.model.hidden_dim = 128
    config.model.n_heads = 8
    config.model.n_transformer_layers = 4
    config.training.epochs = 400
    config.training.patience = 40
    config.training.learning_rate = 3e-5
    config.data.batch_size = 16
    return config


def get_ultra_attention_config() -> Config:
    """注重注意力机制的配置"""
    config = Config()
    config.model.model_type = 'UltraAdvancedResistivityModel'
    config.model.hidden_dim = 192
    config.model.n_heads = 12
    config.model.n_transformer_layers = 8
    config.model.use_channel_attention = True
    config.model.use_spatial_attention = True
    config.model.use_se_block = True
    config.model.attention_dropout = 0.05
    config.training.epochs = 350
    config.training.patience = 35
    return config


def get_ultra_wavelet_config() -> Config:
    """注重小波变换的配置"""
    config = Config()
    config.model.model_type = 'UltraAdvancedResistivityModel'
    config.model.hidden_dim = 160
    config.model.use_wavelet = True
    config.model.wavelet_levels = 4
    config.model.use_fpn = False
    config.model.n_transformer_layers = 3
    config.training.epochs = 300
    config.training.patience = 30
    return config


def get_optimized_light_config() -> Config:
    """轻量级优化配置 - 适合小数据集"""
    config = Config()
    config.model.model_type = 'OptimizedResistivityModel'
    config.model.complexity_level = 'light'
    config.model.hidden_dim = 64
    config.model.num_conv_blocks = 1
    config.model.num_attention_blocks = 1
    config.model.n_heads = 4
    config.model.dropout = 0.1
    config.training.epochs = 150
    config.training.learning_rate = 2e-3
    config.training.patience = 20
    config.data.batch_size = 16
    return config


def get_optimized_medium_config() -> Config:
    """中等复杂度优化配置 - 推荐使用"""
    config = Config()
    config.model.model_type = 'OptimizedResistivityModel'
    config.model.complexity_level = 'medium'
    config.model.hidden_dim = 96
    config.model.num_conv_blocks = 2
    config.model.num_attention_blocks = 2
    config.model.n_heads = 4
    config.model.dropout = 0.1
    config.training.epochs = 200
    config.training.learning_rate = 1e-3
    config.training.patience = 25
    config.data.batch_size = 32
    return config


def get_optimized_strong_config() -> Config:
    """强力优化配置"""
    config = Config()
    config.model.model_type = 'OptimizedResistivityModel'
    config.model.complexity_level = 'strong'
    config.model.hidden_dim = 128
    config.model.num_conv_blocks = 3
    config.model.num_attention_blocks = 2
    config.model.n_heads = 8
    config.model.dropout = 0.15
    config.training.epochs = 250
    config.training.learning_rate = 8e-4
    config.training.patience = 30
    config.data.batch_size = 32
    return config


def get_optimized_ensemble_config() -> Config:
    """优化集成模型配置"""
    config = Config()
    config.model.model_type = 'EnsembleOptimizedModel'
    config.model.use_ensemble = True
    config.model.ensemble_method = 'weighted'
    config.training.epochs = 200
    config.training.learning_rate = 8e-4
    config.training.patience = 30
    config.data.batch_size = 24
    return config
    """超级快速测试配置"""
    config = Config()
    config.model.model_type = 'UltraAdvancedResistivityModel'
    config.model.hidden_dim = 64
    config.model.n_heads = 4
    config.model.n_transformer_layers = 2
    config.model.use_wavelet = False
    config.model.use_fpn = False
    config.training.epochs = 30
    config.training.patience = 10
    config.data.batch_size = 8
    return config


def get_ultra_production_config() -> Config:
    """超级生产环境配置"""
    config = Config()
    config.model.model_type = 'UltraAdvancedResistivityModel'
    config.model.hidden_dim = 256
    config.model.n_heads = 16
    config.model.n_transformer_layers = 6
    config.model.use_wavelet = True
    config.model.use_fpn = True
    config.training.epochs = 500
    config.training.patience = 50
    config.training.learning_rate = 1e-5
    config.data.batch_size = 32
    config.experiment.mixed_precision = True
    return config


def get_ultra_ensemble_config() -> Config:
    """超级集成模型配置"""
    config = Config()
    config.model.model_type = 'EnsembleUltraModel'
    config.model.use_ensemble = True
    config.model.hidden_dim = 128
    config.model.n_heads = 8
    config.model.n_transformer_layers = 4
    config.training.epochs = 400
    config.training.patience = 40
    config.training.learning_rate = 3e-5
    config.data.batch_size = 16
    return config


def get_ultra_attention_config() -> Config:
    """注重注意力机制的配置"""
    config = Config()
    config.model.model_type = 'UltraAdvancedResistivityModel'
    config.model.hidden_dim = 192
    config.model.n_heads = 12
    config.model.n_transformer_layers = 8
    config.model.use_channel_attention = True
    config.model.use_spatial_attention = True
    config.model.use_se_block = True
    config.model.attention_dropout = 0.05
    config.training.epochs = 350
    config.training.patience = 35
    return config


def get_ultra_wavelet_config() -> Config:
    """注重小波变换的配置"""
    config = Config()
    config.model.model_type = 'UltraAdvancedResistivityModel'
    config.model.hidden_dim = 160
    config.model.use_wavelet = True
    config.model.wavelet_levels = 4
    config.model.use_fpn = False
    config.model.n_transformer_layers = 3
    config.training.epochs = 300
    config.training.patience = 30
    return config


def get_optimized_light_config() -> Config:
    """轻量级优化配置 - 适合小数据集"""
    config = Config()
    config.model.model_type = 'OptimizedResistivityModel'
    config.model.complexity_level = 'light'
    config.model.hidden_dim = 64
    config.model.num_conv_blocks = 1
    config.model.num_attention_blocks = 1
    config.model.n_heads = 4
    config.model.dropout = 0.1
    config.training.epochs = 150
    config.training.learning_rate = 2e-3
    config.training.patience = 20
    config.data.batch_size = 16
    return config


def get_optimized_medium_config() -> Config:
    """中等复杂度优化配置 - 推荐使用"""
    config = Config()
    config.model.model_type = 'OptimizedResistivityModel'
    config.model.complexity_level = 'medium'
    config.model.hidden_dim = 96
    config.model.num_conv_blocks = 2
    config.model.num_attention_blocks = 2
    config.model.n_heads = 4
    config.model.dropout = 0.1
    config.training.epochs = 200
    config.training.learning_rate = 1e-3
    config.training.patience = 25
    config.data.batch_size = 32
    return config


def get_optimized_strong_config() -> Config:
    """强力优化配置"""
    config = Config()
    config.model.model_type = 'OptimizedResistivityModel'
    config.model.complexity_level = 'strong'
    config.model.hidden_dim = 128
    config.model.num_conv_blocks = 3
    config.model.num_attention_blocks = 2
    config.model.n_heads = 8
    config.model.dropout = 0.15
    config.training.epochs = 250
    config.training.learning_rate = 8e-4
    config.training.patience = 30
    config.data.batch_size = 32
    return config


def get_optimized_ensemble_config() -> Config:
    """优化集成模型配置"""
    config = Config()
    config.model.model_type = 'EnsembleOptimizedModel'
    config.model.use_ensemble = True
    config.model.ensemble_method = 'weighted'
    config.training.epochs = 200
    config.training.learning_rate = 8e-4
    config.training.patience = 30
    config.data.batch_size = 24
    return config



from typing import Tuple, List, Dict

def validate_config(config: Config) -> Tuple[bool, List[str]]:
    """验证配置的有效性 - 增强版"""
    errors = []

    # 验证数据配置
    if not os.path.exists(config.data.data_path):
        errors.append(f"数据路径不存在: {config.data.data_path}")

    if config.data.seq_length <= 0:
        errors.append("序列长度必须大于0")

    if config.data.batch_size <= 0:
        errors.append("批大小必须大于0")

    if not (0 < config.data.train_ratio < 1):
        errors.append("训练集比例必须在0和1之间")

    if abs(config.data.train_ratio + config.data.val_ratio + config.data.test_ratio - 1.0) > 1e-6:
        errors.append("数据集比例总和必须等于1.0")

    # 验证模型配置
    if config.model.input_dim <= 0:
        errors.append("输入维度必须大于0")

    if config.model.hidden_dim <= 0:
        errors.append("隐藏层维度必须大于0")

    if not (0 <= config.model.dropout < 1):
        errors.append("Dropout比例必须在0和1之间")

    # 验证超级模型特定配置
    if config.model.model_type == 'UltraAdvancedResistivityModel':
        if config.model.n_heads <= 0:
            errors.append("注意力头数必须大于0")

        if config.model.hidden_dim % config.model.n_heads != 0:
            errors.append("隐藏维度必须能被注意力头数整除")

        if config.model.n_transformer_layers <= 0:
            errors.append("Transformer层数必须大于0")

        if config.model.wavelet_levels <= 0:
            errors.append("小波分解层数必须大于0")

    # 验证训练配置
    if config.training.epochs <= 0:
        errors.append("训练轮次必须大于0")

    if config.training.learning_rate <= 0:
        errors.append("学习率必须大于0")

    if config.training.patience <= 0:
        errors.append("早停耐心值必须大于0")

    if config.training.weight_decay < 0:
        errors.append("权重衰减不能为负数")

    if config.training.warmup_epochs < 0:
        errors.append("热身轮次不能为负数")

    if config.training.gradient_accumulation_steps <= 0:
        errors.append("梯度累积步数必须大于0")

    # 验证实验配置
    if not config.experiment.experiment_name:
        errors.append("实验名称不能为空")

    if config.experiment.save_checkpoint_every <= 0:
        errors.append("检查点保存间隔必须大于0")

    if config.experiment.max_checkpoints <= 0:
        errors.append("最大检查点数量必须大于0")

    return len(errors) == 0, errors


# 默认配置实例
DEFAULT_CONFIG = Config()

# 扩展的配置字典
CONFIG_TEMPLATES = {
    # 原始和改进的简单模型
    'original_simple': get_original_simple_config(),
    'improved_simple': get_improved_simple_config(),  # 推荐首先尝试
    'enhanced_simple': get_enhanced_simple_config(),
    'stable_uncertainty': get_stable_uncertainty_config(),

    # 传统模型
    'default': Config(),
    'simple': Config(),  # 使用原始SimpleMultiScaleModel

    # 优化模型
    'optimized_light': get_optimized_light_config(),
    'optimized_medium': get_optimized_medium_config(),
    'optimized_strong': get_optimized_strong_config(),
    'optimized_ensemble': get_optimized_ensemble_config(),

    # 超级模型（仅适合大数据集）
    'ultra_fast': get_ultra_fast_config(),
    'ultra_production': get_ultra_production_config(),
    'ultra_ensemble': get_ultra_ensemble_config(),
    'ultra_attention': get_ultra_attention_config(),
    'ultra_wavelet': get_ultra_wavelet_config(),
}