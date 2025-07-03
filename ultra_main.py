# ultra_main.py (修改后)

# 强制修复matplotlib
try:
    from force_matplotlib_fix import *
    print("Force matplotlib fix loaded successfully")
except ImportError:
    import os
    import warnings
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['NUMEXPR_MAX_THREADS'] = '8'
    os.environ['MPLBACKEND'] = 'Agg'
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use('Agg', force=True)
    print("Manual matplotlib fix applied")

import os
import sys
import argparse
import warnings
import torch
import time
from torch.utils.data import DataLoader

# 导入自定义模块
from dataset import MultiResistivityDataset
from model import SimpleMultiScaleModel, AdvancedResistivityModel
from ultra_model import (
    UltraAdvancedResistivityModel,
    EnsembleUltraModel,
    create_ultra_model,
    analyze_model_complexity
)
from trainer import train_model, Trainer
from evaluator import evaluate, plot_training_history, Evaluator
from utils import (
    set_seeds, safe_dataset_split, create_directories,
    save_model_checkpoint, check_device_info, validate_data_path,
    create_experiment_directory, setup_logging, Timer, format_time,
    save_scalers # 【修改】导入新的scaler保存函数
)
from ultra_config import Config, CONFIG_TEMPLATES, validate_config
from improved_simple_model import ImprovedSimpleModel, EnhancedSimpleModel
from optimized_model import OptimizedResistivityModel, EnsembleOptimizedModel

def parse_arguments():
    """解析命令行参数 - 增强版"""
    parser = argparse.ArgumentParser(description="三维感应测井电阻率反演深度学习模型")

    # 基本参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--template', type=str, choices=list(CONFIG_TEMPLATES.keys()),
                       default='default', help='配置模板选择')
    parser.add_argument('--data_path', type=str, help='数据集路径')
    parser.add_argument('--experiment_name', type=str, help='实验名称')

    # 模型参数
    parser.add_argument('--model_type', type=str, help='模型类型')
    parser.add_argument('--hidden_dim', type=int, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, help='Dropout比例')

    # 超级模型特定参数
    parser.add_argument('--n_heads', type=int, help='多头注意力头数')
    parser.add_argument('--n_transformer_layers', type=int, help='Transformer层数')
    parser.add_argument('--use_wavelet', action='store_true', help='使用小波变换')
    parser.add_argument('--use_fpn', action='store_true', help='使用特征金字塔网络')
    parser.add_argument('--use_ensemble', action='store_true', help='使用集成模型')

    # 训练参数
    parser.add_argument('--epochs', type=int, help='训练轮次')
    parser.add_argument('--batch_size', type=int, help='批大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--patience', type=int, help='早停耐心值')
    parser.add_argument('--warmup_epochs', type=int, help='热身训练轮次')
    parser.add_argument('--mixed_precision', action='store_true', help='使用混合精度训练')

    # 其他参数
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'],
                       default='auto', help='计算设备')
    parser.add_argument('--no_plot', action='store_true', help='不绘制结果图表')
    parser.add_argument('--save_dir', type=str, help='结果保存目录')

    # 模式选择
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict', 'analyze'],
                       default='train', help='运行模式')
    parser.add_argument('--model_path', type=str, help='预训练模型路径（用于评估或预测）')

    return parser.parse_args()


def load_config(args):
    """加载配置 - 增强版"""
    if args.config:
        config = Config.load_from_file(args.config)
        print(f"从文件加载配置: {args.config}")
    else:
        config = CONFIG_TEMPLATES.get(args.template, Config())
        print(f"使用配置模板: {args.template}")

    config.update_with_args(args)
    is_valid, errors = validate_config(config)
    if not is_valid:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    return config


def create_model(config: Config):
    """根据配置创建模型 - 增强版"""
    model_type = config.model.model_type
    model_params = config.model.__dict__

    # 移除不属于构造函数的参数
    if 'model_type' in model_params:
        model_params.pop('model_type')

    factories = {
        'SimpleMultiScaleModel': SimpleMultiScaleModel,
        'AdvancedResistivityModel': AdvancedResistivityModel,
        'UltraAdvancedResistivityModel': UltraAdvancedResistivityModel,
        'ImprovedSimpleModel': ImprovedSimpleModel,
        'EnhancedSimpleModel': EnhancedSimpleModel,
        'OptimizedResistivityModel': OptimizedResistivityModel,
    }

    if model_type in factories:
        # 只传递模型构造函数接受的参数
        import inspect
        sig = inspect.signature(factories[model_type].__init__)
        valid_params = {k: v for k, v in model_params.items() if k in sig.parameters}
        return factories[model_type](**valid_params)
    elif model_type == 'EnsembleUltraModel':
        from ultra_model import get_ultra_model_configs
        return EnsembleUltraModel(get_ultra_model_configs())
    elif model_type == 'EnsembleOptimizedModel':
        from optimized_model import get_optimized_configs, EnsembleOptimizedModel
        configs = get_optimized_configs()
        return EnsembleOptimizedModel([configs['light'], configs['medium']], config.model.ensemble_method)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def setup_experiment(config: Config):
    """设置实验环境 - 增强版"""
    set_seeds(config.experiment.seed)
    device = check_device_info()

    experiment_dir = os.path.join(config.experiment.save_dir, config.experiment.experiment_name)
    create_directories(experiment_dir, os.path.join(experiment_dir, 'models'), os.path.join(experiment_dir, 'plots'))

    # 【修改】保存config.json文件
    config_save_path = os.path.join(experiment_dir, 'config.json')
    config.save_to_file(config_save_path)
    print(f"完整配置已保存至: {config_save_path}")

    tb_writer = None
    if config.experiment.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(experiment_dir, 'tensorboard')
            tb_writer = SummaryWriter(tb_dir)
            print(f"TensorBoard日志保存至: {tb_dir}")
        except ImportError:
            print("警告: 无法导入TensorBoard，将跳过TensorBoard日志")

    return device, experiment_dir, tb_writer

def train_pipeline(config: Config, device, experiment_dir, tb_writer=None):
    """训练流程 - 增强版"""
    print("=== 开始训练流程 ===")
    is_valid, message = validate_data_path(config.data.data_path)
    if not is_valid:
        print(f"数据验证失败: {message}")
        sys.exit(1)
    print(message)

    print("加载数据集...")
    dataset = MultiResistivityDataset(
        data_folder=config.data.data_path,
        seq_length=config.data.seq_length,
        interpolation=config.data.interpolation,
        augmentation=config.data.augmentation
    )

    train_set, val_set, test_set = safe_dataset_split(
        dataset,
        config.data.train_ratio,
        config.data.val_ratio,
        config.data.test_ratio
    )
    print(f"数据划分完成 - 训练集: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=config.data.batch_size, shuffle=config.data.shuffle, num_workers=config.data.num_workers)
    val_loader = DataLoader(val_set, batch_size=config.data.batch_size, num_workers=config.data.num_workers)
    test_loader = DataLoader(test_set, batch_size=config.data.batch_size, num_workers=config.data.num_workers)

    print("创建模型...")
    model = create_model(config)
    print(f"模型创建完成: {model.__class__.__name__}")

    model_save_path = os.path.join(experiment_dir, 'best_model.pth')

    # 使用增强训练器
    from ultra_trainer import UltraTrainer
    trainer = UltraTrainer(model, train_loader, val_loader, device)
    trainer.setup_training(
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        loss_type=config.training.loss_type
    )
    if config.experiment.mixed_precision:
        trainer.setup_mixed_precision()
    if tb_writer:
        trainer.tb_writer = tb_writer

    print("开始训练...")
    history = trainer.train(
        epochs=config.training.epochs,
        patience=config.training.patience,
        save_path=model_save_path
    )
    trained_model = trainer.model

    # 【修改】在训练结束后，保存scaler对象
    scaler_save_path = model_save_path.replace('.pth', '_scalers.pkl')
    save_scalers(dataset, scaler_save_path)

    # 评估和绘图
    print("评估模型...")
    metrics = evaluate(
        model=trained_model,
        test_loader=test_loader,
        dataset=dataset,
        device=device,
        save_dir=os.path.join(experiment_dir, 'plots')
    )

    if config.experiment.plot_results:
        plot_training_history(history, save_dir=os.path.join(experiment_dir, 'plots'))

    if tb_writer:
        tb_writer.close()

    return trained_model, metrics, history

def main():
    """主函数"""
    args = parse_arguments()
    config = load_config(args)
    device, experiment_dir, tb_writer = setup_experiment(config)

    if args.mode == 'train':
        train_pipeline(config, device, experiment_dir, tb_writer)
    else:
        print(f"模式 '{args.mode}' 尚未完全实现。")

if __name__ == "__main__":
    main()