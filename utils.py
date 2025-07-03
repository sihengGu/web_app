# utils.py

import os
import random
import numpy as np
import torch
from torch.utils.data import random_split
import pickle # 【新增】导入pickle库

def set_seeds(seed=42):
    """设置随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_dataset_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=None):
    """
    安全的数据集划分函数

    Args:
        dataset: 数据集对象
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例（如果为None，自动计算）

    Returns:
        train_set, val_set, test_set: 划分后的数据集
    """
    total = len(dataset)

    # 自动计算测试集比例
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio

    # 确保比例总和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"数据集比例总和应为1.0，当前为{total_ratio}")

    # 计算初始划分大小
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    # 调整验证集大小
    if val_size == 0 and total > train_size:
        val_size = 1
        test_size = total - train_size - val_size

    # 调整测试集大小
    if test_size <= 0 and total > train_size + val_size:
        test_size = 1
        val_size = total - train_size - test_size
    
    # 如果总数太少，可能无法满足所有分割
    if train_size + val_size + test_size != total:
        train_size = total - 2
        val_size = 1
        test_size = 1


    # 最终确认
    assert train_size + val_size + test_size == total, "数据集划分错误"
    assert val_size > 0, "验证集不能为空"
    assert test_size > 0, "测试集不能为空"

    return random_split(dataset, [train_size, val_size, test_size])


def create_directories(*directories):
    """创建目录（如果不存在）"""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            # print(f"创建目录: {directory}") # 静默处理


def save_model_checkpoint(model, optimizer, epoch, loss, metrics, filepath):
    """
    保存模型检查点

    Args:
        model: 模型对象
        optimizer: 优化器对象
        epoch: 当前训练轮次
        loss: 当前损失值
        metrics: 评估指标字典
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'model_class': model.__class__.__name__,
        'model_config': getattr(model, 'config', {})
    }

    torch.save(checkpoint, filepath)
    print(f"模型检查点已保存至: {filepath}")


def load_model_checkpoint(model, optimizer, filepath, device=None):
    """
    加载模型检查点

    Args:
        model: 模型对象
        optimizer: 优化器对象
        filepath: 检查点文件路径
        device: 设备

    Returns:
        checkpoint: 检查点字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"模型检查点已从 {filepath} 加载")
    return checkpoint


def get_model_size(model):
    """
    计算模型参数量

    Args:
        model: 模型对象

    Returns:
        total_params, trainable_params: 总参数量和可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def print_model_info(model):
    """打印模型信息"""
    total_params, trainable_params = get_model_size(model)

    print("=== 模型信息 ===")
    print(f"模型类型: {model.__class__.__name__}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


def check_device_info():
    """检查并打印设备信息"""
    if torch.cuda.is_available():
        print(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("CUDA不可用，使用CPU")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_data_path(data_path, required_files=None):
    """
    验证数据路径和文件

    Args:
        data_path: 数据文件夹路径
        required_files: 必需的文件列表（可选）

    Returns:
        is_valid: 路径是否有效
        message: 验证消息
    """
    if not os.path.exists(data_path):
        return False, f"数据路径 {data_path} 不存在"

    if not os.path.isdir(data_path):
        return False, f"{data_path} 不是一个目录"

    # 检查CSV文件
    import glob
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    if not csv_files:
        return False, f"在 {data_path} 中未找到CSV文件"

    # 检查必需文件（如果指定）
    if required_files:
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(data_path, file)):
                missing_files.append(file)

        if missing_files:
            return False, f"缺少必需文件: {missing_files}"

    return True, f"数据路径验证成功，找到 {len(csv_files)} 个CSV文件"


def create_experiment_directory(base_dir="experiments"):
    """
    创建实验目录

    Args:
        base_dir: 基础目录名

    Returns:
        experiment_dir: 创建的实验目录路径
    """
    import datetime

    # 创建基于时间戳的实验目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"exp_{timestamp}")

    # 创建相关子目录
    subdirs = ['models', 'logs', 'plots', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    print(f"实验目录创建成功: {experiment_dir}")
    return experiment_dir


def setup_logging(log_file=None, level='INFO'):
    """
    设置日志记录

    Args:
        log_file: 日志文件路径（可选）
        level: 日志级别
    """
    import logging

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建根日志记录器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 添加文件处理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        print(f"日志将保存至: {log_file}")


def cleanup_checkpoints(checkpoint_dir, keep_last=3):
    """
    清理旧的检查点文件，只保留最新的几个

    Args:
        checkpoint_dir: 检查点目录
        keep_last: 保留最新的几个检查点
    """
    import glob

    # 查找所有检查点文件
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))

    if len(checkpoint_files) <= keep_last:
        return

    # 按修改时间排序
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    # 删除多余的检查点
    for checkpoint in checkpoint_files[keep_last:]:
        os.remove(checkpoint)
        print(f"删除旧检查点: {checkpoint}")


def calculate_memory_usage():
    """计算当前内存使用情况"""
    import psutil

    # 系统内存
    memory = psutil.virtual_memory()
    print(f"系统内存使用: {memory.percent}% ({memory.used / 1024 ** 3:.1f}GB / {memory.total / 1024 ** 3:.1f}GB)")

    # GPU内存（如果可用）
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            cached = torch.cuda.memory_reserved(i) / 1024 ** 3
            total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"GPU {i} 内存使用: 已分配 {allocated:.1f}GB, 缓存 {cached:.1f}GB, 总计 {total:.1f}GB")


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}分{seconds:.1f}秒"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}小时{int(minutes)}分{seconds:.1f}秒"


class Timer:
    """计时器工具类"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """开始计时"""
        import time
        self.start_time = time.time()

    def stop(self):
        """停止计时"""
        import time
        self.end_time = time.time()

    def elapsed(self):
        """获取经过的时间"""
        if self.start_time is None:
            return 0
        if self.end_time is None:
            import time
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def config_to_string(config_dict):
    """将配置字典转换为字符串表示"""
    if not config_dict:
        return ""

    items = []
    for key, value in config_dict.items():
        if isinstance(value, dict):
            items.append(f"{key}: {config_to_string(value)}")
        else:
            items.append(f"{key}: {value}")

    return "{" + ", ".join(items) + "}"

# 【新增】保存scaler对象的函数
def save_scalers(dataset, output_path: str):
    """
    保存数据集中的scaler对象
    
    Args:
        dataset: MultiResistivityDataset的实例
        output_path: 保存 .pkl 文件的路径
    """
    if not hasattr(dataset, 'feat_scalers') or not hasattr(dataset, 'resistivity_scaler'):
        print("警告: 数据集中未找到 'feat_scalers' 或 'resistivity_scaler'。跳过保存。")
        return

    scalers_to_save = {
        'feat_scalers': dataset.feat_scalers,
        'resistivity_scaler': dataset.resistivity_scaler,
        'log_transform': getattr(dataset, 'log_transform', False),
        'offset': getattr(dataset, 'offset', 0)
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(scalers_to_save, f)
    
    print(f"Scaler对象已保存至: {output_path}")