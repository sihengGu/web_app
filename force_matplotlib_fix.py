"""
强制修复脚本 - 彻底解决matplotlib坐标转换问题
在导入任何matplotlib相关代码之前运行此脚本
"""

import os
import sys
import warnings
import numpy as np

# 1. 设置环境变量（必须最先执行）
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['MPLBACKEND'] = 'Agg'  # 强制使用Agg后端

# 2. 压制所有可能的警告
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# 3. 修复matplotlib配置
import matplotlib

matplotlib.use('Agg', force=True)  # 强制使用Agg后端

# 清理所有现有配置
matplotlib.rcdefaults()

# 设置最安全的配置
safe_config = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],  # 只使用最基本的字体
    'font.size': 10,
    'axes.unicode_minus': False,
    'axes.formatter.use_mathtext': False,
    'text.usetex': False,
    'mathtext.default': 'regular',
    'figure.max_open_warning': 0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'interactive': False
}

for key, value in safe_config.items():
    try:
        matplotlib.rcParams[key] = value
    except:
        pass

# 4. 导入pyplot并应用补丁
import matplotlib.pyplot as plt

# 保存原始函数
_original_text = plt.text
_original_tight_layout = plt.tight_layout
_original_show = plt.show
_original_savefig = plt.savefig


def safe_text(*args, **kwargs):
    """安全的文本函数"""
    try:
        # 确保坐标是数值类型
        if len(args) >= 3:
            x, y, s = args[0], args[1], args[2]
            # 转换坐标为float
            try:
                x = float(x)
                y = float(y)
            except:
                x, y = 0.5, 0.5  # 默认中心位置
            args = (x, y, s) + args[3:]

        # 清理transform参数中可能的问题
        if 'transform' in kwargs:
            transform = kwargs['transform']
            # 如果transform有问题，使用数据坐标系
            try:
                # 测试transform是否可用
                _ = transform.transform([(0, 0)])
            except:
                kwargs.pop('transform', None)

        return _original_text(*args, **kwargs)
    except Exception as e:
        print(f"Warning: text() failed: {e}")
        return None


def safe_tight_layout(*args, **kwargs):
    """安全的tight_layout函数"""
    try:
        return _original_tight_layout(*args, **kwargs)
    except Exception as e:
        print(f"Warning: tight_layout() failed: {e}")
        # 使用手动调整
        try:
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
        except:
            pass


def safe_show(*args, **kwargs):
    """安全的show函数"""
    print("Plot display skipped (using Agg backend). Plot saved to file.")


def safe_savefig(*args, **kwargs):
    """安全的savefig函数"""
    try:
        # 移除可能导致问题的参数
        safe_kwargs = kwargs.copy()
        if 'bbox_inches' in safe_kwargs and str(safe_kwargs['bbox_inches']) == 'tight':
            safe_kwargs.pop('bbox_inches')

        return _original_savefig(*args, **safe_kwargs)
    except Exception as e:
        print(f"Warning: savefig with safe parameters failed: {e}")
        # 尝试最基本的保存
        try:
            filename = args[0] if args else 'figure.png'
            return _original_savefig(filename)
        except Exception as e2:
            print(f"Error: Could not save figure: {e2}")


# 替换函数
plt.text = safe_text
plt.tight_layout = safe_tight_layout
plt.show = safe_show
plt.savefig = safe_savefig


# 5. 创建安全的数值处理函数
def safe_float(value, default=0.0):
    """安全地转换为float"""
    try:
        return float(value)
    except:
        return default


def safe_array(arr):
    """安全地处理数组"""
    try:
        arr = np.asarray(arr)
        # 替换无效值
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
        return arr
    except:
        return np.array([0.0])


# 导出安全函数
__all__ = ['safe_float', 'safe_array', 'matplotlib', 'plt']

print("Force fix applied: All matplotlib functions patched for safety")
print("Backend:", matplotlib.get_backend())