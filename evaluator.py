# evaluator.py (修改后 - 分解Dashboard)

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

# --- 绘图风格设置 (保持不变) ---
plt.style.use(['seaborn-v0_8-paper'])
sns.set_palette("deep")

FONT_SIZES = {
    'title': 14, 'label': 12, 'tick': 10, 'legend': 10, 'text': 10
}

COLORS = {
    'primary': '#2E86AB', 'secondary': '#A23B72', 'accent': '#F18F01',
    'success': '#C73E1D', 'neutral': '#CCE1F0', 'dark': '#333333', 'light': '#F5F5F5'
}

def setup_plot_params():
    """设置matplotlib参数"""
    params = {
        'figure.figsize': (10, 6), 'figure.dpi': 300, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.facecolor': 'white', 'axes.linewidth': 1.2,
        'axes.spines.left': True, 'axes.spines.bottom': True,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.linewidth': 0.5, 'grid.alpha': 0.3, 'grid.color': '#cccccc',
        'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 6,
        'ytick.major.size': 6, 'font.size': FONT_SIZES['text'],
        'axes.titlesize': FONT_SIZES['title'], 'axes.labelsize': FONT_SIZES['label'],
        'xtick.labelsize': FONT_SIZES['tick'], 'ytick.labelsize': FONT_SIZES['tick'],
        'legend.fontsize': FONT_SIZES['legend'], 'figure.titlesize': FONT_SIZES['title'] + 2,
    }
    plt.rcParams.update(params)

class Evaluator:
    """模型评估器"""
    def __init__(self, model, device=None, save_dir='./plots'):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        setup_plot_params()

    def compute_metrics(self, y_true, y_pred):
        """计算评估指标"""
        # (此函数无需修改，保持原样)
        if torch.is_tensor(y_true): y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred): y_pred = y_pred.cpu().numpy()
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean, y_pred_clean = y_true[mask], y_pred[mask]
        if len(y_true_clean) == 0: return {}
        return {
            'r2': r2_score(y_true_clean, y_pred_clean),
            'mae': mean_absolute_error(y_true_clean, y_pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            'mape': np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100,
            'pearson_r': stats.pearsonr(y_true_clean.flatten(), y_pred_clean.flatten())[0]
        }

    def evaluate_model(self, test_loader, dataset, print_results=True):
        """全面评估模型"""
        self.model.eval()
        all_predictions, all_targets, all_uncertainties = [], [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if hasattr(self.model, 'estimate_uncertainty') and self.model.estimate_uncertainty:
                    mean, var = self.model(inputs)
                    all_predictions.append(mean.cpu()); all_uncertainties.append(torch.sqrt(var).cpu())
                else:
                    outputs = self.model(inputs)
                    all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        inverse_transform = dataset.get_inverse_transforms()
        predictions_original = inverse_transform(predictions)
        targets_original = inverse_transform(targets)
        metrics = self.compute_metrics(targets_original, predictions_original)

        if print_results: self._print_metrics(metrics)
        self.create_comparison_table(targets_original, predictions_original, all_uncertainties, inverse_transform)

        # 【修改】调用独立的绘图函数
        self.create_evaluation_plots(
            targets_original, predictions_original,
            all_uncertainties if all_uncertainties else None, metrics
        )
        return metrics, predictions_original, targets_original

    def create_comparison_table(self, true_values, predicted_values, uncertainties=None, inverse_transform=None):
        """创建实际值与预测值对照表"""
        # (此函数无需修改，保持原样)
        true_flat, pred_flat = true_values.flatten(), predicted_values.flatten()
        comparison_data = {'True_Resistivity': true_flat, 'Predicted_Resistivity': pred_flat}
        df = pd.DataFrame(comparison_data)
        detailed_table_path = os.path.join(self.save_dir, 'prediction_comparison_detailed.csv')
        df.to_csv(detailed_table_path, index=False, float_format='%.3f')
        print(f"Detailed comparison table saved to: {detailed_table_path}")
        return df

    def _print_metrics(self, metrics):
        """打印评估指标"""
        # (此函数无需修改，保持原样)
        print("\n" + "="*60 + "\nModel Evaluation Results\n" + "="*60)
        print(f"  R² Score:              {metrics['r2']:.4f}")
        print(f"  Mean Absolute Error:   {metrics['mae']:.2f} Ω·m")
        print(f"  Root Mean Square Error:{metrics['rmse']:.2f} Ω·m")
        print(f"  Mean Abs Percent Error:{metrics['mape']:.2f}%")
        print(f"  Pearson Correlation:   {metrics['pearson_r']:.4f}")
        print("="*60)

    # --- 【修改】主绘图流程 ---
    def create_evaluation_plots(self, y_true, y_pred, uncertainties=None, metrics=None):
        """创建一系列独立的评估图表"""
        # 1. 预测 vs 实际值散点图 (复用现有函数)
        self.plot_prediction_scatter(y_true, y_pred, metrics)

        # 2. 残差分析图 (复用现有函数)
        self.plot_residual_analysis(y_true, y_pred)

        # 3. 【新增】调用独立的“预测误差 vs 预测值”图
        self.plot_prediction_errors(y_true, y_pred)

        # 4. 【新增】调用独立的“误差分布直方图”
        self.plot_error_distribution_hist(y_true, y_pred)

        # 5. 不确定性图 (如果支持)
        if uncertainties is not None:
            uncertainties_np = torch.cat(uncertainties, dim=0).numpy()
            self.plot_uncertainty_analysis(y_true, y_pred, uncertainties_np)

    def plot_prediction_scatter(self, y_true, y_pred, metrics=None):
        """绘制预测 vs 实际值散点图 (保持不变)"""
        # (此函数无需修改，保持原样)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, color=COLORS['primary'], edgecolors='white', linewidth=0.5)
        min_val, max_val = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8, label='Perfect prediction')
        textstr = f'R² = {metrics["r2"]:.4f}\nMAE = {metrics["mae"]:.2f} Ω·m\nRMSE = {metrics["rmse"]:.2f} Ω·m'
        props = dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=FONT_SIZES['text'], verticalalignment='top', bbox=props)
        ax.set_xlabel('True Resistivity (Ω·m)'); ax.set_ylabel('Predicted Resistivity (Ω·m)'); ax.set_title('Model Prediction Performance'); ax.legend(loc='lower right')
        ax.set_aspect('equal', adjustable='box'); plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'prediction_scatter.png')); plt.close()
        print(f"Prediction scatter plot saved to: {self.save_dir}/prediction_scatter.png")

    def plot_residual_analysis(self, y_true, y_pred):
        """绘制残差分析图 (保持不变)"""
        # (此函数无需修改，保持原样)
        residuals = y_true - y_pred
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Residual Analysis'); ax1.scatter(y_pred, residuals, alpha=0.6, s=30, color=COLORS['primary']); ax1.axhline(y=0, color='r', linestyle='--'); ax1.set_xlabel('Predicted Values (Ω·m)'); ax1.set_ylabel('Residuals (Ω·m)'); ax1.set_title('Residuals vs Predicted')
        ax2.scatter(y_true, residuals, alpha=0.6, s=30, color=COLORS['secondary']); ax2.axhline(y=0, color='r', linestyle='--'); ax2.set_xlabel('True Values (Ω·m)'); ax2.set_ylabel('Residuals (Ω·m)'); ax2.set_title('Residuals vs True Values')
        stats.probplot(residuals.flatten(), dist="norm", plot=ax3); ax3.set_title('Q-Q Plot of Residuals')
        ax4.hist(residuals.flatten(), bins=30, density=True, alpha=0.7, color=COLORS['accent'], edgecolor='black'); ax4.set_xlabel('Residuals (Ω·m)'); ax4.set_ylabel('Density'); ax4.set_title('Distribution of Residuals')
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, 'residual_analysis.png')); plt.close()
        print(f"Residual analysis plot saved to: {self.save_dir}/residual_analysis.png")

    def plot_uncertainty_analysis(self, y_true, y_pred, uncertainties):
        """绘制不确定性分析图 (保持不变)"""
        # (此函数无需修改，保持原ayan样)
        errors = np.abs(y_true - y_pred)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Uncertainty Analysis')
        ax1.scatter(uncertainties.flatten(), errors.flatten(), alpha=0.6, s=30, color=COLORS['primary']); ax1.set_xlabel('Predicted Uncertainty (Ω·m)'); ax1.set_ylabel('Absolute Error (Ω·m)'); ax1.set_title('Uncertainty vs Absolute Error')
        ax2.scatter(y_pred.flatten(), uncertainties.flatten(), alpha=0.6, s=30, color=COLORS['secondary']); ax2.set_xlabel('Predicted Values (Ω·m)'); ax2.set_ylabel('Predicted Uncertainty (Ω·m)'); ax2.set_title('Predictions vs Uncertainty')
        ax3.hist(uncertainties.flatten(), bins=30, density=True, alpha=0.7, color=COLORS['accent'], edgecolor='black'); ax3.set_xlabel('Predicted Uncertainty (Ω·m)'); ax3.set_ylabel('Density'); ax3.set_title('Uncertainty Distribution')
        confidence_levels, coverage_rates = np.arange(0.1, 3.1, 0.1), []
        for conf in confidence_levels: coverage_rates.append(np.mean(errors.flatten() <= (conf * uncertainties.flatten())))
        ax4.plot(confidence_levels, coverage_rates, 'b-', linewidth=2, label='Empirical coverage'); ax4.plot(confidence_levels, stats.norm.cdf(confidence_levels) - stats.norm.cdf(-confidence_levels), 'r--', linewidth=2, label='Theoretical (Gaussian)'); ax4.set_xlabel('Confidence Level (σ)'); ax4.set_ylabel('Coverage Rate'); ax4.set_title('Coverage Rate vs Confidence Level'); ax4.legend(); ax4.set_ylim(0, 1)
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, 'uncertainty_analysis.png')); plt.close()
        print(f"Uncertainty analysis plot saved to: {self.save_dir}/uncertainty_analysis.png")

    # --- 【新增】独立的“预测误差”图 ---
    def plot_prediction_errors(self, y_true, y_pred, save_name='prediction_errors.png'):
        """绘制独立的“预测误差 vs. 预测值”图"""
        print(f"Generating prediction errors plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        errors = y_pred - y_true

        ax.scatter(y_pred, errors, alpha=0.7, s=30, color=COLORS['accent'], edgecolors='white', linewidth=0.5)
        ax.axhline(y=0, color=COLORS['success'], linestyle='--', lw=2)

        ax.set_xlabel('Predicted Values (Ω·m)', fontsize=FONT_SIZES['label'])
        ax.set_ylabel('Prediction Errors (Ω·m)', fontsize=FONT_SIZES['label'])
        ax.set_title('Prediction Errors vs. Predicted Values', fontsize=FONT_SIZES['title'])

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close(fig)
        print(f"Prediction errors plot saved to: {os.path.join(self.save_dir, save_name)}")

    # --- 【新增】独立的“误差分布”直方图 ---
    def plot_error_distribution_hist(self, y_true, y_pred, save_name='error_distribution_hist.png'):
        """绘制独立的“误差分布”直方图"""
        print(f"Generating error distribution histogram...")
        fig, ax = plt.subplots(figsize=(10, 6))
        errors = y_pred - y_true

        sns.histplot(errors, kde=True, ax=ax, color=COLORS['secondary'], bins=50)

        ax.set_xlabel('Prediction Errors (Ω·m)', fontsize=FONT_SIZES['label'])
        ax.set_ylabel('Density', fontsize=FONT_SIZES['label'])
        ax.set_title('Distribution of Prediction Errors', fontsize=FONT_SIZES['title'])

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close(fig)
        print(f"Error distribution histogram saved to: {os.path.join(self.save_dir, save_name)}")


# --- plot_training_history 和 evaluate 函数 (保持不变) ---
def plot_training_history(history, save_dir='./plots'):
    # (此函数无需修改，保持原样)
    setup_plot_params(); os.makedirs(save_dir, exist_ok=True)
    has_uncertainty = history.get('train_uncertainty') is not None
    if has_uncertainty:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training History')
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss', color=COLORS['primary'])
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', color=COLORS['secondary'])
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training and Validation Loss'); ax1.legend()
    if 'learning_rates' in history:
        ax2.plot(epochs, history['learning_rates'], 'g-', linewidth=2, color=COLORS['accent'])
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Learning Rate'); ax2.set_title('Learning Rate Schedule'); ax2.set_yscale('log')
    if has_uncertainty:
        ax3.plot(epochs, history['train_uncertainty'], 'b-', linewidth=2, label='Training Uncertainty', color=COLORS['primary'])
        ax3.plot(epochs, history['val_uncertainty'], 'r-', linewidth=2, label='Validation Uncertainty', color=COLORS['secondary'])
        ax3.set_xlabel('Epoch'); ax3.set_ylabel('Uncertainty'); ax3.set_title('Training and Validation Uncertainty'); ax3.legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'training_history.png')); plt.close()
    print(f"Training history plot saved to: {save_dir}/training_history.png")


def evaluate(model, test_loader, dataset, device=None, print_results=True, save_dir='./plots'):
    """主评估函数"""
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(model, device, save_dir)
    metrics, _, _ = evaluator.evaluate_model(test_loader, dataset, print_results)
    return metrics