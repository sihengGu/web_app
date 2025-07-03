# ablation_experiment.py (修改后)

import os
import json
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Suppress warnings and fix matplotlib
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Import necessary modules
from dataset import MultiResistivityDataset
from utils import set_seeds, safe_dataset_split, Timer, format_time, save_scalers # 【修改】导入save_scalers
from trainer import Trainer
from evaluator import evaluate
from ablation_models import create_ablation_model, get_ablation_configs, analyze_ablation_model_complexity
from ultra_config import Config, DataConfig, TrainingConfig, ModelConfig # 【修改】导入Config类

class AblationExperiment:
    """Ablation Experiment Manager"""

    def __init__(self, data_path, experiment_name="ablation_experiment", seed=42):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.seed = seed
        self.results_dir = f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
        # 【修改】为每个模型创建独立的文件夹

        # Experiment configuration
        self.data_config = DataConfig(
            seq_length=128, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            batch_size=16, interpolation='linear'
        )
        self.training_config = TrainingConfig(
            epochs=100, learning_rate=1e-3, weight_decay=1e-4,
            patience=15, loss_type='uncertainty'
        )

        print(f"Ablation experiment initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Device: {self.device}")

    def prepare_data(self):
        """Prepare dataset"""
        print("Preparing dataset...")
        set_seeds(self.seed)
        dataset = MultiResistivityDataset(
            data_folder=self.data_path,
            seq_length=self.data_config.seq_length,
            interpolation=self.data_config.interpolation,
            augmentation=False
        )
        train_set, val_set, test_set = safe_dataset_split(
            dataset, self.data_config.train_ratio,
            self.data_config.val_ratio, self.data_config.test_ratio
        )
        self.train_loader = DataLoader(train_set, batch_size=self.data_config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.data_config.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=self.data_config.batch_size, shuffle=False)
        self.dataset = dataset
        print(f"Dataset prepared - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    def run_single_experiment(self, model_name, model_config_dict):
        """Run single model experiment"""
        print(f"\n{'='*60}")
        print(f"Starting experiment: {model_name}")
        print(f"{'='*60}")

        # 【修改】为每个模型创建独立的输出目录
        model_output_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        set_seeds(self.seed)

        # 【修改】创建并保存完整的Config对象
        model_config = ModelConfig(model_type=model_name, **model_config_dict)
        full_config = Config(data=self.data_config, training=self.training_config, model=model_config)
        config_save_path = os.path.join(model_output_dir, 'config.json')
        full_config.save_to_file(config_save_path)
        print(f"Configuration for '{model_name}' saved to {config_save_path}")

        model = create_ablation_model(model_name, **model_config_dict)
        complexity = analyze_ablation_model_complexity(model)
        print(f"Model complexity: {complexity['total_params']:,} parameters, {complexity['model_size_mb']:.2f} MB")

        trainer = Trainer(model, self.train_loader, self.val_loader, self.device)
        trainer.setup_training(
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            loss_type=self.training_config.loss_type
        )

        start_time = time.time()
        model_save_path = os.path.join(model_output_dir, 'best_model.pth')
        history = trainer.train(
            epochs=self.training_config.epochs,
            patience=self.training_config.patience,
            save_path=model_save_path
        )
        training_time = time.time() - start_time

        # 【修改】保存scalers
        scaler_save_path = model_save_path.replace('.pth', '_scalers.pkl')
        save_scalers(self.dataset, scaler_save_path)

        print(f"Evaluating model {model_name}...")
        eval_save_dir = os.path.join(self.results_dir, 'plots', model_name)
        metrics = evaluate(
            model=trainer.model, test_loader=self.test_loader, dataset=self.dataset,
            device=self.device, print_results=True, save_dir=eval_save_dir
        )

        result = {
            'model_name': model_name, 'config': model_config_dict, 'complexity': complexity,
            'metrics': metrics, 'training_time': training_time,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else float('inf'),
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
        }
        return result

    def run_all_experiments(self):
        """Run all ablation experiments"""
        self.prepare_data()
        all_configs = get_ablation_configs()
        experiment_order = [
            'baseline', 'ultra_simple_lstm', 'ultra_basic_resnet', 'ultra_no_transformer',
            'ultra_no_attention', 'ultra_no_fpn', 'ultra_no_wavelet', 'ultra_original'
        ]
        all_results = []
        for model_name in experiment_order:
            if model_name in all_configs:
                result = self.run_single_experiment(model_name, all_configs[model_name])
                all_results.append(result)
        self.analyze_results(all_results)
        return all_results

    def analyze_results(self, results):
        """Analyze experimental results"""
        # (此部分无需修改，仅用于生成汇总报告)
        print(f"\n{'='*60}\nAnalyzing Experimental Results\n{'='*60}")
        df_data = []
        for result in results:
            if result['metrics']:
                row = {
                    'Model': result['model_name'], 'Parameters': result['complexity']['total_params'],
                    'Size_MB': result['complexity']['model_size_mb'],
                    'Training_Time_min': result['training_time'] / 60,
                    'R²': result['metrics']['r2'], 'MAE': result['metrics']['mae'],
                    'RMSE': result['metrics']['rmse'],
                    'Final_Val_Loss': result['final_val_loss']
                }
                df_data.append(row)
        df = pd.DataFrame(df_data)
        results_csv = os.path.join(self.results_dir, 'ablation_summary_results.csv')
        df.to_csv(results_csv, index=False)
        print("\nExperimental Results Summary:")
        print(df.round(4).to_string(index=False))

# (main函数部分无需修改)
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation study script")
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
    args = parser.parse_args()
    experiment = AblationExperiment(data_path=args.data_path)
    experiment.run_all_experiments()

if __name__ == "__main__":
    main()