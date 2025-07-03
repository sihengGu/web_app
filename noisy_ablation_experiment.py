# noisy_ablation_experiment.py (修改后)

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
import numpy as np

# Suppress warnings and fix matplotlib
warnings.filterwarnings("ignore")

# Import necessary modules
from dataset import MultiResistivityDataset
from utils import set_seeds, safe_dataset_split, Timer, format_time, save_scalers # 【修改】导入save_scalers
from trainer import Trainer
from evaluator import evaluate
from noisy_ablation_models import create_noisy_ablation_model, NoiseConfig
from ablation_models import get_ablation_configs # 【修改】导入基础配置
from ultra_config import Config, DataConfig, TrainingConfig, ModelConfig # 【修改】导入Config类


# noisy_ablation_experiment.py (请用以下内容完整替换)

class NoisyAblationExperiment:
    """Noisy Ablation Experiment Manager"""

    # 【修正】确保 __init__ 方法完整，并定义 self.experiment_config
    def __init__(self, data_path, experiment_name="noisy_ablation_experiment", seed=42):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.seed = seed
        self.results_dir = f"noisy_ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'logs'), exist_ok=True)

        # 【关键】定义实验配置字典作为类的属性
        self.experiment_config = {
            'data_config': {
                'seq_length': 128,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'batch_size': 16,
                'interpolation': 'linear'
            },
            'training_config': {
                'epochs': 120,
                'learning_rate': 8e-4,
                'weight_decay': 1e-4,
                'patience': 20,
                'loss_type': 'uncertainty'
            }
        }

        # 定义噪声等级和对应的配置
        self.noise_levels_map = {
            'low_noise': NoiseConfig(input_noise_std=0.05, feature_noise_std=0.01, conv_noise_std=0.015,
                                     attention_noise_std=0.005, lstm_noise_std=0.01),
            'medium_noise': NoiseConfig(input_noise_std=0.1, feature_noise_std=0.02, conv_noise_std=0.03,
                                        attention_noise_std=0.01, lstm_noise_std=0.02),
            'high_noise': NoiseConfig(input_noise_std=0.2, feature_noise_std=0.05, conv_noise_std=0.06,
                                      attention_noise_std=0.02, lstm_noise_std=0.04)
        }

        print(f"Noisy Ablation Experiment Initialized. Results will be saved to: {self.results_dir}")

    def prepare_data(self):
        """Prepare dataset"""
        # (此方法无需修改，保持原样)
        print("Preparing dataset for noisy experiments...")
        set_seeds(self.seed)
        dataset = MultiResistivityDataset(data_folder=self.data_path,
                                          seq_length=self.experiment_config['data_config']['seq_length'])
        train_set, val_set, test_set = safe_dataset_split(dataset, self.experiment_config['data_config']['train_ratio'],
                                                          self.experiment_config['data_config']['val_ratio'],
                                                          self.experiment_config['data_config']['test_ratio'])
        self.train_loader = DataLoader(train_set, batch_size=self.experiment_config['data_config']['batch_size'],
                                       shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.experiment_config['data_config']['batch_size'],
                                     shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=self.experiment_config['data_config']['batch_size'],
                                      shuffle=False)
        self.dataset = dataset
        print(f"Dataset prepared - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        # 【替换此方法】

    def run_single_noisy_experiment(self, model_name_to_create, noise_level, base_config):
        """Run single noisy model experiment with unique plot directories."""
        experiment_name = f"{model_name_to_create}_{noise_level}"
        print(f"\n{'=' * 70}\nStarting noisy experiment: {experiment_name}\n{'=' * 70}")

        # 为每个实验创建独立的模型和配置保存目录
        model_output_dir = os.path.join(self.results_dir, "models", experiment_name)
        os.makedirs(model_output_dir, exist_ok=True)

        set_seeds(self.seed)
        noise_config = self.noise_levels_map[noise_level]

        # 创建并保存Config
        model_config = ModelConfig(model_type=experiment_name, **base_config)
        full_config = Config(data=DataConfig(**self.experiment_config['data_config']),
                             training=TrainingConfig(**self.experiment_config['training_config']), model=model_config)
        if hasattr(noise_config, '__dict__'):
            full_config.model.noise_config = noise_config.__dict__
        config_save_path = os.path.join(model_output_dir, 'config.json')
        full_config.save_to_file(config_save_path)

        # 创建模型
        model = create_noisy_ablation_model(model_name_to_create, noise_config=noise_config, **base_config)

        # 创建训练器并训练
        trainer = Trainer(model, self.train_loader, self.val_loader, self.device)
        trainer.setup_training(learning_rate=self.experiment_config['training_config']['learning_rate'],
                               weight_decay=self.experiment_config['training_config']['weight_decay'],
                               loss_type=self.experiment_config['training_config']['loss_type'])

        start_time = time.time()
        model_save_path = os.path.join(model_output_dir, 'best_model.pth')
        history = trainer.train(epochs=self.experiment_config['training_config']['epochs'],
                                patience=self.experiment_config['training_config']['patience'],
                                save_path=model_save_path)

        # 保存scalers
        scaler_save_path = model_save_path.replace('.pth', '_scalers.pkl')
        save_scalers(self.dataset, scaler_save_path)

        # --- 以下是本次修改的核心 ---
        print(f"Evaluating noisy model {experiment_name}...")

        # 【新增】为当前实验创建一个专属的图表保存文件夹
        plot_save_dir = os.path.join(self.results_dir, 'plots', experiment_name)
        os.makedirs(plot_save_dir, exist_ok=True)

        # 【修改】在调用evaluate函数时，传入专属的图表保存路径
        metrics = evaluate(
            model=trainer.model,
            test_loader=self.test_loader,
            dataset=self.dataset,
            device=self.device,
            print_results=True,
            save_dir=plot_save_dir  # <-- 将路径传递给评估函数
        )

        # 返回结果用于汇总分析
        return {'model_name': experiment_name, 'metrics': metrics}

    def run_all_noisy_experiments(self):
        """Run all noisy ablation experiments with corrected model names."""
        # (使用我们上一轮修正过的方法)
        print(f"Starting complete noisy ablation study...")
        self.prepare_data()
        base_model_configs = get_ablation_configs()

        models_to_run = {
            'baseline': 'baseline',
            'simple_lstm': 'ultra_simple_lstm',
            'basic_resnet': 'ultra_basic_resnet',
            'ultra_original': 'ultra_original'
        }

        noise_levels = self.noise_levels_map.keys()
        all_results = []

        for target_name, config_key in models_to_run.items():
            for noise_level in noise_levels:
                try:
                    noisy_model_name_to_create = f"noisy_{target_name}"
                    base_config = base_model_configs[config_key]
                    result = self.run_single_noisy_experiment(noisy_model_name_to_create, noise_level, base_config)
                    all_results.append(result)
                except Exception as e:
                    print(f"Noisy experiment {target_name}_{noise_level} failed: {str(e)}")
                    import traceback
                    traceback.print_exc()

        # 汇总分析
        if all_results:
            df = pd.DataFrame([res for res in all_results if res['metrics']])
            if not df.empty:
                df = pd.concat([df.drop(['metrics'], axis=1), df['metrics'].apply(pd.Series)], axis=1)
                summary_path = os.path.join(self.results_dir, "noisy_ablation_summary.csv")
                df.to_csv(summary_path, index=False)
                print(f"\nNoisy experiments summary saved to {summary_path}")
        return all_results


def main():
    """Main function - run noisy ablation study"""
    # (此部分无需修改)
    import argparse
    parser = argparse.ArgumentParser(description="Noisy ablation study script")
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
    args = parser.parse_args()
    experiment = NoisyAblationExperiment(data_path=args.data_path)
    experiment.run_all_noisy_experiments()


if __name__ == "__main__":
    main()