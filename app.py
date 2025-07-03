import streamlit as st
import os
import torch
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

# 导入您项目中的必要模块
from ultra_config import Config
from model import SimpleMultiScaleModel, AdvancedResistivityModel
from ultra_model import UltraAdvancedResistivityModel, EnsembleUltraModel, get_ultra_model_configs
from ablation_models import create_ablation_model, get_ablation_configs
from noisy_ablation_models import create_noisy_ablation_model, NoiseConfig
from improved_simple_model import ImprovedSimpleModel, EnhancedSimpleModel
from optimized_model import OptimizedResistivityModel, EnsembleOptimizedModel

# --- 页面配置 ---
st.set_page_config(
    page_title="Resistivity Inversion Model Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 样式与绘图参数 ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# --- 模型创建与加载函数 (保持不变) ---
@st.cache_resource
def load_model_and_assets(config_file, model_file, scalers_file) -> Dict[str, Any]:
    """Loads model, config, and scalers from uploaded file objects."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not all([config_file, model_file, scalers_file]):
        st.error("Please upload all three required model files.")
        return None
    config_data = json.load(config_file)
    config = Config.from_dict(config_data)
    model = create_model_from_config(config)
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    scalers = pickle.load(scalers_file)
    return {"model": model, "config": config, "scalers": scalers, "device": device}


# app.py (请替换此函数)

# --- 模型创建工厂 (最终修正版 V2) ---
def create_model_from_config(config: Config):
    """
    Unified model creation factory (Final, Robust Version).
    This function correctly handles base, ablation, and all noisy models.
    """
    model_type = config.model.model_type

    # 1. 首先处理带噪声的模型
    if 'noisy_' in model_type:
        from noisy_ablation_models import NoiseConfig, create_noisy_ablation_model

        # 步骤 1: 从 model_type 中解析出工厂函数期望的名称
        noise_suffixes = ['_low_noise', '_medium_noise', '_high_noise']
        factory_name = model_type
        for suffix in noise_suffixes:
            if factory_name.endswith(suffix):
                factory_name = factory_name[:-len(suffix)]
                break

        # 步骤 2: 从已加载的 config 对象中提取噪声配置
        noise_config_obj = NoiseConfig(**config.model.noise_config) if hasattr(config.model, 'noise_config') and config.model.noise_config else NoiseConfig()

        # 步骤 3: 直接从已加载的 config 对象中获取基础模型参数
        base_model_params = config.model.__dict__

        # 【修正】在解包前，从字典中安全地移除 'noise_config'，以避免重复传递
        base_model_params.pop('noise_config', None)

        # 步骤 4: 使用正确的工厂名称和已加载的参数调用创建函数
        return create_noisy_ablation_model(
            model_name=factory_name,
            noise_config=noise_config_obj,
            **base_model_params
        )

    # 2. 如果不是带噪模型，再处理消融实验模型
    from ablation_models import get_ablation_configs, create_ablation_model
    ablation_configs = get_ablation_configs()
    if model_type in ablation_configs:
        return create_ablation_model(model_type, **ablation_configs[model_type])

    # 3. 最后处理其他直接匹配的基础模型
    # (此部分逻辑保持不变)
    if model_type == 'SimpleMultiScaleModel':
        return SimpleMultiScaleModel(input_dim=config.model.input_dim, hidden_dim=config.model.hidden_dim, dropout=config.model.dropout, estimate_uncertainty=config.model.estimate_uncertainty)
    if model_type == 'AdvancedResistivityModel':
        return AdvancedResistivityModel(input_dim=config.model.input_dim, hidden_dim=config.model.hidden_dim, num_blocks=config.model.num_blocks, dropout=config.model.dropout)
    if model_type == 'UltraAdvancedResistivityModel':
        return UltraAdvancedResistivityModel(input_dim=config.model.input_dim, hidden_dim=config.model.hidden_dim, n_heads=config.model.n_heads, n_transformer_layers=config.model.n_transformer_layers, dropout=config.model.dropout, estimate_uncertainty=config.model.estimate_uncertainty, use_wavelet=config.model.use_wavelet, use_fpn=config.model.use_fpn)
    if model_type == 'ImprovedSimpleModel':
        return ImprovedSimpleModel(input_dim=config.model.input_dim, hidden_dim=config.model.hidden_dim, dropout=config.model.dropout, estimate_uncertainty=config.model.estimate_uncertainty)
    if model_type == 'EnhancedSimpleModel':
        return EnhancedSimpleModel(input_dim=config.model.input_dim, hidden_dim=config.model.hidden_dim, dropout=config.model.dropout, estimate_uncertainty=config.model.estimate_uncertainty)
    if model_type == 'OptimizedResistivityModel':
        return OptimizedResistivityModel(input_dim=config.model.input_dim, hidden_dim=config.model.hidden_dim, num_conv_blocks=config.model.num_conv_blocks, num_attention_blocks=config.model.num_attention_blocks, n_heads=config.model.n_heads, dropout=config.model.dropout, estimate_uncertainty=config.model.estimate_uncertainty)

    # 如果所有尝试都失败，则抛出错误
    raise ValueError(f"Model type '{model_type}' could not be resolved by the factory.")


# --- 数据处理和反变换函数 (保持不变) ---
def preprocess_data(df: pd.DataFrame, assets: Dict[str, Any]) -> torch.Tensor:
    # (此函数无需修改)
    config = assets['config']
    scalers = assets['scalers']
    required_cols = ['depth', 'Ex_real', 'Ex_imag', 'Ey_real', 'Ey_imag', 'Ez_real', 'Ez_imag']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Uploaded CSV is missing required columns. Please ensure it contains: {', '.join(required_cols)}")
        return None
    df = df.sort_values('depth').reset_index(drop=True)
    feature_cols = ['Ex_real', 'Ex_imag', 'Ey_real', 'Ey_imag', 'Ez_real', 'Ez_imag', 'depth']
    features = df[feature_cols].values
    seq_length = config.data.seq_length
    original_length = features.shape[0]
    x_orig = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, seq_length)
    resampled = np.zeros((seq_length, features.shape[1]))
    for col in range(features.shape[1]):
        resampled[:, col] = np.interp(x_new, x_orig, features[:, col])
    feat_scalers = scalers['feat_scalers']
    for i in range(len(feat_scalers)):
        resampled[:, i] = feat_scalers[i].transform(resampled[:, i].reshape(-1, 1)).flatten()
    return torch.FloatTensor(resampled).unsqueeze(0).to(assets['device'])


def inverse_transform_prediction(pred: np.ndarray, scalers: Dict[str, Any]) -> np.ndarray:
    # (此函数无需修改)
    resistivity_scaler = scalers['resistivity_scaler']
    y_std = resistivity_scaler.inverse_transform(pred)
    if scalers.get('log_transform', False):
        y_exp = np.exp(y_std)
        offset = scalers.get('offset', 0)
        if offset > 0:
            y_exp = y_exp - offset
        return y_exp
    return y_std


# --- 可视化函数 (保持不变) ---
def plot_input_signals(df: pd.DataFrame, normalize: bool = False):
    """
    Plots input signals with subplots for clarity and an option to normalize.
    """
    st.subheader("Input Data Visualization")
    signal_cols = ['Ex_real', 'Ex_imag', 'Ey_real', 'Ey_imag', 'Ez_real', 'Ez_imag']
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharey=True)
    axes = axes.flatten()
    for i, col in enumerate(signal_cols):
        if col in df.columns:
            data = df[col]
            if normalize:
                min_val, max_val = data.min(), data.max()
                if max_val > min_val: data = (data - min_val) / (max_val - min_val)
                plot_title, xlabel = f"{col} (Normalized)", "Normalized Amplitude"
            else:
                plot_title, xlabel = col, "Amplitude"
            axes[i].plot(data, df['depth'])
            axes[i].set_title(plot_title);
            axes[i].set_xlabel(xlabel);
            axes[i].grid(True)
    fig.text(-0.01, 0.5, 'Depth', va='center', rotation='vertical', fontsize=12)
    fig.suptitle("Input Well Log Signals vs. Depth", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    st.pyplot(fig)


# app.py

# ... (文件顶部的所有import和辅助函数保持不变) ...

# --- 【修改】主函数，使用 st.session_state 实现交互式更新 ---
def main():
    """Main function with multi-file upload and interactive state management."""
    st.title("Deep Learning for Resistivity Inversion")
    st.markdown("An advanced tool for model analysis, supporting single and multi-file batch prediction.")

    # --- Sidebar for Controls ---
    st.sidebar.header("Controls")

    st.sidebar.subheader("1. Upload Model Files")
    config_file = st.sidebar.file_uploader("Upload Configuration File (*.json)", type="json")
    model_file = st.sidebar.file_uploader("Upload Model Weights (*.pth)", type="pth")
    scalers_file = st.sidebar.file_uploader("Upload Scalers File (*.pkl)", type="pkl")

    st.sidebar.markdown("---")

    st.sidebar.subheader("2. Upload Data for Prediction")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Well Log Data (*.csv)",
        type="csv",
        accept_multiple_files=True,
        help="You can upload multiple CSV files for batch processing."
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("3. Run")
    run_button = st.sidebar.button("Run Prediction", type="primary")

    # --- 【修改】核心逻辑分离: 计算与显示 ---

    # 1. 计算逻辑: 仅在按钮点击时运行
    if run_button:
        if not all([config_file, model_file, scalers_file, uploaded_files]):
            st.warning("Please upload all required files (config, model, scalers, and at least one data CSV).")
        else:
            MAX_FILES = 5
            if len(uploaded_files) > MAX_FILES:
                st.error(f"Too many files uploaded. Please upload a maximum of {MAX_FILES} files at a time.")
            else:
                # 重置文件指针
                config_file.seek(0);
                model_file.seek(0);
                scalers_file.seek(0)
                for f in uploaded_files: f.seek(0)

                with st.spinner("Loading model and assets..."):
                    assets = load_model_and_assets(config_file, model_file, scalers_file)

                if assets:
                    results_list = []
                    data_frames = {file.name: pd.read_csv(file) for file in uploaded_files}

                    with st.spinner(f"Processing {len(data_frames)} file(s)..."):
                        for filename, df in data_frames.items():
                            input_tensor = preprocess_data(df.copy(), assets)
                            if input_tensor is not None:
                                output = assets['model'](input_tensor)
                                if isinstance(output, tuple):
                                    mean, var = output
                                    pred_mean = inverse_transform_prediction(mean.detach().cpu().numpy(),
                                                                             assets['scalers'])
                                    uncertainty = torch.sqrt(var).detach().cpu().numpy()[0][0]
                                    resistivity = pred_mean[0][0]
                                else:
                                    pred = inverse_transform_prediction(output.detach().cpu().numpy(),
                                                                        assets['scalers'])
                                    resistivity = pred[0][0]
                                    uncertainty = None

                                results_list.append({
                                    "File Name": filename,
                                    "Predicted Resistivity (Ω·m)": resistivity,
                                    "Uncertainty (Scaled Std. Dev.)": uncertainty
                                })

                    # 【关键】将计算结果存入 session_state
                    st.session_state['prediction_run'] = True
                    st.session_state['results_df'] = pd.DataFrame(results_list)
                    st.session_state['data_frames'] = data_frames
                    st.session_state['model_name'] = assets['config'].model.model_type
                    st.rerun()  # 强制立即重新运行脚本以更新显示

    # 2. 显示逻辑: 只要 session_state 中有结果，就总是运行
    if 'prediction_run' in st.session_state and st.session_state['prediction_run']:
        st.success(f"Model '{st.session_state['model_name']}' loaded and prediction complete.")

        st.subheader("Prediction Results Summary")
        if st.session_state['results_df'].empty:
            st.error("No files could be processed successfully. Please check the format of your CSV files.")
        else:
            st.dataframe(st.session_state['results_df'].style.format({
                'Predicted Resistivity (Ω·m)': '{:.4f}',
                'Uncertainty (Scaled Std. Dev.)': '{:.4f}'
            }))

            # 可视化区
            st.subheader("Detailed Visualization")
            data_frames = st.session_state['data_frames']

            if len(data_frames) > 1:
                # 创建下拉菜单，它的值在脚本重新运行时会保持为用户的最新选择
                selected_file = st.selectbox(
                    "Select a file to visualize:",
                    options=data_frames.keys()
                )
            else:
                selected_file = list(data_frames.keys())[0]

            if selected_file:
                df_to_plot = data_frames[selected_file]
                normalize_plot = st.checkbox("Normalize signals for visualization", value=True,
                                             key=f"norm_{selected_file}")
                plot_input_signals(df_to_plot, normalize=normalize_plot)

    # 初始提示信息
    elif not run_button:
        st.info("Please upload model files and data, then click 'Run Prediction' in the sidebar.")


if __name__ == "__main__":
    # 【新增】在主程序入口初始化 session_state 检查
    if 'prediction_run' not in st.session_state:
        st.session_state['prediction_run'] = False
    main()