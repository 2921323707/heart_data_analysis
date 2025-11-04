"""
工具函数模块
提供通用的工具函数，包括数据保存、加载、评估指标计算等
"""
import os
import json
import pickle
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def save_model(model: torch.nn.Module, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None,
               epoch: Optional[int] = None, loss: Optional[float] = None, 
               metrics: Optional[Dict[str, float]] = None):
    """
    保存模型和相关信息
    
    Args:
        model: PyTorch模型
        filepath: 保存路径
        optimizer: 优化器（可选）
        epoch: 训练轮数（可选）
        loss: 损失值（可选）
        metrics: 评估指标字典（可选）
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        save_dict['epoch'] = epoch
    if loss is not None:
        save_dict['loss'] = loss
    if metrics is not None:
        save_dict['metrics'] = metrics
    
    torch.save(save_dict, filepath)
    print(f"模型已保存到: {filepath}")


def load_model(filepath: str, model: Optional[torch.nn.Module] = None, 
              optimizer: Optional[torch.optim.Optimizer] = None,
              device: str = 'cpu'):
    """
    加载模型和相关信息
    
    Args:
        filepath: 模型文件路径
        model: PyTorch模型（如果提供，将加载权重）
        optimizer: 优化器（可选，如果提供将加载状态）
        device: 设备类型
        
    Returns:
        checkpoint字典，包含模型状态、epoch等信息
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型权重已加载")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"优化器状态已加载")
    
    return checkpoint


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     average: str = 'weighted') -> Dict[str, float]:
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        average: 平均方式 ('micro', 'macro', 'weighted')
        
    Returns:
        包含各种评估指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # 计算每个类别的指标
    if len(np.unique(y_true)) <= 10:  # 只对类别数较少的情况计算
        metrics['per_class_precision'] = precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['per_class_recall'] = recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['per_class_f1'] = f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 8),
                         normalize: bool = False):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
        figsize: 图像大小
        normalize: 是否归一化显示
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names if class_names else None,
                yticklabels=class_names if class_names else None)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵' + ('（归一化）' if normalize else ''))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    else:
        plt.show()


def save_results(results: Dict[str, Any], filepath: str, format: str = 'json'):
    """
    保存结果到文件
    
    Args:
        results: 结果字典
        filepath: 保存路径
        format: 保存格式 ('json' 或 'pkl')
    """
    if format == 'json':
        # 将numpy数组转换为列表
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
    elif format == 'pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    print(f"结果已保存到: {filepath}")


def load_results(filepath: str, format: str = 'json'):
    """
    从文件加载结果
    
    Args:
        filepath: 文件路径
        format: 文件格式 ('json' 或 'pkl')
        
    Returns:
        结果字典
    """
    if format == 'json':
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
    elif format == 'pkl':
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    return results


def create_experiment_dir(base_dir: str = 'experiments', 
                         experiment_name: Optional[str] = None) -> str:
    """
    创建实验目录
    
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称，如果为None则使用时间戳
        
    Returns:
        创建的目录路径
    """
    from datetime import datetime
    
    if experiment_name is None:
        experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 创建子目录
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'figures'), exist_ok=True)
    
    print(f"实验目录已创建: {exp_dir}")
    return exp_dir


def set_seed(seed: int = 42):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        包含总参数数和可训练参数数的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(model: torch.nn.Module, input_shape: Tuple[int, ...] = (1, 1, 360)):
    """
    打印模型摘要
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状 (batch_size, channels, length)
    """
    print("=" * 60)
    print("模型摘要")
    print("=" * 60)
    
    # 参数统计
    params = count_parameters(model)
    print(f"\n参数统计:")
    print(f"  总参数数: {params['total']:,}")
    print(f"  可训练参数: {params['trainable']:,}")
    print(f"  不可训练参数: {params['non_trainable']:,}")
    
    # 模型结构
    print(f"\n模型结构:")
    print(model)
    
    # 测试前向传播
    try:
        model.eval()
        with torch.no_grad():
            x = torch.randn(input_shape)
            if torch.cuda.is_available():
                x = x.cuda()
                model = model.cuda()
            output = model(x)
            print(f"\n输入形状: {input_shape}")
            print(f"输出形状: {output.shape}")
    except Exception as e:
        print(f"\n前向传播测试失败: {e}")
    
    print("=" * 60)


def get_device_info() -> Dict[str, Any]:
    """
    获取设备信息
    
    Returns:
        包含设备信息的字典
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_properties'] = {
            'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'major': torch.cuda.get_device_properties(0).major,
            'minor': torch.cuda.get_device_properties(0).minor,
        }
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
    else:
        info['device'] = 'cpu'
    
    return info


def format_time(seconds: float) -> str:
    """
    格式化时间（秒转换为可读格式）
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}小时{int(minutes)}分钟"


if __name__ == "__main__":
    # 测试工具函数
    print("工具函数模块测试")
    
    # 测试设备信息
    device_info = get_device_info()
    print("\n设备信息:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # 测试随机种子
    set_seed(42)
    print("\n随机种子测试完成")

