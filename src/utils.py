"""
Utility Functions for CGCNN

Common utility functions for CGCNN training and evaluation.

Author: LunaZhang
Date: 2023
"""

import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def save_dict_to_json(data: Dict[str, Any], filepath: str) -> None:
    """
    将字典保存为JSON文件
    Save dictionary to JSON file
    
    Parameters
    ----------
    data : Dict[str, Any]
        要保存的数据字典
    filepath : str
        保存路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_dict_from_json(filepath: str) -> Dict[str, Any]:
    """
    从JSON文件加载字典
    Load dictionary from JSON file
    
    Parameters
    ----------
    filepath : str
        JSON文件路径
        
    Returns
    -------
    Dict[str, Any]
        加载的数据字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """
    保存数据为pickle文件
    Save data as pickle file
    
    Parameters
    ----------
    data : Any
        要保存的数据
    filepath : str
        保存路径
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    从pickle文件加载数据
    Load data from pickle file
    
    Parameters
    ----------
    filepath : str
        pickle文件路径
        
    Returns
    -------
    Any
        加载的数据
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量
    Count the number of parameters in a model
    
    Parameters
    ----------
    model : nn.Module
        PyTorch模型
        
    Returns
    -------
    int
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module) -> None:
    """
    打印模型信息
    Print model information
    
    Parameters
    ----------
    model : nn.Module
        PyTorch模型
    """
    total_params = count_parameters(model)
    print(f"模型总参数数: {total_params:,}")
    print(f"模型结构:")
    print(model)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算回归任务的评估指标
    Compute regression metrics
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
        
    Returns
    -------
    Dict[str, float]
        包含各种评估指标的字典
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    计算分类任务的评估指标
    Compute classification metrics
    
    Parameters
    ----------
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray
        预测标签
    y_prob : np.ndarray, optional
        预测概率（用于计算AUC）
        
    Returns
    -------
    Dict[str, float]
        包含各种评估指标的字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        metrics['AUC'] = auc
    
    return metrics


def normalize_data(data: np.ndarray, mean: Optional[float] = None, 
                  std: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """
    标准化数据
    Normalize data using z-score normalization
    
    Parameters
    ----------
    data : np.ndarray
        输入数据
    mean : float, optional
        预设均值（用于测试集）
    std : float, optional
        预设标准差（用于测试集）
        
    Returns
    -------
    Tuple[np.ndarray, float, float]
        标准化后的数据、均值、标准差
    """
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    
    normalized_data = (data - mean) / (std + 1e-8)  # 避免除零
    return normalized_data, mean, std


def denormalize_data(normalized_data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    反标准化数据
    Denormalize data
    
    Parameters
    ----------
    normalized_data : np.ndarray
        标准化后的数据
    mean : float
        原始数据均值
    std : float
        原始数据标准差
        
    Returns
    -------
    np.ndarray
        反标准化后的数据
    """
    return normalized_data * std + mean


def create_directory(path: str) -> None:
    """
    创建目录（如果不存在）
    Create directory if it doesn't exist
    
    Parameters
    ----------
    path : str
        目录路径
    """
    os.makedirs(path, exist_ok=True)


def get_timestamp() -> str:
    """
    获取当前时间戳字符串
    Get current timestamp string
    
    Returns
    -------
    str
        时间戳字符串，格式：YYYY-MM-DD_HH-MM-SS
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def set_random_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保结果可重现
    Set random seed for reproducibility
    
    Parameters
    ----------
    seed : int
        随机种子
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def log_experiment_info(args: Any, model: nn.Module, save_path: str) -> None:
    """
    记录实验信息
    Log experiment information
    
    Parameters
    ----------
    args : Any
        命令行参数或配置对象
    model : nn.Module
        模型对象
    save_path : str
        保存路径
    """
    info = {
        'timestamp': get_timestamp(),
        'model_parameters': count_parameters(model),
        'arguments': vars(args) if hasattr(args, '__dict__') else str(args)
    }
    
    save_dict_to_json(info, os.path.join(save_path, 'experiment_info.json'))


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    Format time display
    
    Parameters
    ----------
    seconds : float
        秒数
        
    Returns
    -------
    str
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m {seconds%60:.0f}s"


def print_progress_bar(iteration: int, total: int, prefix: str = '', 
                      suffix: str = '', length: int = 50, fill: str = '█') -> None:
    """
    打印进度条
    Print progress bar
    
    Parameters
    ----------
    iteration : int
        当前迭代数
    total : int
        总迭代数
    prefix : str
        前缀文本
    suffix : str
        后缀文本
    length : int
        进度条长度
    fill : str
        填充字符
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def validate_cif_file(cif_path: str) -> bool:
    """
    验证CIF文件是否有效
    Validate if CIF file is valid
    
    Parameters
    ----------
    cif_path : str
        CIF文件路径
        
    Returns
    -------
    bool
        文件是否有效
    """
    try:
        from pymatgen.core.structure import Structure
        Structure.from_file(cif_path)
        return True
    except Exception as e:
        print(f"Invalid CIF file {cif_path}: {e}")
        return False


def check_gpu_availability() -> Dict[str, Any]:
    """
    检查GPU可用性
    Check GPU availability
    
    Returns
    -------
    Dict[str, Any]
        GPU信息字典
    """
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': []
    }
    
    if torch.cuda.is_available():
        gpu_info['device_count'] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i)
            }
            gpu_info['devices'].append(device_info)
    
    return gpu_info


class Timer:
    """
    计时器类
    Timer class for measuring execution time
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """开始计时"""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        停止计时并返回耗时
        
        Returns
        -------
        float
            耗时（秒）
        """
        self.end_time = time.time()
        if self.start_time is None:
            raise ValueError("Timer not started")
        return self.end_time - self.start_time
    
    def elapsed(self) -> float:
        """
        获取当前耗时（不停止计时）
        
        Returns
        -------
        float
            当前耗时（秒）
        """
        if self.start_time is None:
            raise ValueError("Timer not started")
        return time.time() - self.start_time 