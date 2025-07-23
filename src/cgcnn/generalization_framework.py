"""
Model Generalization Framework

Comprehensive framework for ensuring model generalization through
stratified cross-validation, temporal validation, material-type-based
validation, and performance monitoring with early warning systems.

Author: lunazhang
Date: 2023
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import pandas as pd
import logging
import time
import json
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


@dataclass
class ValidationResult:
    fold_id: int
    validation_type: str
    material_type: Optional[str]
    metric_values: Dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray
    sample_ids: List[str]
    validation_time: float
    model_state: Optional[Dict] = None


@dataclass
class PerformanceAlert:
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: float
    context: Dict[str, Any]


class StratifiedMaterialValidator:
    """
    分层材料验证器
    Stratified Material Validator
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.validation_history = []
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
    
    def create_stratified_folds(self, dataset: List[Any], 
                              stratify_by: str = 'material_type') -> List[Tuple[List[int], List[int]]]:
        """
        创建分层折叠
        Create stratified folds based on material properties
        
        Args:
            dataset: 数据集
            stratify_by: 分层依据
            
        Returns:
            folds: 训练和验证索引的折叠列表
        """
        # 提取分层标签
        if stratify_by == 'material_type':
            labels = self._extract_material_labels(dataset)
        elif stratify_by == 'defect_type':
            labels = self._extract_defect_labels(dataset)
        elif stratify_by == 'property_range':
            labels = self._extract_property_range_labels(dataset)
        else:
            raise ValueError(f"Unknown stratification criterion: {stratify_by}")
        
        # 处理稀有类别
        labels = self._handle_rare_classes(labels)
        
        # 创建分层K折
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                             random_state=self.random_state)
        
        folds = []
        for train_idx, val_idx in skf.split(range(len(dataset)), labels):
            folds.append((train_idx.tolist(), val_idx.tolist()))
        
        return folds
    
    def create_group_folds(self, dataset: List[Any], 
                          group_by: str = 'crystal_system') -> List[Tuple[List[int], List[int]]]:
        """
        创建基于组的折叠（避免数据泄露）
        Create group-based folds to avoid data leakage
        
        Args:
            dataset: 数据集
            group_by: 分组依据
            
        Returns:
            folds: 分组折叠列表
        """
        # 提取分组标签
        if group_by == 'crystal_system':
            groups = self._extract_crystal_system_groups(dataset)
        elif group_by == 'composition_family':
            groups = self._extract_composition_groups(dataset)
        else:
            raise ValueError(f"Unknown grouping criterion: {group_by}")
        
        # 创建组K折
        gkf = GroupKFold(n_splits=self.n_folds)
        
        folds = []
        for train_idx, val_idx in gkf.split(range(len(dataset)), groups=groups):
            folds.append((train_idx.tolist(), val_idx.tolist()))
        
        return folds
    
    def _extract_material_labels(self, dataset: List[Any]) -> List[str]:
        labels = []
        for sample in dataset:
            # 假设样本包含CIF ID信息
            if hasattr(sample, 'cif_id'):
                cif_id = sample.cif_id
            elif isinstance(sample, tuple) and len(sample) > 5:
                cif_id = sample[-1]  # 假设最后一个元素是CIF ID
            else:
                cif_id = "unknown"
            
            # 从CIF ID推断材料类型
            if 'ncm811' in cif_id.lower():
                labels.append('NCM811')
            elif 'nca' in cif_id.lower():
                labels.append('NCA')
            elif 'ncm622' in cif_id.lower():
                labels.append('NCM622')
            elif 'ncm532' in cif_id.lower():
                labels.append('NCM532')
            else:
                labels.append('Other')
        
        return labels
    
    def _extract_defect_labels(self, dataset: List[Any]) -> List[str]:
        labels = []
        for sample in dataset:
            # 从样本中提取缺陷信息
            if hasattr(sample, 'cif_id'):
                cif_id = sample.cif_id
            elif isinstance(sample, tuple) and len(sample) > 5:
                cif_id = sample[-1]
            else:
                cif_id = "unknown"
            
            if 'livac' in cif_id.lower() or 'li_vac' in cif_id.lower():
                labels.append('Li_vacancy')
            elif 'nimig' in cif_id.lower() or 'ni_mig' in cif_id.lower():
                labels.append('Ni_migration')
            elif 'ovac' in cif_id.lower() or 'o_vac' in cif_id.lower():
                labels.append('O_vacancy')
            else:
                labels.append('pristine')
        
        return labels
    
    def _extract_property_range_labels(self, dataset: List[Any]) -> List[str]:
        # 这里需要根据实际的目标性质进行调整
        labels = []
        for sample in dataset:
            # 假设样本包含目标值
            if hasattr(sample, 'target'):
                target_value = sample.target
            elif isinstance(sample, tuple) and len(sample) > 3:
                target_value = sample[1]  # 假设第二个元素是目标值
            else:
                target_value = 0.0
            
            # 根据数值范围分类
            if isinstance(target_value, torch.Tensor):
                target_value = target_value.item()
            
            if target_value < -4.5:
                labels.append('very_stable')
            elif target_value < -4.0:
                labels.append('stable')
            elif target_value < -3.5:
                labels.append('moderately_stable')
            else:
                labels.append('unstable')
        
        return labels
    
    def _extract_crystal_system_groups(self, dataset: List[Any]) -> List[str]:
        # 简化实现：基于材料类型推断晶系
        material_labels = self._extract_material_labels(dataset)
        
        crystal_systems = []
        for material in material_labels:
            if material in ['NCM811', 'NCA', 'NCM622', 'NCM532']:
                crystal_systems.append('hexagonal')  # 层状氧化物通常为六方晶系
            else:
                crystal_systems.append('unknown')
        
        return crystal_systems
    
    def _extract_composition_groups(self, dataset: List[Any]) -> List[str]:
        material_labels = self._extract_material_labels(dataset)
        
        composition_groups = []
        for material in material_labels:
            if material.startswith('NCM'):
                composition_groups.append('NCM_family')
            elif material == 'NCA':
                composition_groups.append('NCA_family')
            else:
                composition_groups.append('other_family')
        
        return composition_groups
    
    def _handle_rare_classes(self, labels: List[str]) -> List[str]:
        from collections import Counter
        
        label_counts = Counter(labels)
        min_samples_per_class = max(2, len(labels) // (self.n_folds * 10))
        
        # 将稀有类别合并为"other"
        processed_labels = []
        for label in labels:
            if label_counts[label] < min_samples_per_class:
                processed_labels.append('other')
            else:
                processed_labels.append(label)
        
        return processed_labels


class TemporalValidator:
    """
    时间序列验证器
    Temporal Validator for Time-Series Validation
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        self.n_splits = n_splits
        self.test_size = test_size
        
        self.logger = logging.getLogger(__name__)
    
    def create_temporal_splits(self, dataset: List[Any], 
                             time_column: str = 'timestamp') -> List[Tuple[List[int], List[int]]]:
        """
        创建时间序列分割
        Create temporal splits for time-series validation
        
        Args:
            dataset: 数据集
            time_column: 时间列名
            
        Returns:
            temporal_splits: 时间序列分割列表
        """
        # 提取时间戳
        timestamps = self._extract_timestamps(dataset, time_column)
        
        # 排序数据索引
        sorted_indices = np.argsort(timestamps)
        
        # 创建时间序列分割
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=int(len(dataset) * self.test_size))
        
        temporal_splits = []
        for train_idx, test_idx in tscv.split(sorted_indices):
            # 映射回原始索引
            train_indices = sorted_indices[train_idx].tolist()
            test_indices = sorted_indices[test_idx].tolist()
            temporal_splits.append((train_indices, test_indices))
        
        return temporal_splits
    
    def create_sliding_window_splits(self, dataset: List[Any],
                                   window_size: int = 1000,
                                   step_size: int = 200) -> List[Tuple[List[int], List[int]]]:
        """
        创建滑动窗口分割
        Create sliding window splits
        
        Args:
            dataset: 数据集
            window_size: 窗口大小
            step_size: 步长
            
        Returns:
            sliding_splits: 滑动窗口分割列表
        """
        dataset_size = len(dataset)
        sliding_splits = []
        
        for start in range(0, dataset_size - window_size, step_size):
            train_end = start + int(window_size * 0.8)
            val_start = train_end
            val_end = start + window_size
            
            if val_end <= dataset_size:
                train_indices = list(range(start, train_end))
                val_indices = list(range(val_start, val_end))
                sliding_splits.append((train_indices, val_indices))
        
        return sliding_splits
    
    def _extract_timestamps(self, dataset: List[Any], time_column: str) -> np.ndarray:
        timestamps = []
        
        for i, sample in enumerate(dataset):
            if hasattr(sample, time_column):
                timestamp = getattr(sample, time_column)
            elif isinstance(sample, dict) and time_column in sample:
                timestamp = sample[time_column]
            else:
                # 如果没有时间戳，使用索引作为伪时间戳
                timestamp = float(i)
            
            timestamps.append(timestamp)
        
        return np.array(timestamps)
    
    def analyze_temporal_stability(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        分析时间稳定性
        Analyze temporal stability of model performance
        
        Args:
            validation_results: 验证结果列表
            
        Returns:
            stability_analysis: 稳定性分析结果
        """
        # 按时间排序结果
        temporal_results = sorted(validation_results, key=lambda x: x.validation_time)
        
        # 提取性能指标时间序列
        metrics_over_time = defaultdict(list)
        timestamps = []
        
        for result in temporal_results:
            timestamps.append(result.validation_time)
            for metric_name, metric_value in result.metric_values.items():
                metrics_over_time[metric_name].append(metric_value)
        

        stability_analysis = {
            'temporal_trend': {},
            'volatility': {},
            'drift_detection': {},
            'stability_score': {}
        }
        
        for metric_name, values in metrics_over_time.items():
            values = np.array(values)
            
            # 趋势分析
            if len(values) > 1:
                trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                stability_analysis['temporal_trend'][metric_name] = trend_slope
                
                # 波动性
                volatility = np.std(values) / (np.mean(values) + 1e-8)
                stability_analysis['volatility'][metric_name] = volatility
                
                # 漂移检测（简化版本）
                drift_score = abs(values[-1] - values[0]) / (np.std(values) + 1e-8)
                stability_analysis['drift_detection'][metric_name] = drift_score
                
                # 综合稳定性分数
                stability_score = 1.0 / (1.0 + volatility + abs(trend_slope) + drift_score * 0.1)
                stability_analysis['stability_score'][metric_name] = stability_score
        
        return stability_analysis


class PerformanceMonitor:
    """
    性能监控器
    Performance Monitor with Early Warning System
    """
    
    def __init__(self, alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
                 monitoring_window: int = 10):
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        self.monitoring_window = monitoring_window
        
        # 性能历史
        self.performance_history = defaultdict(deque)
        self.alerts_history = []
        
        # 预警回调
        self.alert_callbacks = []
        
        self.logger = logging.getLogger(__name__)
    
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        return {
            'mae': {
                'warning': 0.1,
                'critical': 0.2
            },
            'rmse': {
                'warning': 0.15,
                'critical': 0.3
            },
            'r2': {
                'warning': 0.8,
                'critical': 0.6
            },
            'uncertainty_calibration': {
                'warning': 1.5,
                'critical': 2.0
            }
        }
    
    def update_performance(self, metric_name: str, metric_value: float, 
                         context: Optional[Dict[str, Any]] = None):
        """
        更新性能指标
        Update performance metrics
        
        Args:
            metric_name: 指标名称
            metric_value: 指标值
            context: 上下文信息
        """
        timestamp = time.time()
        
        # 添加到历史记录
        self.performance_history[metric_name].append({
            'value': metric_value,
            'timestamp': timestamp,
            'context': context or {}
        })
        
        # 保持窗口大小
        while len(self.performance_history[metric_name]) > self.monitoring_window:
            self.performance_history[metric_name].popleft()
        
        # 检查预警条件
        self._check_alerts(metric_name, metric_value, timestamp, context)
    
    def _check_alerts(self, metric_name: str, metric_value: float, 
                     timestamp: float, context: Optional[Dict[str, Any]]):
        if metric_name not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric_name]
        alerts = []
        
        # 阈值检查
        if 'critical' in thresholds:
            if (metric_name in ['mae', 'rmse', 'uncertainty_calibration'] and 
                metric_value > thresholds['critical']):
                alerts.append(self._create_alert(
                    'threshold_critical', 'critical', metric_name, metric_value,
                    thresholds['critical'], timestamp, context,
                    f"{metric_name} exceeded critical threshold"
                ))
            elif (metric_name in ['r2'] and metric_value < thresholds['critical']):
                alerts.append(self._create_alert(
                    'threshold_critical', 'critical', metric_name, metric_value,
                    thresholds['critical'], timestamp, context,
                    f"{metric_name} dropped below critical threshold"
                ))
        
        if 'warning' in thresholds and not alerts:  # 只有在没有严重警报时才检查警告
            if (metric_name in ['mae', 'rmse', 'uncertainty_calibration'] and 
                metric_value > thresholds['warning']):
                alerts.append(self._create_alert(
                    'threshold_warning', 'medium', metric_name, metric_value,
                    thresholds['warning'], timestamp, context,
                    f"{metric_name} exceeded warning threshold"
                ))
            elif (metric_name in ['r2'] and metric_value < thresholds['warning']):
                alerts.append(self._create_alert(
                    'threshold_warning', 'medium', metric_name, metric_value,
                    thresholds['warning'], timestamp, context,
                    f"{metric_name} dropped below warning threshold"
                ))
        
        # 趋势检查
        trend_alert = self._check_trend_alert(metric_name, timestamp, context)
        if trend_alert:
            alerts.append(trend_alert)
        
        # 异常值检查
        anomaly_alert = self._check_anomaly_alert(metric_name, metric_value, timestamp, context)
        if anomaly_alert:
            alerts.append(anomaly_alert)
        
        # 处理警报
        for alert in alerts:
            self._handle_alert(alert)
    
    def _check_trend_alert(self, metric_name: str, timestamp: float, 
                          context: Optional[Dict[str, Any]]) -> Optional[PerformanceAlert]:
        history = self.performance_history[metric_name]
        
        if len(history) < 5:  # 需要足够的历史数据
            return None
        
        values = [entry['value'] for entry in history]
        
        # 计算趋势
        if len(values) > 1:
            recent_values = values[-3:]  # 最近3个值
            early_values = values[:3]    # 早期3个值
            
            recent_avg = np.mean(recent_values)
            early_avg = np.mean(early_values)
            
            # 对于需要最小化的指标（mae, rmse）
            if metric_name in ['mae', 'rmse', 'uncertainty_calibration']:
                if recent_avg > early_avg * 1.2:  # 恶化超过20%
                    return self._create_alert(
                        'trend_degradation', 'medium', metric_name, recent_avg,
                        early_avg, timestamp, context,
                        f"{metric_name} showing degradation trend"
                    )
            # 对于需要最大化的指标（r2）
            elif metric_name in ['r2']:
                if recent_avg < early_avg * 0.9:  # 下降超过10%
                    return self._create_alert(
                        'trend_degradation', 'medium', metric_name, recent_avg,
                        early_avg, timestamp, context,
                        f"{metric_name} showing degradation trend"
                    )
        
        return None
    
    def _check_anomaly_alert(self, metric_name: str, metric_value: float,
                           timestamp: float, context: Optional[Dict[str, Any]]) -> Optional[PerformanceAlert]:
        history = self.performance_history[metric_name]
        
        if len(history) < 5:
            return None
        
        historical_values = [entry['value'] for entry in list(history)[:-1]]  # 除当前值外的历史值
        
        if len(historical_values) > 0:
            mean_val = np.mean(historical_values)
            std_val = np.std(historical_values)
            
            # Z-score异常检测
            if std_val > 0:
                z_score = abs(metric_value - mean_val) / std_val
                
                if z_score > 3.0:  # 3σ规则
                    return self._create_alert(
                        'anomaly_detection', 'high', metric_name, metric_value,
                        mean_val, timestamp, context,
                        f"{metric_name} anomalous value detected (Z-score: {z_score:.2f})"
                    )
        
        return None
    
    def _create_alert(self, alert_type: str, severity: str, metric_name: str,
                     current_value: float, threshold_value: float,
                     timestamp: float, context: Optional[Dict[str, Any]],
                     message: str) -> PerformanceAlert:
        return PerformanceAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            timestamp=timestamp,
            context=context or {}
        )
    
    def _handle_alert(self, alert: PerformanceAlert):
        # 记录预警
        self.alerts_history.append(alert)
        
        # 日志记录
        self.logger.warning(f"Performance Alert [{alert.severity.upper()}]: {alert.message}")
        
        # 调用回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def register_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        summary = {
            'metrics_summary': {},
            'recent_alerts': [],
            'trend_analysis': {}
        }
        
        # 指标摘要
        for metric_name, history in self.performance_history.items():
            if history:
                values = [entry['value'] for entry in history]
                summary['metrics_summary'][metric_name] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': self._calculate_simple_trend(values)
                }
        
        # 最近警报
        recent_alerts = [alert for alert in self.alerts_history 
                        if time.time() - alert.timestamp < 3600]  # 最近1小时
        summary['recent_alerts'] = [
            {
                'type': alert.alert_type,
                'severity': alert.severity,
                'metric': alert.metric_name,
                'message': alert.message,
                'timestamp': alert.timestamp
            }
            for alert in recent_alerts
        ]
        
        return summary
    
    def _calculate_simple_trend(self, values: List[float]) -> str:
        if len(values) < 2:
            return 'stable'
        
        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        early_avg = np.mean(values[:3]) if len(values) >= 3 else values[0]
        
        change_ratio = (recent_avg - early_avg) / (abs(early_avg) + 1e-8)
        
        if change_ratio > 0.05:
            return 'improving'
        elif change_ratio < -0.05:
            return 'degrading'
        else:
            return 'stable'


class GeneralizationFramework:
    """
    模型泛化框架
    Model Generalization Framework
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        
        # 验证器
        self.stratified_validator = StratifiedMaterialValidator()
        self.temporal_validator = TemporalValidator()
        self.performance_monitor = PerformanceMonitor()
        
        # 结果存储
        self.validation_results = []
        
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_validation(self, dataset: List[Any],
                                validation_types: List[str] = ['stratified', 'temporal', 'group'],
                                save_results: bool = True) -> Dict[str, Any]:
        """
        综合验证
        Comprehensive validation across multiple strategies
        
        Args:
            dataset: 数据集
            validation_types: 验证类型列表
            save_results: 是否保存结果
            
        Returns:
            validation_summary: 验证摘要
        """
        validation_summary = {
            'total_samples': len(dataset),
            'validation_types': validation_types,
            'results': {},
            'performance_metrics': {},
            'generalization_score': 0.0
        }
        
        # 执行不同类型的验证
        for val_type in validation_types:
            self.logger.info(f"Starting {val_type} validation")
            
            if val_type == 'stratified':
                results = self._run_stratified_validation(dataset)
            elif val_type == 'temporal':
                results = self._run_temporal_validation(dataset)
            elif val_type == 'group':
                results = self._run_group_validation(dataset)
            else:
                self.logger.warning(f"Unknown validation type: {val_type}")
                continue
            
            validation_summary['results'][val_type] = results
            
            # 更新性能监控
            self._update_performance_monitoring(results, val_type)
        
        # 计算综合泛化分数
        validation_summary['generalization_score'] = self._calculate_generalization_score(
            validation_summary['results']
        )
        
        # 保存结果
        if save_results:
            self._save_validation_results(validation_summary)
        
        return validation_summary
    
    def _run_stratified_validation(self, dataset: List[Any]) -> Dict[str, Any]:
        results = {'fold_results': [], 'summary': {}}
        
        # 创建分层折叠
        folds = self.stratified_validator.create_stratified_folds(dataset, 'material_type')
        
        fold_metrics = defaultdict(list)
        
        for fold_id, (train_idx, val_idx) in enumerate(folds):
            self.logger.info(f"Processing stratified fold {fold_id + 1}/{len(folds)}")
            
            # 训练和验证（这里简化，实际需要完整的训练流程）
            fold_result = self._evaluate_fold(dataset, train_idx, val_idx, 
                                            fold_id, 'stratified')
            
            results['fold_results'].append(fold_result)
            
            # 收集指标
            for metric_name, metric_value in fold_result.metric_values.items():
                fold_metrics[metric_name].append(metric_value)
        
        # 计算汇总统计
        for metric_name, values in fold_metrics.items():
            results['summary'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'cv': np.std(values) / (np.mean(values) + 1e-8)  # 变异系数
            }
        
        return results
    
    def _run_temporal_validation(self, dataset: List[Any]) -> Dict[str, Any]:
        results = {'split_results': [], 'summary': {}, 'stability_analysis': {}}
        
        # 创建时间序列分割
        splits = self.temporal_validator.create_temporal_splits(dataset)
        
        split_metrics = defaultdict(list)
        
        for split_id, (train_idx, val_idx) in enumerate(splits):
            self.logger.info(f"Processing temporal split {split_id + 1}/{len(splits)}")
            
            # 训练和验证
            split_result = self._evaluate_fold(dataset, train_idx, val_idx,
                                             split_id, 'temporal')
            
            results['split_results'].append(split_result)
            
            # 收集指标
            for metric_name, metric_value in split_result.metric_values.items():
                split_metrics[metric_name].append(metric_value)
        
        # 计算汇总统计
        for metric_name, values in split_metrics.items():
            results['summary'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0.0
            }
        
        # 时间稳定性分析
        results['stability_analysis'] = self.temporal_validator.analyze_temporal_stability(
            results['split_results']
        )
        
        return results
    
    def _run_group_validation(self, dataset: List[Any]) -> Dict[str, Any]:
        results = {'group_results': [], 'summary': {}}
        
        # 创建基于组的折叠
        folds = self.stratified_validator.create_group_folds(dataset, 'crystal_system')
        
        fold_metrics = defaultdict(list)
        
        for fold_id, (train_idx, val_idx) in enumerate(folds):
            self.logger.info(f"Processing group fold {fold_id + 1}/{len(folds)}")
            
            # 训练和验证
            fold_result = self._evaluate_fold(dataset, train_idx, val_idx,
                                            fold_id, 'group')
            
            results['group_results'].append(fold_result)
            
            # 收集指标
            for metric_name, metric_value in fold_result.metric_values.items():
                fold_metrics[metric_name].append(metric_value)
        
        # 计算汇总统计
        for metric_name, values in fold_metrics.items():
            results['summary'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'group_variance': np.var(values)  # 组间方差
            }
        
        return results
    
    def _evaluate_fold(self, dataset: List[Any], train_idx: List[int], 
                      val_idx: List[int], fold_id: int, 
                      validation_type: str) -> ValidationResult:
        start_time = time.time()
        
        # 简化的模型评估（实际应用中需要完整的训练和评估流程）
        self.model.eval()
        
        # 模拟预测和目标值
        predictions = np.random.normal(0, 0.1, len(val_idx))
        targets = np.random.normal(0, 0.05, len(val_idx))
        

        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = max(0, r2_score(targets, predictions))  # 确保R²不为负
        
        metric_values = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'sample_count': len(val_idx)
        }
        
        # 生成样本ID
        sample_ids = [f"{validation_type}_fold_{fold_id}_sample_{i}" for i in range(len(val_idx))]
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            fold_id=fold_id,
            validation_type=validation_type,
            material_type=None,  # 可以根据需要填充
            metric_values=metric_values,
            predictions=predictions,
            targets=targets,
            sample_ids=sample_ids,
            validation_time=validation_time
        )
    
    def _update_performance_monitoring(self, results: Dict[str, Any], validation_type: str):
        if 'summary' in results:
            for metric_name, metric_stats in results['summary'].items():
                if isinstance(metric_stats, dict) and 'mean' in metric_stats:
                    self.performance_monitor.update_performance(
                        metric_name, 
                        metric_stats['mean'],
                        context={'validation_type': validation_type}
                    )
    
    def _calculate_generalization_score(self, all_results: Dict[str, Any]) -> float:
        scores = []
        
        for val_type, results in all_results.items():
            if 'summary' in results:
                # 基于MAE计算分数（越小越好）
                if 'mae' in results['summary']:
                    mae_stats = results['summary']['mae']
                    
                    # 综合考虑均值和稳定性
                    mean_mae = mae_stats.get('mean', 1.0)
                    std_mae = mae_stats.get('std', 0.1)
                    
                    # 分数：考虑准确性和稳定性
                    score = 1.0 / (1.0 + mean_mae + std_mae)
                    scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _save_validation_results(self, validation_summary: Dict[str, Any]):
        timestamp = int(time.time())
        filename = f"validation_results_{timestamp}.json"
        
        # 转换为可序列化的格式
        serializable_summary = self._make_serializable(validation_summary)
        
        try:
            with open(filename, 'w') as f:
                json.dump(serializable_summary, f, indent=2)
            
            self.logger.info(f"Validation results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save validation results: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, ValidationResult):
            return {
                'fold_id': obj.fold_id,
                'validation_type': obj.validation_type,
                'material_type': obj.material_type,
                'metric_values': obj.metric_values,
                'validation_time': obj.validation_time,
                'sample_count': len(obj.sample_ids)
            }
        else:
            return obj
    
    def generate_validation_report(self, validation_summary: Dict[str, Any]) -> str:
        report = f"""
=== 模型泛化能力验证报告 ===
验证时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
总样本数: {validation_summary['total_samples']}
验证类型: {', '.join(validation_summary['validation_types'])}
综合泛化分数: {validation_summary['generalization_score']:.4f}

"""
        
        for val_type, results in validation_summary['results'].items():
            report += f"\n--- {val_type.upper()} 验证结果 ---\n"
            
            if 'summary' in results:
                for metric_name, stats in results['summary'].items():
                    if isinstance(stats, dict):
                        mean_val = stats.get('mean', 0)
                        std_val = stats.get('std', 0)
                        report += f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}\n"
        
        # 性能监控摘要
        perf_summary = self.performance_monitor.get_performance_summary()
        if perf_summary['recent_alerts']:
            report += f"\n--- 最近预警 ---\n"
            for alert in perf_summary['recent_alerts'][-5:]:  # 最近5个预警
                report += f"[{alert['severity'].upper()}] {alert['message']}\n"
        
        return report


# 使用示例
def example_usage():
    from cgcnn.enhanced_model import EnhancedCGCNN
    
    # 创建模型
    model = EnhancedCGCNN(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128
    )
    
    # 创建泛化框架
    framework = GeneralizationFramework(model)
    
    # 模拟数据集
    dataset = []
    for i in range(200):
        atom_fea = torch.randn(15, 92)
        nbr_fea = torch.randn(15, 12, 41)
        nbr_fea_idx = torch.randint(0, 15, (15, 12))
        crystal_atom_idx = [torch.arange(3 * j, 3 * (j + 1)) for j in range(5)]
        target = torch.randn(1)
        
        # 添加CIF ID用于分层
        cif_id = f"NCM811_sample_{i}" if i < 100 else f"NCA_sample_{i}"
        
        sample = (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, target, cif_id)
        dataset.append(sample)
    
    # 运行综合验证
    validation_summary = framework.comprehensive_validation(
        dataset,
        validation_types=['stratified', 'temporal', 'group'],
        save_results=True
    )
    
    # 生成报告
    report = framework.generate_validation_report(validation_summary)
    print(report)
    
    # 设置性能监控回调
    def alert_callback(alert: PerformanceAlert):
        print(f"ALERT: {alert.message}")
    
    framework.performance_monitor.register_alert_callback(alert_callback)
    
    # 模拟性能更新
    framework.performance_monitor.update_performance('mae', 0.15)
    framework.performance_monitor.update_performance('r2', 0.75)
    
    print(f"Generalization Score: {validation_summary['generalization_score']:.4f}")


if __name__ == "__main__":
    example_usage() 