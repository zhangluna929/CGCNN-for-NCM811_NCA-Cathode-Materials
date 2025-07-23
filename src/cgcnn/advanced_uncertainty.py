"""
Advanced Uncertainty Quantification Module

Enhanced uncertainty analysis with epistemic/aleatoric decomposition,
uncertainty-guided active learning, and automatic DFT validation workflow.

Author: lunazhang
Date: 2023
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.cluster import KMeans
    from scipy.stats import entropy
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .uncertainty import BayesianCGCNN, EnsembleCGCNN, UncertaintyMetrics


class UncertaintyDecomposer:
    """
    不确定性分解器
    Uncertainty Decomposer for Epistemic/Aleatoric Separation
    """
    
    def __init__(self, model: Union[BayesianCGCNN, EnsembleCGCNN], n_samples: int = 100):
        self.model = model
        self.n_samples = n_samples
        self.uncertainty_metrics = UncertaintyMetrics()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def decompose_uncertainty(self, input_data: Tuple, 
                            method: str = 'mc_dropout') -> Dict[str, torch.Tensor]:
        """
        分解不确定性为认识不确定性和偶然不确定性
        Decompose uncertainty into epistemic and aleatoric components
        
        Args:
            input_data: 输入数据
            method: 分解方法 ('mc_dropout', 'ensemble', 'variational')
            
        Returns:
            uncertainty_components: 不确定性组件
        """
        if method == 'mc_dropout':
            return self._mc_dropout_decomposition(input_data)
        elif method == 'ensemble':
            return self._ensemble_decomposition(input_data)
        elif method == 'variational':
            return self._variational_decomposition(input_data)
        else:
            raise ValueError(f"Unknown uncertainty decomposition method: {method}")
    
    def _mc_dropout_decomposition(self, input_data: Tuple) -> Dict[str, torch.Tensor]:
        """蒙特卡洛Dropout分解"""
        self.model.train()  # 启用dropout
        
        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(*input_data)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [n_samples, batch_size, output_dim]
        
        # 预测均值和方差
        predictive_mean = torch.mean(predictions, dim=0)
        predictive_variance = torch.var(predictions, dim=0)
        
        # MC-Dropout中，总不确定性近似等于认识不确定性
        epistemic_uncertainty = torch.sqrt(predictive_variance)
        
        # 偶然不确定性需要额外建模（这里使用简化估计）
        aleatoric_uncertainty = self._estimate_aleatoric_uncertainty(input_data, predictions)
        
        return {
            'predictive_mean': predictive_mean,
            'total_uncertainty': torch.sqrt(predictive_variance),
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'all_predictions': predictions
        }
    
    def _ensemble_decomposition(self, input_data: Tuple) -> Dict[str, torch.Tensor]:
        """集成模型分解"""
        if not isinstance(self.model, EnsembleCGCNN):
            raise ValueError("Ensemble decomposition requires EnsembleCGCNN model")
        
        # 获取集成预测和不确定性
        mean, epistemic_std, aleatoric_std = self.model.predict_with_aleatoric_uncertainty(
            *input_data, n_dropout=self.n_samples)
        
        total_uncertainty = torch.sqrt(epistemic_std**2 + aleatoric_std**2)
        
        return {
            'predictive_mean': mean,
            'total_uncertainty': total_uncertainty,
            'epistemic_uncertainty': epistemic_std,
            'aleatoric_uncertainty': aleatoric_std
        }
    
    def _variational_decomposition(self, input_data: Tuple) -> Dict[str, torch.Tensor]:
        """变分推断分解"""
        if not isinstance(self.model, BayesianCGCNN):
            raise ValueError("Variational decomposition requires BayesianCGCNN model")
        
        # 变分贝叶斯方法
        predictions = []
        kl_divergences = []
        
        for _ in range(self.n_samples):
            pred = self.model(*input_data)
            kl_div = self.model.kl_divergence()
            
            predictions.append(pred)
            kl_divergences.append(kl_div)
        
        predictions = torch.stack(predictions)
        
        # 认识不确定性与模型参数不确定性相关
        predictive_mean = torch.mean(predictions, dim=0)
        epistemic_variance = torch.var(predictions, dim=0)
        
        # 偶然不确定性建模
        aleatoric_variance = self._model_aleatoric_variance(input_data)
        
        return {
            'predictive_mean': predictive_mean,
            'total_uncertainty': torch.sqrt(epistemic_variance + aleatoric_variance),
            'epistemic_uncertainty': torch.sqrt(epistemic_variance),
            'aleatoric_uncertainty': torch.sqrt(aleatoric_variance),
            'kl_divergence': torch.mean(torch.stack(kl_divergences))
        }
    
    def _estimate_aleatoric_uncertainty(self, input_data: Tuple, 
                                      predictions: torch.Tensor) -> torch.Tensor:
        """估计偶然不确定性"""
        # 简化的偶然不确定性估计
        # 实际应用中可能需要专门的噪声建模
        
        # 基于输入复杂度的启发式估计
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 原子数量和结构复杂度
        n_atoms = atom_fea.size(0)
        n_neighbors = nbr_fea.size(1)
        
        # 复杂度指标
        structural_complexity = torch.log(torch.tensor(n_atoms * n_neighbors, dtype=torch.float32))
        
        # 基于特征变异性的不确定性
        atom_variance = torch.var(atom_fea, dim=1).mean()
        nbr_variance = torch.var(nbr_fea, dim=[1, 2]).mean()
        
        # 组合偶然不确定性
        base_aleatoric = 0.01  # 基础噪声水平
        complexity_factor = 0.001 * structural_complexity
        variance_factor = 0.1 * (atom_variance + nbr_variance)
        
        aleatoric_std = base_aleatoric + complexity_factor + variance_factor
        
        # 扩展到预测维度
        batch_size = predictions.size(1)
        output_dim = predictions.size(2) if predictions.dim() > 2 else 1
        
        return aleatoric_std * torch.ones(batch_size, output_dim, device=predictions.device)
    
    def _model_aleatoric_variance(self, input_data: Tuple) -> torch.Tensor:
        """建模偶然方差"""
        # 这里可以实现更复杂的偶然不确定性建模
        # 例如，基于输入特征预测噪声水平
        
        atom_fea, _, _, _ = input_data
        batch_size = len(atom_fea) if isinstance(atom_fea, list) else atom_fea.size(0)
        
        # 简化实现：返回常数偶然方差
        return torch.full((batch_size, 1), 0.01**2, device=atom_fea.device)
    
    def uncertainty_attribution_analysis(self, input_data: Tuple, 
                                       target_property: str) -> Dict[str, Any]:
        """
        不确定性归因分析
        Uncertainty attribution analysis
        
        Args:
            input_data: 输入数据
            target_property: 目标属性
            
        Returns:
            attribution: 不确定性归因结果
        """
        # 分解不确定性
        uncertainty_components = self.decompose_uncertainty(input_data)
        
        # 特征重要性分析
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        attribution = {
            'total_uncertainty': uncertainty_components['total_uncertainty'].item(),
            'epistemic_ratio': (uncertainty_components['epistemic_uncertainty'] / 
                              uncertainty_components['total_uncertainty']).item(),
            'aleatoric_ratio': (uncertainty_components['aleatoric_uncertainty'] / 
                              uncertainty_components['total_uncertainty']).item(),
            'dominant_source': None,
            'feature_contributions': {},
            'recommendations': []
        }
        
        # 确定主导不确定性源
        if attribution['epistemic_ratio'] > 0.7:
            attribution['dominant_source'] = 'epistemic'
            attribution['recommendations'].append('增加训练数据以减少模型不确定性')
            attribution['recommendations'].append('使用集成方法提高预测置信度')
        elif attribution['aleatoric_ratio'] > 0.7:
            attribution['dominant_source'] = 'aleatoric'
            attribution['recommendations'].append('改进数据质量以减少固有噪声')
            attribution['recommendations'].append('使用更精确的测量/计算方法')
        else:
            attribution['dominant_source'] = 'mixed'
            attribution['recommendations'].append('同时改进数据质量和模型容量')
        
        # 特征贡献分析
        if hasattr(atom_fea, 'requires_grad'):
            # 计算梯度以分析特征贡献
            atom_fea.requires_grad_(True)
            output = self.model(*input_data)
            grad = torch.autograd.grad(output.sum(), atom_fea, retain_graph=True)[0]
            
            feature_importance = torch.abs(grad).mean(dim=0)
            attribution['feature_contributions']['atom_features'] = feature_importance.tolist()
        
        return attribution


class UncertaintyGuidedActiveLearning:
    """
    不确定性指导的主动学习
    Uncertainty-Guided Active Learning
    """
    
    def __init__(self, uncertainty_decomposer: UncertaintyDecomposer,
                 acquisition_strategy: str = 'uncertainty_sampling'):
        self.uncertainty_decomposer = uncertainty_decomposer
        self.acquisition_strategy = acquisition_strategy
        self.sample_history = []
        
        # 采样策略配置
        self.strategy_configs = {
            'uncertainty_sampling': {'threshold': 0.1},
            'diverse_uncertainty': {'diversity_weight': 0.3},
            'expected_improvement': {'exploration_factor': 0.1},
            'thompson_sampling': {'n_samples': 50}
        }
    
    def select_samples_for_labeling(self, candidate_pool: List[Any], 
                                  n_samples: int = 10,
                                  current_dataset: Optional[List[Any]] = None) -> List[int]:
        """
        选择样本进行标注
        Select samples for labeling
        
        Args:
            candidate_pool: 候选样本池
            n_samples: 选择的样本数量
            current_dataset: 当前数据集（可选）
            
        Returns:
            selected_indices: 选择的样本索引
        """
        if self.acquisition_strategy == 'uncertainty_sampling':
            return self._uncertainty_sampling(candidate_pool, n_samples)
        elif self.acquisition_strategy == 'diverse_uncertainty':
            return self._diverse_uncertainty_sampling(candidate_pool, n_samples)
        elif self.acquisition_strategy == 'expected_improvement':
            return self._expected_improvement_sampling(candidate_pool, n_samples)
        elif self.acquisition_strategy == 'thompson_sampling':
            return self._thompson_sampling(candidate_pool, n_samples)
        else:
            raise ValueError(f"Unknown acquisition strategy: {self.acquisition_strategy}")
    
    def _uncertainty_sampling(self, candidate_pool: List[Any], n_samples: int) -> List[int]:
        """不确定性采样"""
        uncertainties = []
        
        for i, candidate in enumerate(candidate_pool):
            # 计算候选样本的不确定性
            uncertainty_components = self.uncertainty_decomposer.decompose_uncertainty(candidate)
            total_uncertainty = uncertainty_components['total_uncertainty'].item()
            uncertainties.append((i, total_uncertainty))
        
        # 按不确定性排序并选择前n个
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in uncertainties[:n_samples]]
        
        return selected_indices
    
    def _diverse_uncertainty_sampling(self, candidate_pool: List[Any], n_samples: int) -> List[int]:
        """多样性不确定性采样"""
        if not SKLEARN_AVAILABLE:
            # 如果sklearn不可用，回退到简单不确定性采样
            return self._uncertainty_sampling(candidate_pool, n_samples)
        
        # 计算所有候选样本的不确定性和特征
        uncertainties = []
        features = []
        
        for candidate in candidate_pool:
            uncertainty_components = self.uncertainty_decomposer.decompose_uncertainty(candidate)
            uncertainty = uncertainty_components['total_uncertainty'].item()
            
            # 提取特征用于多样性分析
            atom_fea, _, _, _ = candidate
            feature_vector = atom_fea.mean(dim=0).cpu().numpy()  # 简化的特征表示
            
            uncertainties.append(uncertainty)
            features.append(feature_vector)
        
        uncertainties = np.array(uncertainties)
        features = np.array(features)
        
        # 聚类以确保多样性
        n_clusters = min(n_samples * 2, len(candidate_pool))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # 从每个聚类中选择不确定性最高的样本
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                cluster_uncertainties = uncertainties[cluster_indices]
                best_in_cluster = cluster_indices[np.argmax(cluster_uncertainties)]
                selected_indices.append(best_in_cluster)
        
        # 如果聚类数不足，用不确定性最高的样本补充
        if len(selected_indices) < n_samples:
            remaining_indices = set(range(len(candidate_pool))) - set(selected_indices)
            remaining_uncertainties = [(i, uncertainties[i]) for i in remaining_indices]
            remaining_uncertainties.sort(key=lambda x: x[1], reverse=True)
            
            for i, _ in remaining_uncertainties[:n_samples - len(selected_indices)]:
                selected_indices.append(i)
        
        return selected_indices[:n_samples]
    
    def _expected_improvement_sampling(self, candidate_pool: List[Any], n_samples: int) -> List[int]:
        """期望改进采样"""
        # 实现期望改进采样策略
        expected_improvements = []
        
        for i, candidate in enumerate(candidate_pool):
            uncertainty_components = self.uncertainty_decomposer.decompose_uncertainty(candidate)
            
            # 计算期望改进（这里使用简化版本）
            uncertainty = uncertainty_components['total_uncertainty'].item()
            predicted_value = uncertainty_components['predictive_mean'].item()
            
            # 简化的期望改进计算
            exploration_factor = self.strategy_configs['expected_improvement']['exploration_factor']
            expected_improvement = predicted_value + exploration_factor * uncertainty
            
            expected_improvements.append((i, expected_improvement))
        
        # 选择期望改进最大的样本
        expected_improvements.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in expected_improvements[:n_samples]]
        
        return selected_indices
    
    def _thompson_sampling(self, candidate_pool: List[Any], n_samples: int) -> List[int]:
        """汤普森采样"""
        n_thompson_samples = self.strategy_configs['thompson_sampling']['n_samples']
        
        # 为每个候选样本生成多个预测采样
        candidate_scores = []
        
        for i, candidate in enumerate(candidate_pool):
            uncertainty_components = self.uncertainty_decomposer.decompose_uncertainty(candidate)
            
            # 从预测分布中采样
            mean = uncertainty_components['predictive_mean']
            std = uncertainty_components['total_uncertainty']
            
            # 生成采样分数
            samples = torch.normal(mean, std.expand_as(mean), size=(n_thompson_samples,))
            max_sample = torch.max(samples).item()
            
            candidate_scores.append((i, max_sample))
        
        # 选择分数最高的样本
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in candidate_scores[:n_samples]]
        
        return selected_indices
    
    def update_acquisition_strategy(self, performance_feedback: Dict[str, float]):
        """
        根据性能反馈更新采样策略
        Update acquisition strategy based on performance feedback
        
        Args:
            performance_feedback: 性能反馈信息
        """
        # 记录采样历史
        self.sample_history.append({
            'strategy': self.acquisition_strategy,
            'performance': performance_feedback
        })
        
        # 简单的策略适应
        if len(self.sample_history) >= 5:
            recent_performance = [h['performance']['improvement'] for h in self.sample_history[-5:]]
            avg_improvement = np.mean(recent_performance)
            
            # 如果性能改进不足，尝试更换策略
            if avg_improvement < 0.01:
                current_strategies = ['uncertainty_sampling', 'diverse_uncertainty', 
                                    'expected_improvement', 'thompson_sampling']
                current_idx = current_strategies.index(self.acquisition_strategy)
                next_idx = (current_idx + 1) % len(current_strategies)
                self.acquisition_strategy = current_strategies[next_idx]
                
                print(f"Switching acquisition strategy to: {self.acquisition_strategy}")


class HighUncertaintySampleProcessor:
    """
    高不确定性样本处理器
    High Uncertainty Sample Processor
    """
    
    def __init__(self, uncertainty_threshold: float = 0.15,
                 dft_validation_enabled: bool = True):
        self.uncertainty_threshold = uncertainty_threshold
        self.dft_validation_enabled = dft_validation_enabled
        self.processed_samples = []
        self.validation_queue = []
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
    
    def process_high_uncertainty_batch(self, predictions: torch.Tensor,
                                     uncertainties: torch.Tensor,
                                     sample_ids: List[str]) -> Dict[str, Any]:
        """
        批量处理高不确定性样本
        Process batch of high uncertainty samples
        
        Args:
            predictions: 预测值
            uncertainties: 不确定性
            sample_ids: 样本ID列表
            
        Returns:
            processing_results: 处理结果
        """
        # 识别高不确定性样本
        high_uncertainty_mask = uncertainties > self.uncertainty_threshold
        high_uncertainty_indices = torch.where(high_uncertainty_mask)[0]
        
        processing_results = {
            'total_samples': len(sample_ids),
            'high_uncertainty_samples': len(high_uncertainty_indices),
            'validation_triggered': [],
            'model_updates_needed': [],
            'confidence_intervals': {}
        }
        
        for idx in high_uncertainty_indices:
            sample_id = sample_ids[idx.item()]
            prediction = predictions[idx].item()
            uncertainty = uncertainties[idx].item()
            
            # 记录高不确定性样本
            sample_info = {
                'sample_id': sample_id,
                'prediction': prediction,
                'uncertainty': uncertainty,
                'timestamp': torch.tensor(0.0),  # 简化的时间戳
                'processing_status': 'identified'
            }
            
            # 决定处理策略
            if uncertainty > 2 * self.uncertainty_threshold:
                # 极高不确定性：触发DFT验证
                if self.dft_validation_enabled:
                    self._trigger_dft_validation(sample_info)
                    processing_results['validation_triggered'].append(sample_id)
                
                processing_results['model_updates_needed'].append(sample_id)
            
            # 计算置信区间
            confidence_interval = {
                'lower': prediction - 2 * uncertainty,
                'upper': prediction + 2 * uncertainty,
                'width': 4 * uncertainty
            }
            processing_results['confidence_intervals'][sample_id] = confidence_interval
            
            self.processed_samples.append(sample_info)
        
        return processing_results
    
    def _trigger_dft_validation(self, sample_info: Dict[str, Any]):
        """触发DFT验证"""
        validation_request = {
            'sample_id': sample_info['sample_id'],
            'prediction': sample_info['prediction'],
            'uncertainty': sample_info['uncertainty'],
            'priority': 'high' if sample_info['uncertainty'] > 3 * self.uncertainty_threshold else 'medium',
            'status': 'queued'
        }
        
        self.validation_queue.append(validation_request)
        self.logger.info(f"DFT validation triggered for sample {sample_info['sample_id']}")
    
    def simulate_dft_validation(self, sample_id: str) -> Dict[str, float]:
        """
        模拟DFT验证过程
        Simulate DFT validation process
        
        Args:
            sample_id: 样本ID
            
        Returns:
            dft_results: DFT验证结果
        """
        # 这里模拟DFT计算结果
        # 实际应用中应该调用真实的DFT计算接口
        
        # 从验证队列中找到对应样本
        validation_request = None
        for req in self.validation_queue:
            if req['sample_id'] == sample_id and req['status'] == 'queued':
                validation_request = req
                break
        
        if validation_request is None:
            raise ValueError(f"No validation request found for sample {sample_id}")
        
        # 模拟DFT计算时间延迟
        import time
        time.sleep(0.1)  # 模拟计算时间
        
        # 生成模拟的DFT结果
        cgcnn_prediction = validation_request['prediction']
        uncertainty = validation_request['uncertainty']
        
        # 模拟DFT结果：在不确定性范围内的随机值
        dft_value = np.random.normal(cgcnn_prediction, uncertainty * 0.5)
        
        dft_results = {
            'dft_value': dft_value,
            'cgcnn_prediction': cgcnn_prediction,
            'absolute_error': abs(dft_value - cgcnn_prediction),
            'relative_error': abs(dft_value - cgcnn_prediction) / abs(dft_value) if dft_value != 0 else float('inf'),
            'uncertainty_ratio': abs(dft_value - cgcnn_prediction) / uncertainty,
            'validation_status': 'completed'
        }
        
        # 更新验证状态
        validation_request['status'] = 'completed'
        validation_request['dft_results'] = dft_results
        
        self.logger.info(f"DFT validation completed for sample {sample_id}")
        
        return dft_results
    
    def generate_retraining_recommendations(self) -> Dict[str, Any]:
        """
        生成重训练建议
        Generate retraining recommendations
        
        Returns:
            recommendations: 重训练建议
        """
        completed_validations = [req for req in self.validation_queue 
                               if req['status'] == 'completed']
        
        if len(completed_validations) == 0:
            return {'message': 'No completed validations available'}
        
        # 分析验证结果
        large_errors = []
        prediction_biases = []
        uncertainty_calibration = []
        
        for validation in completed_validations:
            dft_results = validation['dft_results']
            
            # 大误差样本
            if dft_results['uncertainty_ratio'] > 2.0:
                large_errors.append(validation['sample_id'])
            
            # 预测偏差
            prediction_biases.append(dft_results['absolute_error'])
            
            # 不确定性校准
            uncertainty_calibration.append(dft_results['uncertainty_ratio'])
        
        recommendations = {
            'retraining_needed': len(large_errors) > len(completed_validations) * 0.3,
            'large_error_samples': large_errors,
            'avg_prediction_bias': np.mean(prediction_biases),
            'uncertainty_well_calibrated': np.mean(uncertainty_calibration) < 1.5,
            'recommended_actions': []
        }
        
        # 生成具体建议
        if recommendations['retraining_needed']:
            recommendations['recommended_actions'].append(
                '模型需要重训练：超过30%的高不确定性预测存在大误差'
            )
        
        if recommendations['avg_prediction_bias'] > 0.1:
            recommendations['recommended_actions'].append(
                '存在系统性预测偏差，建议检查训练数据质量'
            )
        
        if not recommendations['uncertainty_well_calibrated']:
            recommendations['recommended_actions'].append(
                '不确定性估计需要校准，考虑使用校准方法'
            )
        
        return recommendations


# 使用示例
def example_usage():
    """使用示例"""
    # 创建贝叶斯CGCNN模型
    model = BayesianCGCNN(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=2
    )
    
    # 创建不确定性分解器
    uncertainty_decomposer = UncertaintyDecomposer(model, n_samples=50)
    
    # 创建主动学习器
    active_learner = UncertaintyGuidedActiveLearning(
        uncertainty_decomposer, 
        acquisition_strategy='diverse_uncertainty'
    )
    
    # 创建高不确定性样本处理器
    sample_processor = HighUncertaintySampleProcessor(
        uncertainty_threshold=0.15,
        dft_validation_enabled=True
    )
    
    # 模拟数据
    batch_size = 10
    atom_fea = torch.randn(50, 92)
    nbr_fea = torch.randn(50, 12, 41)
    nbr_fea_idx = torch.randint(0, 50, (50, 12))
    crystal_atom_idx = [torch.arange(5 * i, 5 * (i + 1)) for i in range(batch_size)]
    
    input_data = (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
    
    # 不确定性分解
    uncertainty_components = uncertainty_decomposer.decompose_uncertainty(input_data)
    print("Uncertainty Decomposition:")
    print(f"Total Uncertainty: {uncertainty_components['total_uncertainty'].mean().item():.4f}")
    print(f"Epistemic Uncertainty: {uncertainty_components['epistemic_uncertainty'].mean().item():.4f}")
    print(f"Aleatoric Uncertainty: {uncertainty_components['aleatoric_uncertainty'].mean().item():.4f}")
    
    # 不确定性归因分析
    attribution = uncertainty_decomposer.uncertainty_attribution_analysis(input_data, 'formation_energy')
    print(f"\nUncertainty Attribution:")
    print(f"Dominant Source: {attribution['dominant_source']}")
    print(f"Recommendations: {attribution['recommendations']}")
    
    # 高不确定性样本处理
    sample_ids = [f"sample_{i}" for i in range(batch_size)]
    processing_results = sample_processor.process_high_uncertainty_batch(
        uncertainty_components['predictive_mean'],
        uncertainty_components['total_uncertainty'],
        sample_ids
    )
    
    print(f"\nHigh Uncertainty Processing:")
    print(f"High Uncertainty Samples: {processing_results['high_uncertainty_samples']}")
    print(f"Validation Triggered: {len(processing_results['validation_triggered'])}")


if __name__ == "__main__":
    example_usage() 