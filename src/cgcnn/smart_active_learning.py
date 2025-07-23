"""
Smart Active Learning System

Advanced active learning implementation with improved structure selection
strategies, incremental learning, and online DFT integration for materials
discovery acceleration.

Author: lunazhang
Date: 2023
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import time
import json
import os
from collections import deque, defaultdict
import threading
import queue
from dataclasses import dataclass

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from scipy.optimize import minimize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .advanced_uncertainty import UncertaintyDecomposer, UncertaintyGuidedActiveLearning


@dataclass
class StructureCandidate:
    """结构候选样本数据类"""
    id: str
    structure_data: Any
    features: np.ndarray
    predicted_property: float
    uncertainty: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    diversity_score: float
    expected_improvement: float
    acquisition_score: float
    metadata: Dict[str, Any]


@dataclass
class DFTJob:
    """DFT计算任务数据类"""
    job_id: str
    structure_id: str
    structure_data: Any
    property_type: str
    priority: str
    status: str  # 'queued', 'running', 'completed', 'failed'
    submitted_time: float
    completion_time: Optional[float]
    dft_result: Optional[float]
    error_message: Optional[str]


class AdvancedAcquisitionFunction:
    """
    高级采集函数
    Advanced Acquisition Function for Structure Selection
    """
    
    def __init__(self, strategy: str = 'multi_criteria', 
                 uncertainty_weight: float = 0.4,
                 diversity_weight: float = 0.3,
                 improvement_weight: float = 0.3):
        self.strategy = strategy
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.improvement_weight = improvement_weight
        
        self.history = []
        self.performance_tracker = defaultdict(list)
        
    def calculate_acquisition_score(self, candidates: List[StructureCandidate],
                                  current_best: float,
                                  existing_dataset: Optional[List[Any]] = None) -> np.ndarray:
        """
        计算采集分数
        Calculate acquisition scores for candidate structures
        
        Args:
            candidates: 候选结构列表
            current_best: 当前最佳值
            existing_dataset: 现有数据集
            
        Returns:
            acquisition_scores: 采集分数数组
        """
        if self.strategy == 'multi_criteria':
            return self._multi_criteria_acquisition(candidates, current_best, existing_dataset)
        elif self.strategy == 'thompson_sampling':
            return self._thompson_sampling_acquisition(candidates)
        elif self.strategy == 'expected_improvement':
            return self._expected_improvement_acquisition(candidates, current_best)
        elif self.strategy == 'upper_confidence_bound':
            return self._ucb_acquisition(candidates)
        else:
            raise ValueError(f"Unknown acquisition strategy: {self.strategy}")
    
    def _multi_criteria_acquisition(self, candidates: List[StructureCandidate],
                                  current_best: float,
                                  existing_dataset: Optional[List[Any]]) -> np.ndarray:
        """多准则采集函数"""
        n_candidates = len(candidates)
        scores = np.zeros(n_candidates)
        
        # 1. 不确定性分数
        uncertainties = np.array([c.uncertainty for c in candidates])
        uncertainty_scores = uncertainties / (np.max(uncertainties) + 1e-8)
        
        # 2. 多样性分数
        diversity_scores = self._calculate_diversity_scores(candidates, existing_dataset)
        
        # 3. 期望改进分数
        improvement_scores = self._calculate_expected_improvement(candidates, current_best)
        
        # 4. 认识不确定性权重
        epistemic_ratios = np.array([
            c.epistemic_uncertainty / (c.uncertainty + 1e-8) for c in candidates
        ])
        
        # 5. 综合评分
        for i in range(n_candidates):
            scores[i] = (
                self.uncertainty_weight * uncertainty_scores[i] +
                self.diversity_weight * diversity_scores[i] +
                self.improvement_weight * improvement_scores[i] +
                0.1 * epistemic_ratios[i]  # 偏向认识不确定性高的样本
            )
        
        return scores
    
    def _calculate_diversity_scores(self, candidates: List[StructureCandidate],
                                  existing_dataset: Optional[List[Any]]) -> np.ndarray:
        """计算多样性分数"""
        if not existing_dataset or not SKLEARN_AVAILABLE:
            return np.ones(len(candidates))
        
        # 提取候选结构的特征
        candidate_features = np.array([c.features for c in candidates])
        
        # 计算与现有数据集的相似性
        diversity_scores = np.zeros(len(candidates))
        
        for i, candidate in enumerate(candidates):
            # 与现有样本的最小距离（多样性指标）
            min_distance = float('inf')
            
            for existing_sample in existing_dataset:
                if hasattr(existing_sample, 'features'):
                    distance = np.linalg.norm(candidate.features - existing_sample.features)
                    min_distance = min(min_distance, distance)
            
            # 距离越大，多样性分数越高
            diversity_scores[i] = min_distance
        
        # 归一化
        if np.max(diversity_scores) > 0:
            diversity_scores = diversity_scores / np.max(diversity_scores)
        
        return diversity_scores
    
    def _calculate_expected_improvement(self, candidates: List[StructureCandidate],
                                     current_best: float) -> np.ndarray:
        """计算期望改进"""
        improvement_scores = np.zeros(len(candidates))
        
        for i, candidate in enumerate(candidates):
            mu = candidate.predicted_property
            sigma = candidate.uncertainty
            
            if sigma > 0:
                # 标准化改进量
                z = (mu - current_best) / sigma
                
                # 期望改进公式
                from scipy.stats import norm
                improvement = sigma * (z * norm.cdf(z) + norm.pdf(z))
                improvement_scores[i] = max(0, improvement)
            else:
                improvement_scores[i] = max(0, mu - current_best)
        
        # 归一化
        if np.max(improvement_scores) > 0:
            improvement_scores = improvement_scores / np.max(improvement_scores)
        
        return improvement_scores
    
    def _thompson_sampling_acquisition(self, candidates: List[StructureCandidate]) -> np.ndarray:
        """汤普森采样"""
        scores = np.zeros(len(candidates))
        
        for i, candidate in enumerate(candidates):
            # 从预测分布中采样
            if candidate.uncertainty > 0:
                sampled_value = np.random.normal(candidate.predicted_property, 
                                               candidate.uncertainty)
            else:
                sampled_value = candidate.predicted_property
            
            scores[i] = sampled_value
        
        return scores
    
    def _expected_improvement_acquisition(self, candidates: List[StructureCandidate],
                                        current_best: float) -> np.ndarray:
        """期望改进采集"""
        return self._calculate_expected_improvement(candidates, current_best)
    
    def _ucb_acquisition(self, candidates: List[StructureCandidate], 
                        exploration_factor: float = 2.0) -> np.ndarray:
        """上置信界采集"""
        scores = np.array([
            c.predicted_property + exploration_factor * c.uncertainty 
            for c in candidates
        ])
        return scores
    
    def update_strategy_performance(self, selected_indices: List[int],
                                  performance_improvement: float):
        """更新策略性能"""
        self.performance_tracker[self.strategy].append(performance_improvement)
        
        # 自适应策略调整
        if len(self.performance_tracker[self.strategy]) >= 5:
            recent_performance = self.performance_tracker[self.strategy][-5:]
            avg_improvement = np.mean(recent_performance)
            
            # 如果性能不佳，调整权重
            if avg_improvement < 0.01:
                self._adjust_weights()
    
    def _adjust_weights(self):
        """自适应调整权重"""
        # 简单的权重调整策略
        if self.uncertainty_weight > 0.2:
            self.uncertainty_weight -= 0.1
            self.diversity_weight += 0.05
            self.improvement_weight += 0.05


class IncrementalLearning:
    """
    增量学习模块
    Incremental Learning Module
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001,
                 memory_size: int = 1000, rehearsal_ratio: float = 0.2):
        self.model = model
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.rehearsal_ratio = rehearsal_ratio
        
        # 记忆缓冲区
        self.memory_buffer = deque(maxlen=memory_size)
        
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 性能跟踪
        self.performance_history = []
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def update_model(self, new_samples: List[Tuple], 
                    validation_data: Optional[List[Tuple]] = None) -> Dict[str, float]:
        """
        增量更新模型
        Incrementally update model with new samples
        
        Args:
            new_samples: 新的训练样本
            validation_data: 验证数据
            
        Returns:
            update_metrics: 更新指标
        """
        # 1. 将新样本添加到记忆缓冲区
        for sample in new_samples:
            self.memory_buffer.append(sample)
        
        # 2. 创建增量训练批次
        training_batch = self._create_training_batch(new_samples)
        
        # 3. 模型更新
        update_metrics = self._incremental_update(training_batch)
        
        # 4. 验证性能
        if validation_data:
            val_metrics = self._validate_performance(validation_data)
            update_metrics.update(val_metrics)
        
        # 5. 记录性能
        self.performance_history.append(update_metrics)
        
        return update_metrics
    
    def _create_training_batch(self, new_samples: List[Tuple]) -> List[Tuple]:
        """创建训练批次"""
        training_batch = list(new_samples)
        
        # 添加记忆回放样本以防止灾难性遗忘
        if len(self.memory_buffer) > 0:
            rehearsal_size = int(len(new_samples) * self.rehearsal_ratio)
            rehearsal_samples = np.random.choice(
                list(self.memory_buffer), 
                size=min(rehearsal_size, len(self.memory_buffer)), 
                replace=False
            )
            training_batch.extend(rehearsal_samples)
        
        return training_batch
    
    def _incremental_update(self, training_batch: List[Tuple]) -> Dict[str, float]:
        """执行增量更新"""
        self.model.train()
        
        total_loss = 0.0
        n_batches = 0
        
        # 简化的训练循环（实际应用中需要完整的数据加载器）
        for sample in training_batch:
            # 这里需要根据实际数据格式进行调整
            inputs, targets = sample[0], sample[1]  # 简化假设
            
            # 前向传播
            outputs = self.model(*inputs)
            loss = nn.MSELoss()(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        
        return {
            'avg_training_loss': avg_loss,
            'n_training_samples': len(training_batch),
            'n_new_samples': len(training_batch) - int(len(training_batch) * self.rehearsal_ratio)
        }
    
    def _validate_performance(self, validation_data: List[Tuple]) -> Dict[str, float]:
        """验证模型性能"""
        self.model.eval()
        
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for sample in validation_data:
                inputs, targets = sample[0], sample[1]
                outputs = self.model(*inputs)
                loss = nn.MSELoss()(outputs, targets)
                
                total_loss += loss.item()
                n_samples += 1
        
        avg_val_loss = total_loss / max(n_samples, 1)
        
        return {
            'avg_validation_loss': avg_val_loss,
            'n_validation_samples': n_samples
        }
    
    def detect_catastrophic_forgetting(self, baseline_performance: float,
                                     current_performance: float,
                                     threshold: float = 0.1) -> bool:
        """检测灾难性遗忘"""
        performance_drop = baseline_performance - current_performance
        return performance_drop > threshold
    
    def adapt_learning_rate(self, performance_trend: List[float]):
        """自适应学习率调整"""
        if len(performance_trend) >= 3:
            recent_trend = performance_trend[-3:]
            
            # 如果性能持续下降，降低学习率
            if all(recent_trend[i] > recent_trend[i+1] for i in range(len(recent_trend)-1)):
                self.learning_rate *= 0.8
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                
                self.logger.info(f"Learning rate adapted to {self.learning_rate:.6f}")


class OnlineDFTIntegration:
    """
    在线DFT集成模块
    Online DFT Integration Module
    """
    
    def __init__(self, max_concurrent_jobs: int = 5,
                 priority_threshold: float = 0.2,
                 dft_timeout: float = 3600.0):  # 1小时超时
        self.max_concurrent_jobs = max_concurrent_jobs
        self.priority_threshold = priority_threshold
        self.dft_timeout = dft_timeout
        
        # 任务队列
        self.job_queue = queue.PriorityQueue()
        self.active_jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        
        # 线程池
        self.worker_threads = []
        self.shutdown_flag = threading.Event()
        
        # 结果回调
        self.result_callbacks = []
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 启动工作线程
        self._start_workers()
    
    def submit_dft_job(self, structure_candidate: StructureCandidate,
                      property_type: str, callback: Optional[Callable] = None) -> str:
        """
        提交DFT计算任务
        Submit DFT calculation job
        
        Args:
            structure_candidate: 结构候选
            property_type: 性质类型
            callback: 结果回调函数
            
        Returns:
            job_id: 任务ID
        """
        job_id = f"dft_{int(time.time())}_{structure_candidate.id}"
        
        # 确定优先级
        priority = self._calculate_priority(structure_candidate)
        
        # 创建DFT任务
        dft_job = DFTJob(
            job_id=job_id,
            structure_id=structure_candidate.id,
            structure_data=structure_candidate.structure_data,
            property_type=property_type,
            priority=priority,
            status='queued',
            submitted_time=time.time(),
            completion_time=None,
            dft_result=None,
            error_message=None
        )
        
        # 添加到队列
        priority_score = 1.0 / (structure_candidate.uncertainty + 1e-8)
        self.job_queue.put((priority_score, time.time(), dft_job))
        
        # 注册回调
        if callback:
            self.result_callbacks.append((job_id, callback))
        
        self.logger.info(f"DFT job {job_id} submitted with priority {priority}")
        
        return job_id
    
    def _calculate_priority(self, candidate: StructureCandidate) -> str:
        """计算任务优先级"""
        if candidate.uncertainty > self.priority_threshold * 2:
            return 'high'
        elif candidate.uncertainty > self.priority_threshold:
            return 'medium'
        else:
            return 'low'
    
    def _start_workers(self):
        """启动工作线程"""
        for i in range(self.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
    
    def _worker_loop(self, worker_id: int):
        """工作线程循环"""
        self.logger.info(f"DFT worker {worker_id} started")
        
        while not self.shutdown_flag.is_set():
            try:
                # 获取任务（超时1秒）
                priority_score, submit_time, dft_job = self.job_queue.get(timeout=1.0)
                
                # 检查是否超过最大活跃任务数
                if len(self.active_jobs) >= self.max_concurrent_jobs:
                    # 重新放回队列
                    self.job_queue.put((priority_score, submit_time, dft_job))
                    time.sleep(0.1)
                    continue
                
                # 执行DFT计算
                self._execute_dft_job(dft_job, worker_id)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    def _execute_dft_job(self, dft_job: DFTJob, worker_id: int):
        """执行DFT计算任务"""
        self.logger.info(f"Worker {worker_id} executing job {dft_job.job_id}")
        
        # 更新任务状态
        dft_job.status = 'running'
        self.active_jobs[dft_job.job_id] = dft_job
        
        try:
            start_time = time.time()
            
            # 模拟DFT计算（实际应用中应调用真实的DFT计算接口）
            dft_result = self._simulate_dft_calculation(dft_job)
            
            # 检查超时
            if time.time() - start_time > self.dft_timeout:
                raise TimeoutError("DFT calculation timed out")
            
            # 更新结果
            dft_job.dft_result = dft_result
            dft_job.status = 'completed'
            dft_job.completion_time = time.time()
            
            # 移动到完成队列
            self.completed_jobs[dft_job.job_id] = dft_job
            del self.active_jobs[dft_job.job_id]
            
            # 调用回调函数
            self._trigger_callbacks(dft_job)
            
            self.logger.info(f"DFT job {dft_job.job_id} completed successfully")
            
        except Exception as e:
            # 处理错误
            dft_job.status = 'failed'
            dft_job.error_message = str(e)
            dft_job.completion_time = time.time()
            
            self.failed_jobs[dft_job.job_id] = dft_job
            if dft_job.job_id in self.active_jobs:
                del self.active_jobs[dft_job.job_id]
            
            self.logger.error(f"DFT job {dft_job.job_id} failed: {e}")
    
    def _simulate_dft_calculation(self, dft_job: DFTJob) -> float:
        """模拟DFT计算"""
        # 这里模拟DFT计算时间和结果
        # 实际应用中应该调用真实的DFT计算接口
        
        calculation_time = np.random.uniform(10, 120)  # 10秒到2分钟
        time.sleep(min(calculation_time, 5))  # 实际只等待最多5秒用于演示
        
        # 模拟计算结果
        base_value = -4.0  # 假设形成能基础值
        noise = np.random.normal(0, 0.1)
        
        return base_value + noise
    
    def _trigger_callbacks(self, dft_job: DFTJob):
        """触发结果回调"""
        for job_id, callback in self.result_callbacks:
            if job_id == dft_job.job_id:
                try:
                    callback(dft_job)
                except Exception as e:
                    self.logger.error(f"Callback error for job {job_id}: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[str]:
        """获取任务状态"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].status
        elif job_id in self.completed_jobs:
            return 'completed'
        elif job_id in self.failed_jobs:
            return 'failed'
        else:
            return None
    
    def get_job_result(self, job_id: str) -> Optional[float]:
        """获取任务结果"""
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id].dft_result
        else:
            return None
    
    def shutdown(self):
        """关闭DFT集成模块"""
        self.shutdown_flag.set()
        
        # 等待所有工作线程结束
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        self.logger.info("DFT integration module shutdown")


class SmartActiveLearningSystem:
    """
    智能主动学习系统
    Smart Active Learning System
    """
    
    def __init__(self, model: nn.Module, uncertainty_decomposer: UncertaintyDecomposer,
                 acquisition_strategy: str = 'multi_criteria',
                 enable_dft_integration: bool = True,
                 enable_incremental_learning: bool = True):
        
        self.model = model
        self.uncertainty_decomposer = uncertainty_decomposer
        
        # 组件初始化
        self.acquisition_function = AdvancedAcquisitionFunction(acquisition_strategy)
        
        if enable_incremental_learning:
            self.incremental_learner = IncrementalLearning(model)
        else:
            self.incremental_learner = None
        
        if enable_dft_integration:
            self.dft_integration = OnlineDFTIntegration()
        else:
            self.dft_integration = None
        
        # 状态跟踪
        self.iteration_count = 0
        self.performance_history = []
        self.selected_structures = []
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def run_active_learning_cycle(self, candidate_pool: List[Any],
                                 n_select: int = 10,
                                 property_type: str = 'formation_energy') -> Dict[str, Any]:
        """
        运行主动学习周期
        Run active learning cycle
        
        Args:
            candidate_pool: 候选结构池
            n_select: 选择的结构数量
            property_type: 性质类型
            
        Returns:
            cycle_results: 周期结果
        """
        self.iteration_count += 1
        self.logger.info(f"Starting active learning cycle {self.iteration_count}")
        
        # 1. 预测和不确定性量化
        structure_candidates = self._evaluate_candidates(candidate_pool, property_type)
        
        # 2. 计算采集分数
        current_best = self._get_current_best_value(property_type)
        acquisition_scores = self.acquisition_function.calculate_acquisition_score(
            structure_candidates, current_best, self.selected_structures
        )
        
        # 3. 选择结构
        selected_indices = np.argsort(acquisition_scores)[-n_select:]
        selected_candidates = [structure_candidates[i] for i in selected_indices]
        
        # 4. 提交DFT计算
        dft_jobs = []
        if self.dft_integration:
            for candidate in selected_candidates:
                job_id = self.dft_integration.submit_dft_job(
                    candidate, property_type, 
                    callback=self._dft_result_callback
                )
                dft_jobs.append(job_id)
        
        # 5. 等待部分DFT结果（非阻塞）
        initial_results = self._collect_initial_dft_results(dft_jobs, timeout=30)
        
        # 6. 增量学习更新
        if self.incremental_learner and initial_results:
            update_metrics = self.incremental_learner.update_model(initial_results)
        else:
            update_metrics = {}
        
        # 7. 记录选择的结构
        self.selected_structures.extend(selected_candidates)
        
        # 8. 生成周期报告
        cycle_results = {
            'iteration': self.iteration_count,
            'n_candidates_evaluated': len(structure_candidates),
            'n_structures_selected': len(selected_candidates),
            'selected_structures': [c.id for c in selected_candidates],
            'dft_jobs_submitted': dft_jobs,
            'initial_dft_results': len(initial_results),
            'model_update_metrics': update_metrics,
            'acquisition_strategy': self.acquisition_function.strategy,
            'performance_improvement': self._calculate_performance_improvement()
        }
        
        self.performance_history.append(cycle_results)
        
        return cycle_results
    
    def _evaluate_candidates(self, candidate_pool: List[Any], 
                           property_type: str) -> List[StructureCandidate]:
        """评估候选结构"""
        structure_candidates = []
        
        for i, structure_data in enumerate(candidate_pool):
            # 预测性质
            prediction_result = self._predict_structure_property(structure_data, property_type)
            
            # 不确定性分解
            uncertainty_components = self.uncertainty_decomposer.decompose_uncertainty(
                structure_data
            )
            
            # 提取特征用于多样性计算
            features = self._extract_structure_features(structure_data)
            
            # 创建候选结构对象
            candidate = StructureCandidate(
                id=f"candidate_{self.iteration_count}_{i}",
                structure_data=structure_data,
                features=features,
                predicted_property=prediction_result['prediction'],
                uncertainty=uncertainty_components['total_uncertainty'].item(),
                epistemic_uncertainty=uncertainty_components['epistemic_uncertainty'].item(),
                aleatoric_uncertainty=uncertainty_components['aleatoric_uncertainty'].item(),
                diversity_score=0.0,  # 将在采集函数中计算
                expected_improvement=0.0,  # 将在采集函数中计算
                acquisition_score=0.0,  # 将在采集函数中计算
                metadata={
                    'property_type': property_type,
                    'evaluation_time': time.time()
                }
            )
            
            structure_candidates.append(candidate)
        
        return structure_candidates
    
    def _predict_structure_property(self, structure_data: Any, 
                                  property_type: str) -> Dict[str, float]:
        """预测结构性质"""
        # 这里需要根据实际的数据格式进行调整
        self.model.eval()
        
        with torch.no_grad():
            prediction = self.model(*structure_data)
        
        return {
            'prediction': prediction.item(),
            'confidence': 0.8  # 简化的置信度
        }
    
    def _extract_structure_features(self, structure_data: Any) -> np.ndarray:
        """提取结构特征"""
        # 简化的特征提取
        # 实际应用中应该提取更丰富的结构描述符
        
        atom_fea, nbr_fea, _, _ = structure_data
        
        # 使用原子特征的统计量作为结构特征
        atom_mean = torch.mean(atom_fea, dim=0).cpu().numpy()
        atom_std = torch.std(atom_fea, dim=0).cpu().numpy()
        
        # 邻居特征统计
        nbr_mean = torch.mean(nbr_fea, dim=[0, 1]).cpu().numpy()
        nbr_std = torch.std(nbr_fea, dim=[0, 1]).cpu().numpy()
        
        # 连接所有特征
        features = np.concatenate([atom_mean, atom_std, nbr_mean, nbr_std])
        
        return features
    
    def _get_current_best_value(self, property_type: str) -> float:
        """获取当前最佳值"""
        # 简化实现：返回历史性能的最佳值
        if self.performance_history:
            # 这里应该根据具体的性质类型和目标（最大化或最小化）来确定
            return -4.0  # 假设形成能的最佳值
        else:
            return -3.5  # 初始估计
    
    def _collect_initial_dft_results(self, dft_jobs: List[str], 
                                   timeout: float = 30) -> List[Tuple]:
        """收集初始DFT结果"""
        if not self.dft_integration:
            return []
        
        initial_results = []
        start_time = time.time()
        
        while time.time() - start_time < timeout and dft_jobs:
            completed_jobs = []
            
            for job_id in dft_jobs:
                status = self.dft_integration.get_job_status(job_id)
                if status == 'completed':
                    result = self.dft_integration.get_job_result(job_id)
                    if result is not None:
                        # 这里需要构造训练样本格式
                        # initial_results.append((structure_data, result))
                        pass
                    completed_jobs.append(job_id)
                elif status == 'failed':
                    completed_jobs.append(job_id)
            
            # 移除已完成的任务
            for job_id in completed_jobs:
                dft_jobs.remove(job_id)
            
            if not dft_jobs:
                break
            
            time.sleep(1.0)
        
        return initial_results
    
    def _dft_result_callback(self, dft_job: DFTJob):
        """DFT结果回调函数"""
        self.logger.info(f"DFT result received for job {dft_job.job_id}")
        
        if dft_job.status == 'completed' and self.incremental_learner:
            # 异步更新模型
            # 这里可以实现更复杂的增量学习策略
            pass
    
    def _calculate_performance_improvement(self) -> float:
        """计算性能改进"""
        if len(self.performance_history) < 2:
            return 0.0
        
        # 简化的性能改进计算
        # 实际应用中应该基于验证集性能或其他指标
        current_best = self._get_current_best_value('formation_energy')
        previous_best = -3.8  # 简化假设
        
        return current_best - previous_best
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        stats = {
            'total_iterations': self.iteration_count,
            'total_structures_selected': len(self.selected_structures),
            'avg_uncertainty_reduction': 0.0,
            'model_improvement_trend': [],
            'acquisition_strategy_performance': {}
        }
        
        if self.performance_history:
            improvements = [h.get('performance_improvement', 0) for h in self.performance_history]
            stats['model_improvement_trend'] = improvements
            stats['avg_performance_improvement'] = np.mean(improvements)
        
        return stats
    
    def shutdown(self):
        """关闭主动学习系统"""
        if self.dft_integration:
            self.dft_integration.shutdown()
        
        self.logger.info("Smart active learning system shutdown")


# 使用示例
def example_usage():
    """使用示例"""
    # 创建模型和不确定性分解器
    from .enhanced_model import EnhancedCGCNN
    from .advanced_uncertainty import UncertaintyDecomposer, BayesianCGCNN
    
    model = EnhancedCGCNN(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128
    )
    
    bayesian_model = BayesianCGCNN(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128
    )
    
    uncertainty_decomposer = UncertaintyDecomposer(bayesian_model)
    
    # 创建智能主动学习系统
    smart_al_system = SmartActiveLearningSystem(
        model=model,
        uncertainty_decomposer=uncertainty_decomposer,
        acquisition_strategy='multi_criteria',
        enable_dft_integration=True,
        enable_incremental_learning=True
    )
    
    # 模拟候选结构池
    candidate_pool = []
    for i in range(50):
        atom_fea = torch.randn(20, 92)
        nbr_fea = torch.randn(20, 12, 41)
        nbr_fea_idx = torch.randint(0, 20, (20, 12))
        crystal_atom_idx = [torch.arange(4 * j, 4 * (j + 1)) for j in range(5)]
        
        candidate_pool.append((atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx))
    
    # 运行主动学习周期
    cycle_results = smart_al_system.run_active_learning_cycle(
        candidate_pool, n_select=5, property_type='formation_energy'
    )
    
    print("Active Learning Cycle Results:")
    print(f"Iteration: {cycle_results['iteration']}")
    print(f"Candidates evaluated: {cycle_results['n_candidates_evaluated']}")
    print(f"Structures selected: {cycle_results['n_structures_selected']}")
    print(f"DFT jobs submitted: {len(cycle_results['dft_jobs_submitted'])}")
    
    # 获取学习统计
    stats = smart_al_system.get_learning_statistics()
    print(f"\nLearning Statistics:")
    print(f"Total iterations: {stats['total_iterations']}")
    print(f"Total structures selected: {stats['total_structures_selected']}")
    
    # 关闭系统
    smart_al_system.shutdown()


if __name__ == "__main__":
    example_usage() 