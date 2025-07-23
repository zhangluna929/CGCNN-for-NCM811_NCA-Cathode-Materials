"""
Robust Training Module

Comprehensive robustness enhancement through physics constraints,
adversarial training, and ensemble methods for materials property
prediction with noise robustness and domain transfer capabilities.

Author: lunazhang
Date: 2023
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import copy
from collections import defaultdict

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .physics import PhysicsConstrainedLoss
from .enhanced_model import EnhancedCGCNN


class AdversarialTrainer:
    """
    对抗训练器
    Adversarial Trainer for Robustness Enhancement
    """
    
    def __init__(self, model: nn.Module, epsilon: float = 0.1, 
                 alpha: float = 0.01, num_steps: int = 10,
                 attack_type: str = 'fgsm'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.attack_type = attack_type
        
        self.logger = logging.getLogger(__name__)
    
    def generate_adversarial_examples(self, input_data: Tuple, 
                                    targets: torch.Tensor,
                                    criterion: nn.Module) -> Tuple:
        """
        生成对抗样本
        Generate adversarial examples
        
        Args:
            input_data: 输入数据
            targets: 目标标签
            criterion: 损失函数
            
        Returns:
            adversarial_data: 对抗样本
        """
        if self.attack_type == 'fgsm':
            return self._fgsm_attack(input_data, targets, criterion)
        elif self.attack_type == 'pgd':
            return self._pgd_attack(input_data, targets, criterion)
        elif self.attack_type == 'materials_aware':
            return self._materials_aware_attack(input_data, targets, criterion)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
    
    def _fgsm_attack(self, input_data: Tuple, targets: torch.Tensor,
                    criterion: nn.Module) -> Tuple:
        """快速梯度符号方法攻击"""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 复制数据并设置requires_grad
        adv_atom_fea = atom_fea.clone().detach().requires_grad_(True)
        adv_nbr_fea = nbr_fea.clone().detach().requires_grad_(True)
        
        # 前向传播
        outputs = self.model(adv_atom_fea, adv_nbr_fea, nbr_fea_idx, crystal_atom_idx)
        loss = criterion(outputs, targets)
        
        # 反向传播获取梯度
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗扰动
        atom_grad = adv_atom_fea.grad.data
        nbr_grad = adv_nbr_fea.grad.data
        
        # FGSM扰动
        atom_perturbation = self.epsilon * atom_grad.sign()
        nbr_perturbation = self.epsilon * nbr_grad.sign()
        
        # 应用扰动
        adv_atom_fea = atom_fea + atom_perturbation
        adv_nbr_fea = nbr_fea + nbr_perturbation
        
        # 应用物理约束
        adv_atom_fea, adv_nbr_fea = self._apply_physical_constraints(
            adv_atom_fea, adv_nbr_fea, atom_fea, nbr_fea
        )
        
        return (adv_atom_fea.detach(), adv_nbr_fea.detach(), 
                nbr_fea_idx, crystal_atom_idx)
    
    def _pgd_attack(self, input_data: Tuple, targets: torch.Tensor,
                   criterion: nn.Module) -> Tuple:
        """投影梯度下降攻击"""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 初始化对抗样本
        adv_atom_fea = atom_fea.clone().detach()
        adv_nbr_fea = nbr_fea.clone().detach()
        
        # 添加随机初始扰动
        adv_atom_fea += torch.empty_like(adv_atom_fea).uniform_(-self.epsilon, self.epsilon)
        adv_nbr_fea += torch.empty_like(adv_nbr_fea).uniform_(-self.epsilon, self.epsilon)
        
        for step in range(self.num_steps):
            adv_atom_fea.requires_grad_(True)
            adv_nbr_fea.requires_grad_(True)
            
            # 前向传播
            outputs = self.model(adv_atom_fea, adv_nbr_fea, nbr_fea_idx, crystal_atom_idx)
            loss = criterion(outputs, targets)
            
            # 计算梯度
            self.model.zero_grad()
            loss.backward()
            
            atom_grad = adv_atom_fea.grad.data
            nbr_grad = adv_nbr_fea.grad.data
            
            # PGD更新
            adv_atom_fea = adv_atom_fea.detach() + self.alpha * atom_grad.sign()
            adv_nbr_fea = adv_nbr_fea.detach() + self.alpha * nbr_grad.sign()
            
            # 投影到epsilon球内
            atom_delta = torch.clamp(adv_atom_fea - atom_fea, -self.epsilon, self.epsilon)
            nbr_delta = torch.clamp(adv_nbr_fea - nbr_fea, -self.epsilon, self.epsilon)
            
            adv_atom_fea = atom_fea + atom_delta
            adv_nbr_fea = nbr_fea + nbr_delta
            
            # 应用物理约束
            adv_atom_fea, adv_nbr_fea = self._apply_physical_constraints(
                adv_atom_fea, adv_nbr_fea, atom_fea, nbr_fea
            )
        
        return (adv_atom_fea.detach(), adv_nbr_fea.detach(), 
                nbr_fea_idx, crystal_atom_idx)
    
    def _materials_aware_attack(self, input_data: Tuple, targets: torch.Tensor,
                              criterion: nn.Module) -> Tuple:
        """材料感知对抗攻击"""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 识别重要特征维度
        important_atom_dims = self._identify_important_atom_features(input_data)
        important_nbr_dims = self._identify_important_neighbor_features(input_data)
        
        adv_atom_fea = atom_fea.clone().detach().requires_grad_(True)
        adv_nbr_fea = nbr_fea.clone().detach().requires_grad_(True)
        
        # 前向传播
        outputs = self.model(adv_atom_fea, adv_nbr_fea, nbr_fea_idx, crystal_atom_idx)
        loss = criterion(outputs, targets)
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        atom_grad = adv_atom_fea.grad.data
        nbr_grad = adv_nbr_fea.grad.data
        
        # 创建选择性扰动
        atom_perturbation = torch.zeros_like(atom_fea)
        nbr_perturbation = torch.zeros_like(nbr_fea)
        
        # 只对重要特征添加扰动
        atom_perturbation[:, important_atom_dims] = \
            self.epsilon * atom_grad[:, important_atom_dims].sign()
        nbr_perturbation[:, :, important_nbr_dims] = \
            self.epsilon * nbr_grad[:, :, important_nbr_dims].sign()
        
        adv_atom_fea = atom_fea + atom_perturbation
        adv_nbr_fea = nbr_fea + nbr_perturbation
        
        # 应用材料特定约束
        adv_atom_fea, adv_nbr_fea = self._apply_materials_specific_constraints(
            adv_atom_fea, adv_nbr_fea, atom_fea, nbr_fea
        )
        
        return (adv_atom_fea.detach(), adv_nbr_fea.detach(), 
                nbr_fea_idx, crystal_atom_idx)
    
    def _identify_important_atom_features(self, input_data: Tuple) -> List[int]:
        """识别重要的原子特征维度"""
        # 简化实现：基于特征方差选择重要维度
        atom_fea = input_data[0]
        feature_variance = torch.var(atom_fea, dim=0)
        
        # 选择方差最大的前50%特征
        n_important = atom_fea.size(1) // 2
        _, important_indices = torch.topk(feature_variance, n_important)
        
        return important_indices.tolist()
    
    def _identify_important_neighbor_features(self, input_data: Tuple) -> List[int]:
        """识别重要的邻居特征维度"""
        nbr_fea = input_data[1]
        feature_variance = torch.var(nbr_fea.view(-1, nbr_fea.size(-1)), dim=0)
        
        n_important = nbr_fea.size(-1) // 2
        _, important_indices = torch.topk(feature_variance, n_important)
        
        return important_indices.tolist()
    
    def _apply_physical_constraints(self, adv_atom_fea: torch.Tensor, 
                                  adv_nbr_fea: torch.Tensor,
                                  orig_atom_fea: torch.Tensor,
                                  orig_nbr_fea: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用物理约束"""
        # 限制扰动幅度以保持物理合理性
        max_atom_change = 0.2  # 最大20%变化
        max_nbr_change = 0.15   # 最大15%变化
        
        # 原子特征约束
        atom_change_ratio = torch.abs(adv_atom_fea - orig_atom_fea) / (torch.abs(orig_atom_fea) + 1e-8)
        atom_mask = atom_change_ratio > max_atom_change
        adv_atom_fea[atom_mask] = orig_atom_fea[atom_mask] + \
                                 max_atom_change * torch.sign(adv_atom_fea[atom_mask] - orig_atom_fea[atom_mask]) * torch.abs(orig_atom_fea[atom_mask])
        
        # 邻居特征约束
        nbr_change_ratio = torch.abs(adv_nbr_fea - orig_nbr_fea) / (torch.abs(orig_nbr_fea) + 1e-8)
        nbr_mask = nbr_change_ratio > max_nbr_change
        adv_nbr_fea[nbr_mask] = orig_nbr_fea[nbr_mask] + \
                               max_nbr_change * torch.sign(adv_nbr_fea[nbr_mask] - orig_nbr_fea[nbr_mask]) * torch.abs(orig_nbr_fea[nbr_mask])
        
        return adv_atom_fea, adv_nbr_fea
    
    def _apply_materials_specific_constraints(self, adv_atom_fea: torch.Tensor,
                                            adv_nbr_fea: torch.Tensor,
                                            orig_atom_fea: torch.Tensor,
                                            orig_nbr_fea: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用材料特定约束"""
        # 确保某些特征保持非负（如原子电荷、键长等）
        # 这里假设前10个原子特征应该非负
        adv_atom_fea[:, :10] = torch.clamp(adv_atom_fea[:, :10], min=0)
        
        # 确保邻居特征的物理合理性（如距离特征）
        # 假设邻居特征的前5维是距离相关特征
        adv_nbr_fea[:, :, :5] = torch.clamp(adv_nbr_fea[:, :, :5], min=0.5, max=10.0)
        
        return adv_atom_fea, adv_nbr_fea


class NoiseRobustnessTrainer:
    """
    噪声鲁棒性训练器
    Noise Robustness Trainer
    """
    
    def __init__(self, noise_types: List[str] = ['gaussian', 'uniform', 'feature_dropout'],
                 noise_levels: List[float] = [0.01, 0.05, 0.1]):
        self.noise_types = noise_types
        self.noise_levels = noise_levels
        
        self.logger = logging.getLogger(__name__)
    
    def add_training_noise(self, input_data: Tuple, 
                          noise_type: str = 'gaussian',
                          noise_level: float = 0.05) -> Tuple:
        """
        添加训练噪声
        Add training noise to input data
        
        Args:
            input_data: 输入数据
            noise_type: 噪声类型
            noise_level: 噪声水平
            
        Returns:
            noisy_data: 添加噪声后的数据
        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        if noise_type == 'gaussian':
            noisy_atom_fea = atom_fea + torch.randn_like(atom_fea) * noise_level
            noisy_nbr_fea = nbr_fea + torch.randn_like(nbr_fea) * noise_level
        
        elif noise_type == 'uniform':
            noisy_atom_fea = atom_fea + torch.empty_like(atom_fea).uniform_(-noise_level, noise_level)
            noisy_nbr_fea = nbr_fea + torch.empty_like(nbr_fea).uniform_(-noise_level, noise_level)
        
        elif noise_type == 'feature_dropout':
            # 随机将部分特征置零
            atom_mask = torch.bernoulli(torch.full_like(atom_fea, 1 - noise_level))
            nbr_mask = torch.bernoulli(torch.full_like(nbr_fea, 1 - noise_level))
            
            noisy_atom_fea = atom_fea * atom_mask
            noisy_nbr_fea = nbr_fea * nbr_mask
        
        elif noise_type == 'experimental_error':
            # 模拟实验误差：不同特征有不同的误差分布
            noisy_atom_fea = self._add_experimental_noise(atom_fea, noise_level)
            noisy_nbr_fea = self._add_experimental_noise(nbr_fea, noise_level)
        
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return (noisy_atom_fea, noisy_nbr_fea, nbr_fea_idx, crystal_atom_idx)
    
    def _add_experimental_noise(self, features: torch.Tensor, 
                              base_noise_level: float) -> torch.Tensor:
        """添加实验误差噪声"""
        noisy_features = features.clone()
        
        # 不同类型的特征有不同的噪声水平
        n_features = features.size(-1)
        
        # 假设前1/3的特征是高精度测量（低噪声）
        high_precision_end = n_features // 3
        noisy_features[..., :high_precision_end] += \
            torch.randn_like(features[..., :high_precision_end]) * base_noise_level * 0.5
        
        # 中间1/3是中等精度（中等噪声）
        medium_precision_start = high_precision_end
        medium_precision_end = 2 * n_features // 3
        noisy_features[..., medium_precision_start:medium_precision_end] += \
            torch.randn_like(features[..., medium_precision_start:medium_precision_end]) * base_noise_level
        
        # 最后1/3是低精度测量（高噪声）
        low_precision_start = medium_precision_end
        noisy_features[..., low_precision_start:] += \
            torch.randn_like(features[..., low_precision_start:]) * base_noise_level * 2.0
        
        return noisy_features
    
    def create_noise_schedule(self, n_epochs: int) -> List[Tuple[str, float]]:
        """创建噪声调度"""
        schedule = []
        
        for epoch in range(n_epochs):
            # 噪声水平随训练进程递减
            progress = epoch / n_epochs
            base_noise = 0.1 * (1 - progress * 0.8)  # 从0.1递减到0.02
            
            # 随机选择噪声类型
            noise_type = np.random.choice(self.noise_types)
            
            schedule.append((noise_type, base_noise))
        
        return schedule


class EnsembleTrainer:
    """
    集成训练器
    Ensemble Trainer for Robustness
    """
    
    def __init__(self, n_models: int = 5, diversity_method: str = 'bagging'):
        self.n_models = n_models
        self.diversity_method = diversity_method
        self.models = []
        
        self.logger = logging.getLogger(__name__)
    
    def create_diverse_models(self, base_model_config: Dict[str, Any]) -> List[nn.Module]:
        """
        创建多样化的模型集合
        Create diverse model ensemble
        
        Args:
            base_model_config: 基础模型配置
            
        Returns:
            diverse_models: 多样化模型列表
        """
        diverse_models = []
        
        for i in range(self.n_models):
            if self.diversity_method == 'architecture':
                # 架构多样性
                model_config = self._modify_architecture(base_model_config, i)
            elif self.diversity_method == 'initialization':
                # 初始化多样性
                model_config = base_model_config.copy()
            elif self.diversity_method == 'hyperparameter':
                # 超参数多样性
                model_config = self._modify_hyperparameters(base_model_config, i)
            else:  # bagging
                model_config = base_model_config.copy()
            
            model = EnhancedCGCNN(**model_config)
            
            # 不同的初始化策略
            if self.diversity_method == 'initialization':
                self._apply_diverse_initialization(model, i)
            
            diverse_models.append(model)
        
        self.models = diverse_models
        return diverse_models
    
    def _modify_architecture(self, base_config: Dict[str, Any], model_id: int) -> Dict[str, Any]:
        """修改模型架构以增加多样性"""
        config = base_config.copy()
        
        # 随机调整架构参数
        variations = [
            {'n_conv': 2, 'h_fea_len': 96},
            {'n_conv': 3, 'h_fea_len': 128},
            {'n_conv': 4, 'h_fea_len': 160},
            {'n_conv': 3, 'h_fea_len': 96, 'n_h': 2},
            {'n_conv': 3, 'h_fea_len': 128, 'attention_heads': 4}
        ]
        
        variation = variations[model_id % len(variations)]
        config.update(variation)
        
        return config
    
    def _modify_hyperparameters(self, base_config: Dict[str, Any], model_id: int) -> Dict[str, Any]:
        """修改超参数以增加多样性"""
        config = base_config.copy()
        
        # 不同的dropout率
        dropout_rates = [0.05, 0.1, 0.15, 0.2, 0.25]
        # 注意：这里需要在模型中添加dropout参数支持
        
        return config
    
    def _apply_diverse_initialization(self, model: nn.Module, model_id: int):
        """应用多样化初始化"""
        init_methods = ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal']
        init_method = init_methods[model_id % len(init_methods)]
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if init_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif init_method == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif init_method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight)
                elif init_method == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight)
                elif init_method == 'orthogonal':
                    nn.init.orthogonal_(module.weight)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def train_ensemble(self, train_loader, val_loader, 
                      n_epochs: int = 50, device: str = 'cpu') -> Dict[str, Any]:
        """
        训练集成模型
        Train ensemble models
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            n_epochs: 训练轮数
            device: 设备
            
        Returns:
            training_results: 训练结果
        """
        training_results = {
            'model_performances': [],
            'ensemble_performance': {},
            'diversity_metrics': {}
        }
        
        # 训练每个模型
        for i, model in enumerate(self.models):
            self.logger.info(f"Training ensemble model {i+1}/{len(self.models)}")
            
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # 简化的训练循环
            model_performance = self._train_single_model(
                model, train_loader, val_loader, optimizer, criterion, n_epochs, device
            )
            
            training_results['model_performances'].append(model_performance)
        
        # 评估集成性能
        ensemble_perf = self._evaluate_ensemble(val_loader, device)
        training_results['ensemble_performance'] = ensemble_perf
        
        # 计算多样性指标
        diversity_metrics = self._calculate_diversity_metrics(val_loader, device)
        training_results['diversity_metrics'] = diversity_metrics
        
        return training_results
    
    def _train_single_model(self, model: nn.Module, train_loader, val_loader,
                           optimizer, criterion, n_epochs: int, device: str) -> Dict[str, float]:
        """训练单个模型"""
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_idx, (inputs, targets, _) in enumerate(train_loader):
                # 移动数据到设备
                inputs = [inp.to(device) if torch.is_tensor(inp) 
                         else [t.to(device) for t in inp] for inp in inputs]
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(*inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            if epoch % 10 == 0:
                val_loss = self._evaluate_single_model(model, val_loader, criterion, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
        
        return {'best_val_loss': best_val_loss}
    
    def _evaluate_single_model(self, model: nn.Module, val_loader, 
                              criterion, device: str) -> float:
        """评估单个模型"""
        model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = [inp.to(device) if torch.is_tensor(inp) 
                         else [t.to(device) for t in inp] for inp in inputs]
                targets = targets.to(device)
                
                outputs = model(*inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _evaluate_ensemble(self, val_loader, device: str) -> Dict[str, float]:
        """评估集成性能"""
        all_predictions = []
        all_targets = []
        
        # 收集所有模型的预测
        for model in self.models:
            model.eval()
            model_predictions = []
            
            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    inputs = [inp.to(device) if torch.is_tensor(inp) 
                             else [t.to(device) for t in inp] for inp in inputs]
                    
                    outputs = model(*inputs)
                    model_predictions.append(outputs.cpu())
                    
                    if len(all_targets) == 0:
                        all_targets.extend(targets.cpu().numpy())
            
            all_predictions.append(torch.cat(model_predictions, dim=0).numpy())
        
        # 计算集成预测
        ensemble_predictions = np.mean(all_predictions, axis=0)
        
        # 计算性能指标
        mse = np.mean((ensemble_predictions.flatten() - np.array(all_targets).flatten()) ** 2)
        mae = np.mean(np.abs(ensemble_predictions.flatten() - np.array(all_targets).flatten()))
        
        return {'ensemble_mse': mse, 'ensemble_mae': mae}
    
    def _calculate_diversity_metrics(self, val_loader, device: str) -> Dict[str, float]:
        """计算集成多样性指标"""
        all_predictions = []
        
        # 收集预测
        for model in self.models:
            model.eval()
            model_predictions = []
            
            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    inputs = [inp.to(device) if torch.is_tensor(inp) 
                             else [t.to(device) for t in inp] for inp in inputs]
                    
                    outputs = model(*inputs)
                    model_predictions.append(outputs.cpu())
            
            all_predictions.append(torch.cat(model_predictions, dim=0).numpy())
        
        predictions_array = np.array(all_predictions)  # [n_models, n_samples, output_dim]
        
        # 计算多样性指标
        # 1. 预测方差（多样性指标）
        prediction_variance = np.var(predictions_array, axis=0).mean()
        
        # 2. 平均成对差异
        pairwise_diffs = []
        for i in range(len(all_predictions)):
            for j in range(i+1, len(all_predictions)):
                diff = np.mean((all_predictions[i] - all_predictions[j]) ** 2)
                pairwise_diffs.append(diff)
        
        avg_pairwise_diff = np.mean(pairwise_diffs)
        
        return {
            'prediction_variance': prediction_variance,
            'avg_pairwise_difference': avg_pairwise_diff
        }


class DomainAdaptationTrainer:
    """
    域适应训练器
    Domain Adaptation Trainer
    """
    
    def __init__(self, adaptation_method: str = 'dann'):
        self.adaptation_method = adaptation_method
        
        self.logger = logging.getLogger(__name__)
    
    def adapt_to_new_conditions(self, source_model: nn.Module,
                               source_data: Any, target_data: Any,
                               adaptation_epochs: int = 20) -> nn.Module:
        """
        适应新的实验条件
        Adapt model to new experimental conditions
        
        Args:
            source_model: 源域模型
            source_data: 源域数据
            target_data: 目标域数据
            adaptation_epochs: 适应训练轮数
            
        Returns:
            adapted_model: 适应后的模型
        """
        if self.adaptation_method == 'fine_tuning':
            return self._fine_tuning_adaptation(source_model, target_data, adaptation_epochs)
        elif self.adaptation_method == 'dann':
            return self._dann_adaptation(source_model, source_data, target_data, adaptation_epochs)
        elif self.adaptation_method == 'coral':
            return self._coral_adaptation(source_model, source_data, target_data, adaptation_epochs)
        else:
            raise ValueError(f"Unknown adaptation method: {self.adaptation_method}")
    
    def _fine_tuning_adaptation(self, source_model: nn.Module, 
                               target_data: Any, epochs: int) -> nn.Module:
        """微调适应"""
        adapted_model = copy.deepcopy(source_model)
        
        # 冻结早期层，只微调后面的层
        for name, param in adapted_model.named_parameters():
            if 'embedding' in name or 'conv' in name:
                param.requires_grad = False
        
        # 使用较小的学习率进行微调
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, adapted_model.parameters()),
            lr=0.0001
        )
        criterion = nn.MSELoss()
        
        # 简化的微调训练循环
        adapted_model.train()
        for epoch in range(epochs):
            # 这里需要实际的目标域数据加载器
            # 简化实现
            pass
        
        return adapted_model
    
    def _dann_adaptation(self, source_model: nn.Module, 
                        source_data: Any, target_data: Any, epochs: int) -> nn.Module:
        """域对抗神经网络适应"""
        # DANN需要域分类器
        class DomainClassifier(nn.Module):
            def __init__(self, feature_dim: int):
                super().__init__()
                self.classifier = nn.Sequential(
                    nn.Linear(feature_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2)  # 二分类：源域 vs 目标域
                )
            
            def forward(self, x):
                return self.classifier(x)
        
        # 创建域分类器
        domain_classifier = DomainClassifier(128)  # 假设特征维度为128
        
        # 梯度反转层
        class GradientReversalLayer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lambda_param):
                ctx.lambda_param = lambda_param
                return x
            
            @staticmethod
            def backward(ctx, grad_output):
                return -ctx.lambda_param * grad_output, None
        
        # DANN训练逻辑（简化）
        adapted_model = copy.deepcopy(source_model)
        
        return adapted_model
    
    def _coral_adaptation(self, source_model: nn.Module,
                         source_data: Any, target_data: Any, epochs: int) -> nn.Module:
        """CORAL域适应"""
        def coral_loss(source_features: torch.Tensor, 
                      target_features: torch.Tensor) -> torch.Tensor:
            """计算CORAL损失"""
            # 计算协方差矩阵
            source_cov = torch.cov(source_features.T)
            target_cov = torch.cov(target_features.T)
            
            # Frobenius范数
            coral_loss = torch.norm(source_cov - target_cov, p='fro') ** 2
            return coral_loss / (4 * source_features.size(1) ** 2)
        
        adapted_model = copy.deepcopy(source_model)
        
        # CORAL适应训练（简化）
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            # 提取源域和目标域特征
            # 计算CORAL损失
            # 反向传播更新
            pass
        
        return adapted_model


class RobustTrainingOrchestrator:
    """
    鲁棒训练协调器
    Robust Training Orchestrator
    """
    
    def __init__(self, model: nn.Module, 
                 enable_adversarial: bool = True,
                 enable_noise_training: bool = True,
                 enable_ensemble: bool = True,
                 enable_physics_constraints: bool = True):
        
        self.model = model
        self.enable_adversarial = enable_adversarial
        self.enable_noise_training = enable_noise_training
        self.enable_ensemble = enable_ensemble
        self.enable_physics_constraints = enable_physics_constraints
        
        # 初始化组件
        if enable_adversarial:
            self.adversarial_trainer = AdversarialTrainer(model)
        
        if enable_noise_training:
            self.noise_trainer = NoiseRobustnessTrainer()
        
        if enable_ensemble:
            self.ensemble_trainer = EnsembleTrainer()
        
        if enable_physics_constraints:
            self.physics_loss = PhysicsConstrainedLoss()
        
        self.logger = logging.getLogger(__name__)
    
    def robust_training_epoch(self, train_loader, optimizer, criterion,
                             epoch: int, device: str = 'cpu') -> Dict[str, float]:
        """
        鲁棒训练单个epoch
        Robust training for single epoch
        
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            epoch: 当前epoch
            device: 设备
            
        Returns:
            epoch_metrics: epoch指标
        """
        self.model.train()
        
        total_loss = 0.0
        adversarial_loss = 0.0
        physics_loss = 0.0
        n_batches = 0
        
        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            # 移动数据到设备
            inputs = [inp.to(device) if torch.is_tensor(inp) 
                     else [t.to(device) for t in inp] for inp in inputs]
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # 1. 标准训练
            outputs = self.model(*inputs)
            base_loss = criterion(outputs, targets)
            
            total_batch_loss = base_loss
            
            # 2. 对抗训练
            if self.enable_adversarial and epoch >= 10:  # 从第10个epoch开始对抗训练
                adv_inputs = self.adversarial_trainer.generate_adversarial_examples(
                    inputs, targets, criterion
                )
                adv_outputs = self.model(*adv_inputs)
                adv_loss = criterion(adv_outputs, targets)
                
                total_batch_loss += 0.3 * adv_loss  # 对抗损失权重
                adversarial_loss += adv_loss.item()
            
            # 3. 噪声训练
            if self.enable_noise_training:
                noise_type = np.random.choice(['gaussian', 'uniform', 'feature_dropout'])
                noise_level = np.random.uniform(0.01, 0.05)
                
                noisy_inputs = self.noise_trainer.add_training_noise(
                    inputs, noise_type, noise_level
                )
                noisy_outputs = self.model(*noisy_inputs)
                noise_loss = criterion(noisy_outputs, targets)
                
                total_batch_loss += 0.2 * noise_loss  # 噪声损失权重
            
            # 4. 物理约束
            if self.enable_physics_constraints:
                phys_loss, _ = self.physics_loss(outputs, targets, self.model, inputs)
                total_batch_loss += phys_loss
                physics_loss += phys_loss.item()
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            n_batches += 1
        
        epoch_metrics = {
            'avg_total_loss': total_loss / max(n_batches, 1),
            'avg_adversarial_loss': adversarial_loss / max(n_batches, 1),
            'avg_physics_loss': physics_loss / max(n_batches, 1)
        }
        
        return epoch_metrics
    
    def evaluate_robustness(self, test_loader, device: str = 'cpu') -> Dict[str, Any]:
        """
        评估模型鲁棒性
        Evaluate model robustness
        
        Args:
            test_loader: 测试数据加载器
            device: 设备
            
        Returns:
            robustness_metrics: 鲁棒性指标
        """
        self.model.eval()
        
        robustness_metrics = {
            'clean_performance': {},
            'adversarial_robustness': {},
            'noise_robustness': {},
            'overall_robustness_score': 0.0
        }
        
        # 1. 清洁数据性能
        clean_metrics = self._evaluate_clean_performance(test_loader, device)
        robustness_metrics['clean_performance'] = clean_metrics
        
        # 2. 对抗鲁棒性
        if self.enable_adversarial:
            adv_metrics = self._evaluate_adversarial_robustness(test_loader, device)
            robustness_metrics['adversarial_robustness'] = adv_metrics
        
        # 3. 噪声鲁棒性
        if self.enable_noise_training:
            noise_metrics = self._evaluate_noise_robustness(test_loader, device)
            robustness_metrics['noise_robustness'] = noise_metrics
        
        # 4. 计算综合鲁棒性分数
        overall_score = self._calculate_overall_robustness_score(robustness_metrics)
        robustness_metrics['overall_robustness_score'] = overall_score
        
        return robustness_metrics
    
    def _evaluate_clean_performance(self, test_loader, device: str) -> Dict[str, float]:
        """评估清洁数据性能"""
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for inputs, targets, _ in test_loader:
                inputs = [inp.to(device) if torch.is_tensor(inp) 
                         else [t.to(device) for t in inp] for inp in inputs]
                targets = targets.to(device)
                
                outputs = self.model(*inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        return {'mse': mse, 'mae': mae}
    
    def _evaluate_adversarial_robustness(self, test_loader, device: str) -> Dict[str, float]:
        """评估对抗鲁棒性"""
        criterion = nn.MSELoss()
        
        # 不同攻击强度下的性能
        epsilons = [0.01, 0.05, 0.1, 0.2]
        adv_metrics = {}
        
        for epsilon in epsilons:
            self.adversarial_trainer.epsilon = epsilon
            
            total_loss = 0.0
            n_batches = 0
            
            for inputs, targets, _ in test_loader:
                inputs = [inp.to(device) if torch.is_tensor(inp) 
                         else [t.to(device) for t in inp] for inp in inputs]
                targets = targets.to(device)
                
                # 生成对抗样本
                adv_inputs = self.adversarial_trainer.generate_adversarial_examples(
                    inputs, targets, criterion
                )
                
                with torch.no_grad():
                    adv_outputs = self.model(*adv_inputs)
                    adv_loss = criterion(adv_outputs, targets)
                
                total_loss += adv_loss.item()
                n_batches += 1
            
            adv_metrics[f'epsilon_{epsilon}'] = total_loss / max(n_batches, 1)
        
        return adv_metrics
    
    def _evaluate_noise_robustness(self, test_loader, device: str) -> Dict[str, float]:
        """评估噪声鲁棒性"""
        criterion = nn.MSELoss()
        
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        noise_types = ['gaussian', 'uniform', 'feature_dropout']
        
        noise_metrics = {}
        
        for noise_type in noise_types:
            for noise_level in noise_levels:
                total_loss = 0.0
                n_batches = 0
                
                for inputs, targets, _ in test_loader:
                    inputs = [inp.to(device) if torch.is_tensor(inp) 
                             else [t.to(device) for t in inp] for inp in inputs]
                    targets = targets.to(device)
                    
                    # 添加噪声
                    noisy_inputs = self.noise_trainer.add_training_noise(
                        inputs, noise_type, noise_level
                    )
                    
                    with torch.no_grad():
                        noisy_outputs = self.model(*noisy_inputs)
                        noise_loss = criterion(noisy_outputs, targets)
                    
                    total_loss += noise_loss.item()
                    n_batches += 1
                
                key = f'{noise_type}_level_{noise_level}'
                noise_metrics[key] = total_loss / max(n_batches, 1)
        
        return noise_metrics
    
    def _calculate_overall_robustness_score(self, metrics: Dict[str, Any]) -> float:
        """计算综合鲁棒性分数"""
        clean_mae = metrics['clean_performance'].get('mae', 1.0)
        
        # 对抗鲁棒性分数
        adv_score = 1.0
        if 'adversarial_robustness' in metrics:
            adv_losses = list(metrics['adversarial_robustness'].values())
            if adv_losses:
                avg_adv_loss = np.mean(adv_losses)
                adv_score = clean_mae / (avg_adv_loss + 1e-8)
        
        # 噪声鲁棒性分数
        noise_score = 1.0
        if 'noise_robustness' in metrics:
            noise_losses = list(metrics['noise_robustness'].values())
            if noise_losses:
                avg_noise_loss = np.mean(noise_losses)
                noise_score = clean_mae / (avg_noise_loss + 1e-8)
        
        # 综合分数
        overall_score = (adv_score + noise_score) / 2.0
        
        return min(1.0, overall_score)  # 限制在[0,1]范围内


# 使用示例
def example_usage():
    """使用示例"""
    from cgcnn.enhanced_model import EnhancedCGCNN
    
    # 创建模型
    model = EnhancedCGCNN(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128
    )
    
    # 创建鲁棒训练协调器
    robust_trainer = RobustTrainingOrchestrator(
        model=model,
        enable_adversarial=True,
        enable_noise_training=True,
        enable_ensemble=False,  # 简化示例
        enable_physics_constraints=True
    )
    
    # 模拟数据
    batch_size = 4
    n_atoms = 20
    max_neighbors = 12
    
    atom_fea = torch.randn(n_atoms, 92)
    nbr_fea = torch.randn(n_atoms, max_neighbors, 41)
    nbr_fea_idx = torch.randint(0, n_atoms, (n_atoms, max_neighbors))
    crystal_atom_idx = [torch.arange(5 * i, 5 * (i + 1)) for i in range(batch_size)]
    targets = torch.randn(batch_size, 1)
    
    inputs = (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
    
    # 模拟训练数据加载器
    class MockDataLoader:
        def __init__(self, inputs, targets, n_batches=5):
            self.inputs = inputs
            self.targets = targets
            self.n_batches = n_batches
        
        def __iter__(self):
            for _ in range(self.n_batches):
                yield self.inputs, self.targets, ['sample_id']
        
        def __len__(self):
            return self.n_batches
    
    train_loader = MockDataLoader(inputs, targets)
    test_loader = MockDataLoader(inputs, targets, 3)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 鲁棒训练
    print("Starting robust training...")
    epoch_metrics = robust_trainer.robust_training_epoch(
        train_loader, optimizer, criterion, epoch=15
    )
    
    print(f"Epoch metrics: {epoch_metrics}")
    
    # 评估鲁棒性
    print("Evaluating robustness...")
    robustness_metrics = robust_trainer.evaluate_robustness(test_loader)
    
    print(f"Clean performance: {robustness_metrics['clean_performance']}")
    print(f"Overall robustness score: {robustness_metrics['overall_robustness_score']:.4f}")
    
    # 集成训练示例
    print("\nTesting ensemble training...")
    ensemble_trainer = EnsembleTrainer(n_models=3)
    
    base_config = {
        'orig_atom_fea_len': 92,
        'nbr_fea_len': 41,
        'atom_fea_len': 64,
        'n_conv': 3,
        'h_fea_len': 128
    }
    
    diverse_models = ensemble_trainer.create_diverse_models(base_config)
    print(f"Created {len(diverse_models)} diverse models")


if __name__ == "__main__":
    example_usage() 