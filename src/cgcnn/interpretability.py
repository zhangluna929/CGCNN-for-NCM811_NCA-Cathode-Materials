"""
Model Interpretability for CGCNN

Explainability tools for understanding CGCNN predictions including
gradient-based methods, attention visualization, and feature importance
analysis for materials science applications.

Author: lunazhang
Date: 2023
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import matplotlib.pyplot as plt
import warnings

try:
    from pymatgen.core.structure import Structure
    from pymatgen.analysis.local_env import CrystalNN
    PYMATGEN_AVAILABLE = True
except ImportError:
    warnings.warn("PyMatGen not available. Some features may be limited.")
    PYMATGEN_AVAILABLE = False
    Structure = None


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CGCNN
    针对CGCNN的梯度加权类激活映射
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 找到目标层
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")
        
        # 注册钩子
        self.hooks.append(target_module.register_forward_hook(forward_hook))
        self.hooks.append(target_module.register_backward_hook(backward_hook))
    
    def generate_cam(self, input_data: Tuple, target_class: Optional[int] = None) -> torch.Tensor:
        """
        生成类激活映射
        Generate Class Activation Map
        
        Args:
            input_data: 输入数据 (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            target_class: 目标类别（用于分类任务）
        
        Returns:
            cam: 类激活映射
        """
        self.model.eval()
        
        # 前向传播
        output = self.model(*input_data)
        
        # 计算目标分数
        if target_class is not None:
            target_score = output[0, target_class]
        else:
            target_score = output.mean()  # 对于回归任务，使用平均值
        
        # 反向传播
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # 计算CAM
        if self.gradients is not None and self.activations is not None:
            # 全局平均池化梯度
            weights = torch.mean(self.gradients, dim=0, keepdim=True)
            
            # 加权求和
            cam = torch.sum(weights * self.activations, dim=1)
            cam = F.relu(cam)  # 只保留正值
            
            # 归一化到[0,1]
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            return cam
        else:
            warnings.warn("No gradients or activations captured")
            return torch.zeros(1)
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


class AttentionVisualization:
    """
    注意力机制可视化
    Attention Mechanism Visualization
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        def attention_hook(name):
            def hook(module, input, output):
                # 假设输出包含注意力权重
                if isinstance(output, tuple) and len(output) > 1:
                    self.attention_weights[name] = output[1].detach()
                elif hasattr(module, 'attention_weights'):
                    self.attention_weights[name] = module.attention_weights.detach()
            return hook
        
        # 为所有包含注意力的层注册钩子
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or hasattr(module, 'attention_weights'):
                self.hooks.append(module.register_forward_hook(attention_hook(name)))
    
    def visualize_attention(self, input_data: Tuple, structure: Structure, 
                          save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        可视化注意力权重
        Visualize attention weights
        
        Args:
            input_data: 输入数据
            structure: 对应的晶体结构
            save_path: 保存路径
        
        Returns:
            attention_maps: 注意力图字典
        """
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(*input_data)
        
        attention_maps = {}
        
        for layer_name, weights in self.attention_weights.items():
            # 处理注意力权重
            if weights.dim() == 3:  # [batch, seq_len, seq_len]
                attention_map = weights[0].cpu().numpy()
            elif weights.dim() == 2:  # [seq_len, seq_len]
                attention_map = weights.cpu().numpy()
            else:
                continue
            
            attention_maps[layer_name] = attention_map
            
            # 可视化
            if save_path:
                self._plot_attention_matrix(attention_map, structure, 
                                          f"{save_path}_{layer_name}_attention.png")
        
        return attention_maps
    
    def _plot_attention_matrix(self, attention_matrix: np.ndarray, structure: Structure, 
                              save_path: str):
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Attention Weight')
        
        # 添加原子标签
        atom_labels = [f"{site.specie}_{i}" for i, site in enumerate(structure)]
        if len(atom_labels) <= 20:  # 只有原子数少时才显示标签
            plt.xticks(range(len(atom_labels)), atom_labels, rotation=45)
            plt.yticks(range(len(atom_labels)), atom_labels)
        
        plt.title('Attention Matrix')
        plt.xlabel('Atom Index')
        plt.ylabel('Atom Index')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


class FeatureImportance:
    """
    特征重要性分析
    Feature Importance Analysis
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def integrated_gradients(self, input_data: Tuple, baseline: Optional[Tuple] = None, 
                           steps: int = 50) -> Dict[str, torch.Tensor]:
        """
        集成梯度方法计算特征重要性
        Integrated Gradients for feature importance
        
        Args:
            input_data: 输入数据
            baseline: 基线数据（通常为零向量）
            steps: 积分步数
        
        Returns:
            importance_scores: 特征重要性分数
        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 创建基线
        if baseline is None:
            baseline_atom_fea = torch.zeros_like(atom_fea)
            baseline_nbr_fea = torch.zeros_like(nbr_fea)
            baseline = (baseline_atom_fea, baseline_nbr_fea, nbr_fea_idx, crystal_atom_idx)
        
        # 积分路径
        alphas = torch.linspace(0, 1, steps + 1)
        gradients = []
        
        for alpha in alphas:
            # 插值输入
            interpolated_atom_fea = baseline[0] + alpha * (atom_fea - baseline[0])
            interpolated_nbr_fea = baseline[1] + alpha * (nbr_fea - baseline[1])
            interpolated_input = (interpolated_atom_fea, interpolated_nbr_fea, 
                                nbr_fea_idx, crystal_atom_idx)
            
            # 计算梯度
            interpolated_atom_fea.requires_grad_(True)
            interpolated_nbr_fea.requires_grad_(True)
            
            output = self.model(*interpolated_input)
            self.model.zero_grad()
            
            # 对输出求和以获得标量
            if output.dim() > 0:
                scalar_output = output.sum()
            else:
                scalar_output = output
            
            scalar_output.backward(retain_graph=True)
            
            gradients.append({
                'atom_gradients': interpolated_atom_fea.grad.clone() if interpolated_atom_fea.grad is not None else torch.zeros_like(interpolated_atom_fea),
                'nbr_gradients': interpolated_nbr_fea.grad.clone() if interpolated_nbr_fea.grad is not None else torch.zeros_like(interpolated_nbr_fea)
            })
        
        # 计算积分
        atom_gradients = torch.stack([g['atom_gradients'] for g in gradients])
        nbr_gradients = torch.stack([g['nbr_gradients'] for g in gradients])
        
        # 梯形积分
        atom_integrated_grad = torch.trapz(atom_gradients, alphas.unsqueeze(-1).unsqueeze(-1), dim=0)
        nbr_integrated_grad = torch.trapz(nbr_gradients, alphas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), dim=0)
        
        # 计算重要性分数
        atom_importance = atom_integrated_grad * (atom_fea - baseline[0])
        nbr_importance = nbr_integrated_grad * (nbr_fea - baseline[1])
        
        return {
            'atom_importance': atom_importance,
            'nbr_importance': nbr_importance
        }
    
    def permutation_importance(self, input_data: Tuple, n_permutations: int = 10) -> Dict[str, float]:
        """
        置换重要性分析
        Permutation importance analysis
        
        Args:
            input_data: 输入数据
            n_permutations: 置换次数
        
        Returns:
            importance_scores: 特征重要性分数
        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 原始预测
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(*input_data)
            original_score = original_output.mean().item()
        
        importance_scores = {}
        
        # 原子特征置换
        atom_importance = []
        for feature_idx in range(atom_fea.shape[1]):
            scores = []
            for _ in range(n_permutations):
                # 复制输入
                permuted_atom_fea = atom_fea.clone()
                
                # 置换特定特征
                perm_indices = torch.randperm(permuted_atom_fea.shape[0])
                permuted_atom_fea[:, feature_idx] = permuted_atom_fea[perm_indices, feature_idx]
                
                # 预测
                with torch.no_grad():
                    permuted_output = self.model(permuted_atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                    permuted_score = permuted_output.mean().item()
                
                # 计算重要性（原始分数 - 置换后分数）
                scores.append(original_score - permuted_score)
            
            atom_importance.append(np.mean(scores))
        
        importance_scores['atom_features'] = atom_importance
        
        # 邻居特征置换
        nbr_importance = []
        for feature_idx in range(nbr_fea.shape[2]):
            scores = []
            for _ in range(n_permutations):
                # 复制输入
                permuted_nbr_fea = nbr_fea.clone()
                
                # 置换特定特征
                perm_indices = torch.randperm(permuted_nbr_fea.shape[0])
                permuted_nbr_fea[:, :, feature_idx] = permuted_nbr_fea[perm_indices, :, feature_idx]
                
                # 预测
                with torch.no_grad():
                    permuted_output = self.model(atom_fea, permuted_nbr_fea, nbr_fea_idx, crystal_atom_idx)
                    permuted_score = permuted_output.mean().item()
                
                scores.append(original_score - permuted_score)
            
            nbr_importance.append(np.mean(scores))
        
        importance_scores['neighbor_features'] = nbr_importance
        
        return importance_scores
    
    def lime_explanation(self, input_data: Tuple, n_samples: int = 1000, 
                        n_features: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """
        LIME (Local Interpretable Model-agnostic Explanations) 解释
        LIME explanation for local interpretability
        
        Args:
            input_data: 输入数据
            n_samples: 采样数量
            n_features: 返回的重要特征数量
        
        Returns:
            explanations: 局部解释
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics.pairwise import cosine_similarity
        
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 原始预测
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(*input_data)
            original_prediction = original_output.mean().item()
        
        # 生成扰动样本
        samples = []
        predictions = []
        
        for _ in range(n_samples):
            # 随机选择特征进行扰动
            perturbed_atom_fea = atom_fea.clone()
            perturbed_nbr_fea = nbr_fea.clone()
            
            # 扰动向量（二进制，表示是否扰动该特征）
            atom_perturbation = torch.bernoulli(torch.full((atom_fea.shape[1],), 0.5))
            nbr_perturbation = torch.bernoulli(torch.full((nbr_fea.shape[2],), 0.5))
            
            # 应用扰动（将选中的特征设为零）
            for i, perturb in enumerate(atom_perturbation):
                if perturb:
                    perturbed_atom_fea[:, i] = 0
            
            for i, perturb in enumerate(nbr_perturbation):
                if perturb:
                    perturbed_nbr_fea[:, :, i] = 0
            
            # 记录样本和预测
            sample_vector = torch.cat([atom_perturbation, nbr_perturbation])
            samples.append(sample_vector.numpy())
            
            with torch.no_grad():
                perturbed_output = self.model(perturbed_atom_fea, perturbed_nbr_fea, 
                                            nbr_fea_idx, crystal_atom_idx)
                predictions.append(perturbed_output.mean().item())
        
        samples = np.array(samples)
        predictions = np.array(predictions)
        
        # 计算样本权重（基于与原始样本的相似性）
        original_sample = np.ones(samples.shape[1])
        similarities = cosine_similarity(samples, original_sample.reshape(1, -1)).flatten()
        weights = np.exp(-((1 - similarities) / 0.25) ** 2)
        
        # 训练线性回归模型
        ridge = Ridge(alpha=1.0)
        ridge.fit(samples, predictions, sample_weight=weights)
        
        # 提取特征重要性
        feature_importance = ridge.coef_
        
        # 分离原子特征和邻居特征的重要性
        atom_importance = feature_importance[:atom_fea.shape[1]]
        nbr_importance = feature_importance[atom_fea.shape[1]:]
        
        # 获取最重要的特征
        atom_top_features = sorted(enumerate(atom_importance), 
                                 key=lambda x: abs(x[1]), reverse=True)[:n_features]
        nbr_top_features = sorted(enumerate(nbr_importance), 
                                key=lambda x: abs(x[1]), reverse=True)[:n_features]
        
        return {
            'atom_features': atom_top_features,
            'neighbor_features': nbr_top_features,
            'model_accuracy': ridge.score(samples, predictions, sample_weight=weights)
        }


class ModelExplainer:
    """
    模型解释器
    Model Explainer combining multiple interpretability methods
    """
    
    def __init__(self, model: nn.Module, target_layer: str = 'conv_to_fc'):
        self.model = model
        self.grad_cam = GradCAM(model, target_layer)
        self.attention_viz = AttentionVisualization(model)
        self.feature_importance = FeatureImportance(model)
    
    def comprehensive_analysis(self, input_data: Tuple, structure: Structure, 
                             save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        综合分析
        Comprehensive interpretability analysis
        
        Args:
            input_data: 输入数据
            structure: 对应的晶体结构
            save_dir: 保存目录
        
        Returns:
            analysis_results: 分析结果
        """
        results = {}
        
        # 1. GradCAM分析
        try:
            cam = self.grad_cam.generate_cam(input_data)
            results['grad_cam'] = cam.cpu().numpy()
        except Exception as e:
            warnings.warn(f"GradCAM analysis failed: {e}")
            results['grad_cam'] = None
        
        # 2. 注意力可视化
        try:
            attention_maps = self.attention_viz.visualize_attention(
                input_data, structure, save_dir)
            results['attention_maps'] = attention_maps
        except Exception as e:
            warnings.warn(f"Attention visualization failed: {e}")
            results['attention_maps'] = {}
        
        # 3. 集成梯度
        try:
            integrated_grads = self.feature_importance.integrated_gradients(input_data)
            results['integrated_gradients'] = {
                k: v.cpu().numpy() for k, v in integrated_grads.items()
            }
        except Exception as e:
            warnings.warn(f"Integrated gradients failed: {e}")
            results['integrated_gradients'] = {}
        
        # 4. 置换重要性
        try:
            perm_importance = self.feature_importance.permutation_importance(input_data)
            results['permutation_importance'] = perm_importance
        except Exception as e:
            warnings.warn(f"Permutation importance failed: {e}")
            results['permutation_importance'] = {}
        
        # 5. LIME解释
        try:
            lime_explanation = self.feature_importance.lime_explanation(input_data)
            results['lime_explanation'] = lime_explanation
        except Exception as e:
            warnings.warn(f"LIME explanation failed: {e}")
            results['lime_explanation'] = {}
        
        return results
    
    def visualize_results(self, results: Dict[str, Any], structure: Structure, 
                         save_path: str):
        """
        可视化结果
        Visualize interpretation results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. GradCAM
        if results['grad_cam'] is not None:
            axes[0, 0].imshow(results['grad_cam'].reshape(-1, 1), aspect='auto', cmap='hot')
            axes[0, 0].set_title('GradCAM Activation')
            axes[0, 0].set_ylabel('Atom Index')
        
        # 2. 置换重要性
        if 'atom_features' in results['permutation_importance']:
            atom_imp = results['permutation_importance']['atom_features']
            axes[0, 1].bar(range(len(atom_imp)), atom_imp)
            axes[0, 1].set_title('Atom Feature Importance')
            axes[0, 1].set_xlabel('Feature Index')
            axes[0, 1].set_ylabel('Importance Score')
        
        # 3. LIME特征重要性
        if 'atom_features' in results['lime_explanation']:
            lime_features = results['lime_explanation']['atom_features']
            indices, values = zip(*lime_features)
            axes[1, 0].bar(range(len(values)), values)
            axes[1, 0].set_title('LIME Feature Importance')
            axes[1, 0].set_xlabel('Top Features')
            axes[1, 0].set_ylabel('LIME Score')
        
        # 4. 集成梯度
        if 'atom_importance' in results['integrated_gradients']:
            atom_ig = results['integrated_gradients']['atom_importance']
            if atom_ig.ndim > 1:
                atom_ig_sum = np.sum(np.abs(atom_ig), axis=1)
            else:
                atom_ig_sum = np.abs(atom_ig)
            axes[1, 1].plot(atom_ig_sum)
            axes[1, 1].set_title('Integrated Gradients')
            axes[1, 1].set_xlabel('Atom Index')
            axes[1, 1].set_ylabel('Importance')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def cleanup(self):
        self.grad_cam.cleanup()
        self.attention_viz.cleanup()


# 使用示例
def example_usage():
    # 创建解释器
    # model = CrystalGraphConvNet(...)
    # explainer = ModelExplainer(model)
    
    # 进行综合分析
    # results = explainer.comprehensive_analysis(input_data, structure, save_dir="explanations")
    
    # 可视化结果
    # explainer.visualize_results(results, structure, "interpretation_results.png")
    
    # 清理资源
    # explainer.cleanup()
    
    pass 