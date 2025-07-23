"""
Comprehensive Interpretability System

Advanced interpretability analysis for CGCNN including atomic-level feature
importance, chemical bond contribution assessment, and physics mechanism
visualization to help materials scientists understand prediction foundations.

Author: lunazhang
Date: 2023
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from pymatgen.core.structure import Structure
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.analysis.bond_valence import BVAnalyzer
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

from .interpretability import ModelExplainer


class AtomicFeatureAnalyzer:
    """
    原子特征分析器
    Atomic Feature Analyzer
    """
    
    def __init__(self, model: nn.Module, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or self._default_feature_names()
        
        self.logger = logging.getLogger(__name__)
    
    def _default_feature_names(self) -> List[str]:
        """默认原子特征名称"""
        # 基于常见的原子描述符
        feature_names = []
        
        # 基本原子性质
        basic_features = [
            'atomic_number', 'atomic_mass', 'atomic_radius', 'ionic_radius',
            'electronegativity', 'electron_affinity', 'ionization_energy',
            'valence_electrons', 'period', 'group'
        ]
        feature_names.extend(basic_features)
        
        # 电子结构特征
        electronic_features = [
            's_electrons', 'p_electrons', 'd_electrons', 'f_electrons',
            'unpaired_electrons', 'oxidation_state'
        ]
        feature_names.extend(electronic_features)
        
        # 物理性质
        physical_features = [
            'melting_point', 'boiling_point', 'density', 'thermal_conductivity',
            'electrical_conductivity', 'magnetic_moment'
        ]
        feature_names.extend(physical_features)
        
        # 化学性质
        chemical_features = [
            'bond_strength', 'coordination_number', 'polarizability',
            'hardness', 'bond_length_avg', 'bond_angle_avg'
        ]
        feature_names.extend(chemical_features)
        
        # 扩展到92维（如果需要）
        while len(feature_names) < 92:
            feature_names.append(f'feature_{len(feature_names)}')
        
        return feature_names[:92]
    
    def analyze_atomic_contributions(self, input_data: Tuple, 
                                   structure: Optional[Structure] = None,
                                   method: str = 'integrated_gradients') -> Dict[str, Any]:
        """
        分析原子贡献度
        Analyze atomic contributions to prediction
        
        Args:
            input_data: 输入数据
            structure: 晶体结构（可选）
            method: 分析方法
            
        Returns:
            atomic_analysis: 原子分析结果
        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        if method == 'integrated_gradients':
            return self._integrated_gradients_analysis(input_data, structure)
        elif method == 'layer_wise_relevance':
            return self._lrp_analysis(input_data, structure)
        elif method == 'attention_weights':
            return self._attention_analysis(input_data, structure)
        elif method == 'occlusion':
            return self._occlusion_analysis(input_data, structure)
        else:
            raise ValueError(f"Unknown analysis method: {method}")
    
    def _integrated_gradients_analysis(self, input_data: Tuple, 
                                     structure: Optional[Structure]) -> Dict[str, Any]:
        """集成梯度分析"""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 创建基线（零特征）
        baseline_atom_fea = torch.zeros_like(atom_fea)
        baseline_nbr_fea = torch.zeros_like(nbr_fea)
        
        # 积分步数
        n_steps = 50
        alphas = torch.linspace(0, 1, n_steps + 1)
        
        # 收集梯度
        atom_gradients = []
        nbr_gradients = []
        
        for alpha in alphas:
            # 插值
            interp_atom_fea = baseline_atom_fea + alpha * (atom_fea - baseline_atom_fea)
            interp_nbr_fea = baseline_nbr_fea + alpha * (nbr_fea - baseline_nbr_fea)
            
            interp_atom_fea.requires_grad_(True)
            interp_nbr_fea.requires_grad_(True)
            
            # 前向传播
            output = self.model(interp_atom_fea, interp_nbr_fea, nbr_fea_idx, crystal_atom_idx)
            
            # 反向传播
            self.model.zero_grad()
            output.sum().backward()
            
            atom_gradients.append(interp_atom_fea.grad.clone())
            nbr_gradients.append(interp_nbr_fea.grad.clone())
        
        # 计算积分
        atom_gradients = torch.stack(atom_gradients)
        nbr_gradients = torch.stack(nbr_gradients)
        
        atom_integrated_grad = torch.trapz(atom_gradients, alphas.unsqueeze(-1).unsqueeze(-1), dim=0)
        nbr_integrated_grad = torch.trapz(nbr_gradients, alphas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), dim=0)
        
        # 计算重要性分数
        atom_importance = atom_integrated_grad * (atom_fea - baseline_atom_fea)
        nbr_importance = nbr_integrated_grad * (nbr_fea - baseline_nbr_fea)
        
        # 分析结果
        analysis_results = {
            'method': 'integrated_gradients',
            'atom_importance': atom_importance.detach().cpu().numpy(),
            'neighbor_importance': nbr_importance.detach().cpu().numpy(),
            'atom_feature_importance': {},
            'atom_contributions': {},
            'summary_statistics': {}
        }
        
        # 按特征维度分析
        for i, feature_name in enumerate(self.feature_names):
            feature_importance = atom_importance[:, i].abs().sum().item()
            analysis_results['atom_feature_importance'][feature_name] = feature_importance
        
        # 按原子分析
        for atom_idx in range(atom_fea.size(0)):
            atom_contribution = atom_importance[atom_idx].abs().sum().item()
            analysis_results['atom_contributions'][f'atom_{atom_idx}'] = atom_contribution
        
        # 统计信息
        analysis_results['summary_statistics'] = {
            'total_importance': atom_importance.abs().sum().item(),
            'max_atom_contribution': max(analysis_results['atom_contributions'].values()),
            'min_atom_contribution': min(analysis_results['atom_contributions'].values()),
            'top_features': sorted(analysis_results['atom_feature_importance'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
        }
        
        return analysis_results
    
    def _attention_analysis(self, input_data: Tuple, 
                          structure: Optional[Structure]) -> Dict[str, Any]:
        """注意力权重分析"""
        # 这需要模型支持注意力权重提取
        if not hasattr(self.model, 'get_attention_weights'):
            self.logger.warning("Model does not support attention weight extraction")
            return {'error': 'Attention weights not available'}
        
        attention_weights = self.model.get_attention_weights(*input_data)
        
        analysis_results = {
            'method': 'attention_weights',
            'attention_maps': [],
            'attention_statistics': {}
        }
        
        for layer_idx, attention in enumerate(attention_weights):
            # 分析每层的注意力模式
            attention_np = attention.detach().cpu().numpy()
            
            layer_analysis = {
                'layer': layer_idx,
                'attention_matrix': attention_np,
                'max_attention': np.max(attention_np),
                'min_attention': np.min(attention_np),
                'attention_entropy': self._calculate_attention_entropy(attention_np),
                'focused_atoms': self._identify_focused_atoms(attention_np)
            }
            
            analysis_results['attention_maps'].append(layer_analysis)
        
        return analysis_results
    
    def _occlusion_analysis(self, input_data: Tuple, 
                          structure: Optional[Structure]) -> Dict[str, Any]:
        """遮挡分析"""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 原始预测
        with torch.no_grad():
            original_output = self.model(*input_data)
            original_prediction = original_output.item()
        
        analysis_results = {
            'method': 'occlusion',
            'original_prediction': original_prediction,
            'atom_occlusion_effects': {},
            'feature_occlusion_effects': {}
        }
        
        # 原子遮挡分析
        for atom_idx in range(atom_fea.size(0)):
            # 遮挡单个原子
            occluded_atom_fea = atom_fea.clone()
            occluded_atom_fea[atom_idx] = 0  # 将原子特征置零
            
            with torch.no_grad():
                occluded_output = self.model(occluded_atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                occluded_prediction = occluded_output.item()
            
            # 计算影响
            effect = original_prediction - occluded_prediction
            analysis_results['atom_occlusion_effects'][f'atom_{atom_idx}'] = effect
        
        # 特征遮挡分析
        for feature_idx, feature_name in enumerate(self.feature_names):
            # 遮挡单个特征维度
            occluded_atom_fea = atom_fea.clone()
            occluded_atom_fea[:, feature_idx] = 0
            
            with torch.no_grad():
                occluded_output = self.model(occluded_atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                occluded_prediction = occluded_output.item()
            
            effect = original_prediction - occluded_prediction
            analysis_results['feature_occlusion_effects'][feature_name] = effect
        
        return analysis_results
    
    def _calculate_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """计算注意力熵"""
        # 归一化注意力权重
        attention_flat = attention_matrix.flatten()
        attention_prob = attention_flat / (np.sum(attention_flat) + 1e-8)
        
        # 计算熵
        entropy = -np.sum(attention_prob * np.log(attention_prob + 1e-8))
        return entropy
    
    def _identify_focused_atoms(self, attention_matrix: np.ndarray, 
                              top_k: int = 5) -> List[int]:
        """识别注意力集中的原子"""
        # 计算每个原子的总注意力权重
        atom_attention = np.sum(attention_matrix, axis=1)
        
        # 找到top-k原子
        top_indices = np.argsort(atom_attention)[-top_k:][::-1]
        
        return top_indices.tolist()


class ChemicalBondAnalyzer:
    """
    化学键分析器
    Chemical Bond Analyzer
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_bond_contributions(self, input_data: Tuple, 
                                 structure: Optional[Structure] = None) -> Dict[str, Any]:
        """
        分析化学键贡献
        Analyze chemical bond contributions
        
        Args:
            input_data: 输入数据
            structure: 晶体结构
            
        Returns:
            bond_analysis: 化学键分析结果
        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        bond_analysis = {
            'bond_importance_matrix': {},
            'bond_type_contributions': {},
            'distance_effect_analysis': {},
            'coordination_analysis': {}
        }
        
        # 1. 邻居特征重要性分析
        bond_importance = self._analyze_neighbor_importance(input_data)
        bond_analysis['bond_importance_matrix'] = bond_importance
        
        # 2. 化学键类型分析
        if structure is not None:
            bond_type_analysis = self._analyze_bond_types(input_data, structure)
            bond_analysis['bond_type_contributions'] = bond_type_analysis
        
        # 3. 距离效应分析
        distance_analysis = self._analyze_distance_effects(input_data)
        bond_analysis['distance_effect_analysis'] = distance_analysis
        
        # 4. 配位环境分析
        coordination_analysis = self._analyze_coordination_environment(input_data, structure)
        bond_analysis['coordination_analysis'] = coordination_analysis
        
        return bond_analysis
    
    def _analyze_neighbor_importance(self, input_data: Tuple) -> Dict[str, Any]:
        """分析邻居重要性"""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 计算邻居特征的梯度
        nbr_fea_grad = nbr_fea.clone().requires_grad_(True)
        
        output = self.model(atom_fea, nbr_fea_grad, nbr_fea_idx, crystal_atom_idx)
        
        self.model.zero_grad()
        output.sum().backward()
        
        neighbor_gradients = nbr_fea_grad.grad.detach().cpu().numpy()
        
        # 计算重要性矩阵
        importance_matrix = np.abs(neighbor_gradients * nbr_fea.detach().cpu().numpy())
        
        # 按原子-邻居对分析
        bond_importance = {}
        for atom_idx in range(importance_matrix.shape[0]):
            atom_bonds = {}
            for nbr_idx in range(importance_matrix.shape[1]):
                neighbor_atom_idx = nbr_fea_idx[atom_idx, nbr_idx].item()
                bond_strength = np.sum(importance_matrix[atom_idx, nbr_idx])
                
                bond_key = f"atom_{atom_idx}_to_atom_{neighbor_atom_idx}"
                atom_bonds[bond_key] = bond_strength
            
            bond_importance[f"atom_{atom_idx}"] = atom_bonds
        
        return {
            'importance_matrix': importance_matrix.tolist(),
            'bond_strengths': bond_importance,
            'summary': {
                'max_bond_importance': np.max(importance_matrix),
                'min_bond_importance': np.min(importance_matrix),
                'avg_bond_importance': np.mean(importance_matrix)
            }
        }
    
    def _analyze_bond_types(self, input_data: Tuple, 
                          structure: Structure) -> Dict[str, Any]:
        """分析化学键类型"""
        if not PYMATGEN_AVAILABLE:
            return {'error': 'PyMatGen not available'}
        
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 使用CrystalNN分析局部环境
        crystal_nn = CrystalNN()
        
        bond_type_contributions = defaultdict(list)
        
        for atom_idx, site in enumerate(structure):
            try:
                # 获取邻居信息
                nn_info = crystal_nn.get_nn_info(structure, atom_idx)
                
                for neighbor_info in nn_info:
                    neighbor_idx = neighbor_info['site_index']
                    bond_length = neighbor_info['weight']
                    
                    # 确定化学键类型
                    central_element = site.specie.symbol
                    neighbor_element = structure[neighbor_idx].specie.symbol
                    bond_type = f"{central_element}-{neighbor_element}"
                    
                    # 找到对应的邻居特征重要性
                    nbr_position = np.where(nbr_fea_idx[atom_idx].cpu().numpy() == neighbor_idx)[0]
                    if len(nbr_position) > 0:
                        nbr_pos = nbr_position[0]
                        
                        # 计算该键的贡献
                        bond_contribution = self._calculate_single_bond_contribution(
                            input_data, atom_idx, nbr_pos
                        )
                        
                        bond_type_contributions[bond_type].append({
                            'atom_pair': (atom_idx, neighbor_idx),
                            'bond_length': bond_length,
                            'contribution': bond_contribution
                        })
            
            except Exception as e:
                self.logger.warning(f"Failed to analyze atom {atom_idx}: {e}")
                continue
        
        # 统计不同键类型的平均贡献
        bond_type_summary = {}
        for bond_type, contributions in bond_type_contributions.items():
            if contributions:
                avg_contribution = np.mean([c['contribution'] for c in contributions])
                avg_bond_length = np.mean([c['bond_length'] for c in contributions])
                
                bond_type_summary[bond_type] = {
                    'count': len(contributions),
                    'avg_contribution': avg_contribution,
                    'avg_bond_length': avg_bond_length,
                    'contributions': contributions
                }
        
        return bond_type_summary
    
    def _calculate_single_bond_contribution(self, input_data: Tuple, 
                                          atom_idx: int, nbr_idx: int) -> float:
        """计算单个化学键的贡献"""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 原始预测
        with torch.no_grad():
            original_output = self.model(*input_data)
            original_pred = original_output.item()
        
        # 移除特定邻居的影响
        modified_nbr_fea = nbr_fea.clone()
        modified_nbr_fea[atom_idx, nbr_idx] = 0  # 将该邻居特征置零
        
        with torch.no_grad():
            modified_output = self.model(atom_fea, modified_nbr_fea, nbr_fea_idx, crystal_atom_idx)
            modified_pred = modified_output.item()
        
        # 计算贡献度
        contribution = abs(original_pred - modified_pred)
        
        return contribution
    
    def _analyze_distance_effects(self, input_data: Tuple) -> Dict[str, Any]:
        """分析距离效应"""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 假设邻居特征的第一维是距离相关特征
        distance_features = nbr_fea[:, :, 0].detach().cpu().numpy()
        
        # 计算距离对重要性的影响
        nbr_fea_grad = nbr_fea.clone().requires_grad_(True)
        output = self.model(atom_fea, nbr_fea_grad, nbr_fea_idx, crystal_atom_idx)
        
        self.model.zero_grad()
        output.sum().backward()
        
        importance_scores = np.abs(nbr_fea_grad.grad[:, :, 0].detach().cpu().numpy())
        
        # 分析距离-重要性关系
        distance_bins = np.linspace(np.min(distance_features), np.max(distance_features), 10)
        bin_indices = np.digitize(distance_features.flatten(), distance_bins)
        
        distance_importance_correlation = {}
        for bin_idx in range(1, len(distance_bins)):
            mask = bin_indices == bin_idx
            if np.any(mask):
                avg_distance = np.mean(distance_features.flatten()[mask])
                avg_importance = np.mean(importance_scores.flatten()[mask])
                
                distance_importance_correlation[f'bin_{bin_idx}'] = {
                    'distance_range': (distance_bins[bin_idx-1], distance_bins[bin_idx]),
                    'avg_distance': avg_distance,
                    'avg_importance': avg_importance,
                    'sample_count': np.sum(mask)
                }
        
        return {
            'distance_importance_correlation': distance_importance_correlation,
            'correlation_coefficient': np.corrcoef(
                distance_features.flatten(), 
                importance_scores.flatten()
            )[0, 1]
        }
    
    def _analyze_coordination_environment(self, input_data: Tuple, 
                                        structure: Optional[Structure]) -> Dict[str, Any]:
        """分析配位环境"""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        coordination_analysis = {
            'coordination_numbers': {},
            'coordination_contributions': {},
            'environment_types': {}
        }
        
        # 计算每个原子的配位数
        for atom_idx in range(atom_fea.size(0)):
            # 计算有效邻居数（非零邻居特征）
            atom_neighbors = nbr_fea[atom_idx]
            non_zero_neighbors = torch.sum(torch.norm(atom_neighbors, dim=1) > 1e-6).item()
            
            coordination_analysis['coordination_numbers'][f'atom_{atom_idx}'] = non_zero_neighbors
            
            # 计算配位环境对预测的贡献
            coord_contribution = self._calculate_coordination_contribution(input_data, atom_idx)
            coordination_analysis['coordination_contributions'][f'atom_{atom_idx}'] = coord_contribution
        
        # 如果有结构信息，分析配位环境类型
        if structure is not None and PYMATGEN_AVAILABLE:
            coordination_analysis['environment_types'] = self._classify_coordination_environments(structure)
        
        return coordination_analysis
    
    def _calculate_coordination_contribution(self, input_data: Tuple, atom_idx: int) -> float:
        """计算配位环境贡献"""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 原始预测
        with torch.no_grad():
            original_output = self.model(*input_data)
            original_pred = original_output.item()
        
        # 移除该原子的所有邻居
        modified_nbr_fea = nbr_fea.clone()
        modified_nbr_fea[atom_idx] = 0
        
        with torch.no_grad():
            modified_output = self.model(atom_fea, modified_nbr_fea, nbr_fea_idx, crystal_atom_idx)
            modified_pred = modified_output.item()
        
        return abs(original_pred - modified_pred)
    
    def _classify_coordination_environments(self, structure: Structure) -> Dict[str, Any]:
        """分类配位环境"""
        try:
            crystal_nn = CrystalNN()
            environment_types = {}
            
            for atom_idx, site in enumerate(structure):
                nn_info = crystal_nn.get_nn_info(structure, atom_idx)
                coord_num = len(nn_info)
                
                # 简单的配位环境分类
                if coord_num <= 4:
                    env_type = 'tetrahedral'
                elif coord_num == 6:
                    env_type = 'octahedral'
                elif coord_num == 8:
                    env_type = 'cubic'
                else:
                    env_type = f'coordination_{coord_num}'
                
                environment_types[f'atom_{atom_idx}'] = {
                    'type': env_type,
                    'coordination_number': coord_num,
                    'neighbors': [info['site_index'] for info in nn_info]
                }
            
            return environment_types
        
        except Exception as e:
            self.logger.warning(f"Failed to classify coordination environments: {e}")
            return {}


class PhysicsMechanismVisualizer:
    """
    物理机制可视化器
    Physics Mechanism Visualizer
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
        self.logger = logging.getLogger(__name__)
    
    def create_comprehensive_visualization(self, input_data: Tuple,
                                         structure: Optional[Structure] = None,
                                         atomic_analysis: Optional[Dict] = None,
                                         bond_analysis: Optional[Dict] = None,
                                         save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        创建综合可视化
        Create comprehensive visualization
        
        Args:
            input_data: 输入数据
            structure: 晶体结构
            atomic_analysis: 原子分析结果
            bond_analysis: 化学键分析结果
            save_path: 保存路径
            
        Returns:
            visualization_results: 可视化结果
        """
        if not PLOTTING_AVAILABLE:
            return {'error': 'Plotting libraries not available'}
        
        visualization_results = {
            'figures': {},
            'interactive_plots': {},
            'summary_statistics': {}
        }
        
        # 1. 原子重要性热图
        if atomic_analysis:
            atomic_heatmap = self._create_atomic_importance_heatmap(atomic_analysis)
            visualization_results['figures']['atomic_heatmap'] = atomic_heatmap
        
        # 2. 化学键网络图
        if bond_analysis:
            bond_network = self._create_bond_network_plot(bond_analysis, structure)
            visualization_results['figures']['bond_network'] = bond_network
        
        # 3. 特征重要性条形图
        if atomic_analysis:
            feature_importance_plot = self._create_feature_importance_plot(atomic_analysis)
            visualization_results['figures']['feature_importance'] = feature_importance_plot
        
        # 4. 3D结构可视化（如果有结构信息）
        if structure is not None:
            structure_3d = self._create_3d_structure_plot(structure, atomic_analysis)
            visualization_results['interactive_plots']['structure_3d'] = structure_3d
        
        # 5. 距离-重要性关系图
        if bond_analysis and 'distance_effect_analysis' in bond_analysis:
            distance_plot = self._create_distance_importance_plot(bond_analysis['distance_effect_analysis'])
            visualization_results['figures']['distance_importance'] = distance_plot
        
        # 保存图片
        if save_path:
            self._save_visualizations(visualization_results, save_path)
        
        return visualization_results
    
    def _create_atomic_importance_heatmap(self, atomic_analysis: Dict) -> plt.Figure:
        """创建原子重要性热图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原子重要性矩阵
        if 'atom_importance' in atomic_analysis:
            importance_matrix = atomic_analysis['atom_importance']
            
            # 热图1：原子-特征重要性
            im1 = ax1.imshow(importance_matrix, cmap='viridis', aspect='auto')
            ax1.set_title('Atomic Feature Importance Matrix')
            ax1.set_xlabel('Feature Index')
            ax1.set_ylabel('Atom Index')
            plt.colorbar(im1, ax=ax1, label='Importance Score')
        
        # 特征重要性排序
        if 'atom_feature_importance' in atomic_analysis:
            feature_importance = atomic_analysis['atom_feature_importance']
            
            # 选择前20个最重要的特征
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            
            features, scores = zip(*sorted_features)
            
            bars = ax2.barh(range(len(features)), scores)
            ax2.set_yticks(range(len(features)))
            ax2.set_yticklabels(features, fontsize=8)
            ax2.set_xlabel('Importance Score')
            ax2.set_title('Top 20 Feature Importance')
            
            # 颜色映射
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        return fig
    
    def _create_bond_network_plot(self, bond_analysis: Dict, 
                                structure: Optional[Structure]) -> plt.Figure:
        """创建化学键网络图"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        if 'bond_importance_matrix' in bond_analysis:
            bond_data = bond_analysis['bond_importance_matrix']
            
            if 'bond_strengths' in bond_data:
                # 创建网络图
                import networkx as nx
                
                G = nx.Graph()
                
                # 添加节点和边
                for atom_key, bonds in bond_data['bond_strengths'].items():
                    atom_idx = int(atom_key.split('_')[1])
                    G.add_node(atom_idx)
                    
                    for bond_key, strength in bonds.items():
                        # 解析键信息
                        parts = bond_key.split('_to_atom_')
                        if len(parts) == 2:
                            target_atom = int(parts[1])
                            G.add_edge(atom_idx, target_atom, weight=strength)
                
                # 绘制网络
                pos = nx.spring_layout(G, k=2, iterations=50)
                
                # 绘制节点
                node_sizes = [G.degree(node) * 100 for node in G.nodes()]
                nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                     node_color='lightblue', alpha=0.7, ax=ax)
                
                # 绘制边，宽度表示重要性
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]
                
                # 归一化权重用于线宽
                max_weight = max(weights) if weights else 1
                edge_widths = [w / max_weight * 5 for w in weights]
                
                nx.draw_networkx_edges(G, pos, width=edge_widths, 
                                     alpha=0.6, edge_color='gray', ax=ax)
                
                # 添加标签
                nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                
                ax.set_title('Chemical Bond Network\n(Node size: coordination, Edge width: importance)')
                ax.axis('off')
        
        return fig
    
    def _create_feature_importance_plot(self, atomic_analysis: Dict) -> plt.Figure:
        """创建特征重要性图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if 'atom_feature_importance' in atomic_analysis:
            feature_importance = atomic_analysis['atom_feature_importance']
            
            # 1. 总体特征重要性
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # 前20个特征
            top_features = sorted_features[:20]
            features, scores = zip(*top_features)
            
            axes[0, 0].bar(range(len(features)), scores, color='skyblue')
            axes[0, 0].set_xticks(range(len(features)))
            axes[0, 0].set_xticklabels(features, rotation=45, ha='right', fontsize=8)
            axes[0, 0].set_title('Top 20 Feature Importance')
            axes[0, 0].set_ylabel('Importance Score')
            
            # 2. 特征重要性分布
            all_scores = list(feature_importance.values())
            axes[0, 1].hist(all_scores, bins=30, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Feature Importance Distribution')
            axes[0, 1].set_xlabel('Importance Score')
            axes[0, 1].set_ylabel('Frequency')
            
            # 3. 累积重要性
            sorted_scores = sorted(all_scores, reverse=True)
            cumulative_scores = np.cumsum(sorted_scores) / np.sum(sorted_scores)
            
            axes[1, 0].plot(range(len(cumulative_scores)), cumulative_scores, 'b-', linewidth=2)
            axes[1, 0].axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
            axes[1, 0].set_title('Cumulative Feature Importance')
            axes[1, 0].set_xlabel('Feature Rank')
            axes[1, 0].set_ylabel('Cumulative Importance')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 特征类别分析（如果可以分类的话）
            feature_categories = self._categorize_features(list(features))
            category_scores = defaultdict(list)
            
            for feature, score in top_features:
                category = self._get_feature_category(feature)
                category_scores[category].append(score)
            
            categories = list(category_scores.keys())
            avg_scores = [np.mean(category_scores[cat]) for cat in categories]
            
            axes[1, 1].bar(categories, avg_scores, color='orange', alpha=0.7)
            axes[1, 1].set_title('Average Importance by Feature Category')
            axes[1, 1].set_ylabel('Average Importance')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def _create_3d_structure_plot(self, structure: Structure, 
                                atomic_analysis: Optional[Dict] = None):
        """创建3D结构图"""
        if not PLOTTING_AVAILABLE:
            return None
        
        # 使用plotly创建交互式3D图
        fig = go.Figure()
        
        # 获取原子位置和类型
        positions = [site.coords for site in structure]
        elements = [site.specie.symbol for site in structure]
        
        # 原子重要性（如果有的话）
        atom_importance = None
        if atomic_analysis and 'atom_contributions' in atomic_analysis:
            contributions = atomic_analysis['atom_contributions']
            atom_importance = [contributions.get(f'atom_{i}', 0) for i in range(len(structure))]
        
        # 元素颜色映射
        element_colors = {
            'Li': 'purple', 'Ni': 'green', 'Co': 'blue', 'Mn': 'orange',
            'Al': 'gray', 'O': 'red', 'C': 'black'
        }
        
        # 添加原子
        for i, (pos, element) in enumerate(zip(positions, elements)):
            color = element_colors.get(element, 'silver')
            size = 10
            
            # 根据重要性调整大小
            if atom_importance:
                size = 5 + atom_importance[i] * 20  # 基础大小5，重要性影响最多20
            
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(size=size, color=color, opacity=0.8),
                name=f'{element}_{i}',
                text=f'Atom {i}: {element}<br>Importance: {atom_importance[i] if atom_importance else "N/A"}',
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # 添加化学键（简化版本）
        if len(structure) < 50:  # 只对小结构显示键
            for i, site1 in enumerate(structure):
                for j, site2 in enumerate(structure[i+1:], i+1):
                    distance = site1.distance(site2)
                    if distance < 3.0:  # 假设3Å内的原子有键连接
                        fig.add_trace(go.Scatter3d(
                            x=[site1.coords[0], site2.coords[0]],
                            y=[site1.coords[1], site2.coords[1]],
                            z=[site1.coords[2], site2.coords[2]],
                            mode='lines',
                            line=dict(color='gray', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        fig.update_layout(
            title='3D Crystal Structure with Atomic Importance',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='cube'
            ),
            showlegend=True
        )
        
        return fig
    
    def _create_distance_importance_plot(self, distance_analysis: Dict) -> plt.Figure:
        """创建距离-重要性关系图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if 'distance_importance_correlation' in distance_analysis:
            correlation_data = distance_analysis['distance_importance_correlation']
            
            distances = []
            importances = []
            sample_counts = []
            
            for bin_data in correlation_data.values():
                distances.append(bin_data['avg_distance'])
                importances.append(bin_data['avg_importance'])
                sample_counts.append(bin_data['sample_count'])
            
            # 散点图：距离 vs 重要性
            scatter = ax1.scatter(distances, importances, s=[c*10 for c in sample_counts], 
                                alpha=0.6, c=sample_counts, cmap='viridis')
            ax1.set_xlabel('Average Distance (Å)')
            ax1.set_ylabel('Average Importance')
            ax1.set_title('Distance vs Importance Correlation')
            ax1.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(distances) > 1:
                z = np.polyfit(distances, importances, 1)
                p = np.poly1d(z)
                ax1.plot(distances, p(distances), "r--", alpha=0.8, 
                        label=f'Trend (slope: {z[0]:.3f})')
                ax1.legend()
            
            plt.colorbar(scatter, ax=ax1, label='Sample Count')
            
            # 相关系数显示
            correlation_coef = distance_analysis.get('correlation_coefficient', 0)
            ax1.text(0.05, 0.95, f'Correlation: {correlation_coef:.3f}', 
                    transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 距离分布直方图
        ax2.bar(range(len(distances)), importances, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Distance Bin')
        ax2.set_ylabel('Average Importance')
        ax2.set_title('Importance by Distance Bins')
        ax2.set_xticks(range(len(distances)))
        ax2.set_xticklabels([f'{d:.2f}' for d in distances], rotation=45)
        
        plt.tight_layout()
        return fig
    
    def _categorize_features(self, features: List[str]) -> Dict[str, List[str]]:
        """特征分类"""
        categories = {
            'atomic_properties': [],
            'electronic_structure': [],
            'physical_properties': [],
            'chemical_properties': [],
            'other': []
        }
        
        for feature in features:
            feature_lower = feature.lower()
            
            if any(prop in feature_lower for prop in ['atomic_number', 'atomic_mass', 'radius']):
                categories['atomic_properties'].append(feature)
            elif any(prop in feature_lower for prop in ['electron', 'valence', 'oxidation']):
                categories['electronic_structure'].append(feature)
            elif any(prop in feature_lower for prop in ['melting', 'boiling', 'density', 'conductivity']):
                categories['physical_properties'].append(feature)
            elif any(prop in feature_lower for prop in ['bond', 'coordination', 'electronegativity']):
                categories['chemical_properties'].append(feature)
            else:
                categories['other'].append(feature)
        
        return categories
    
    def _get_feature_category(self, feature: str) -> str:
        """获取特征类别"""
        feature_lower = feature.lower()
        
        if any(prop in feature_lower for prop in ['atomic_number', 'atomic_mass', 'radius']):
            return 'Atomic Properties'
        elif any(prop in feature_lower for prop in ['electron', 'valence', 'oxidation']):
            return 'Electronic Structure'
        elif any(prop in feature_lower for prop in ['melting', 'boiling', 'density', 'conductivity']):
            return 'Physical Properties'
        elif any(prop in feature_lower for prop in ['bond', 'coordination', 'electronegativity']):
            return 'Chemical Properties'
        else:
            return 'Other'
    
    def _save_visualizations(self, visualization_results: Dict, save_path: str):
        """保存可视化结果"""
        import os
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 保存matplotlib图片
        for fig_name, fig in visualization_results.get('figures', {}).items():
            if fig is not None:
                fig_path = os.path.join(save_path, f'{fig_name}.png')
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # 保存plotly交互图
        for plot_name, fig in visualization_results.get('interactive_plots', {}).items():
            if fig is not None:
                html_path = os.path.join(save_path, f'{plot_name}.html')
                fig.write_html(html_path)
        
        self.logger.info(f"Visualizations saved to {save_path}")


class ComprehensiveInterpretabilitySystem:
    """
    综合可解释性系统
    Comprehensive Interpretability System
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
        # 初始化分析器
        self.atomic_analyzer = AtomicFeatureAnalyzer(model)
        self.bond_analyzer = ChemicalBondAnalyzer(model)
        self.visualizer = PhysicsMechanismVisualizer(model)
        
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_analysis(self, input_data: Tuple,
                             structure: Optional[Structure] = None,
                             analysis_methods: List[str] = ['integrated_gradients', 'occlusion'],
                             save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        综合可解释性分析
        Comprehensive interpretability analysis
        
        Args:
            input_data: 输入数据
            structure: 晶体结构
            analysis_methods: 分析方法列表
            save_path: 保存路径
            
        Returns:
            comprehensive_results: 综合分析结果
        """
        self.logger.info("Starting comprehensive interpretability analysis")
        
        comprehensive_results = {
            'model_prediction': {},
            'atomic_analysis': {},
            'bond_analysis': {},
            'visualization_results': {},
            'summary_insights': {},
            'recommendations': []
        }
        
        # 1. 获取模型预测
        with torch.no_grad():
            prediction = self.model(*input_data)
            comprehensive_results['model_prediction'] = {
                'value': prediction.item(),
                'confidence': 'high'  # 可以基于不确定性估计
            }
        
        # 2. 原子级分析
        self.logger.info("Performing atomic-level analysis")
        atomic_results = {}
        
        for method in analysis_methods:
            try:
                atomic_analysis = self.atomic_analyzer.analyze_atomic_contributions(
                    input_data, structure, method
                )
                atomic_results[method] = atomic_analysis
            except Exception as e:
                self.logger.error(f"Atomic analysis failed for method {method}: {e}")
                atomic_results[method] = {'error': str(e)}
        
        comprehensive_results['atomic_analysis'] = atomic_results
        
        # 3. 化学键分析
        self.logger.info("Performing chemical bond analysis")
        try:
            bond_analysis = self.bond_analyzer.analyze_bond_contributions(input_data, structure)
            comprehensive_results['bond_analysis'] = bond_analysis
        except Exception as e:
            self.logger.error(f"Bond analysis failed: {e}")
            comprehensive_results['bond_analysis'] = {'error': str(e)}
        
        # 4. 可视化
        self.logger.info("Creating visualizations")
        try:
            # 选择最佳的原子分析结果用于可视化
            best_atomic_analysis = None
            for method in analysis_methods:
                if method in atomic_results and 'error' not in atomic_results[method]:
                    best_atomic_analysis = atomic_results[method]
                    break
            
            visualization_results = self.visualizer.create_comprehensive_visualization(
                input_data, structure, best_atomic_analysis, 
                comprehensive_results['bond_analysis'], save_path
            )
            comprehensive_results['visualization_results'] = visualization_results
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            comprehensive_results['visualization_results'] = {'error': str(e)}
        
        # 5. 生成洞察和建议
        insights = self._generate_insights(comprehensive_results, structure)
        comprehensive_results['summary_insights'] = insights['insights']
        comprehensive_results['recommendations'] = insights['recommendations']
        
        # 6. 保存结果
        if save_path:
            self._save_analysis_results(comprehensive_results, save_path)
        
        self.logger.info("Comprehensive interpretability analysis completed")
        
        return comprehensive_results
    
    def _generate_insights(self, results: Dict[str, Any], 
                         structure: Optional[Structure]) -> Dict[str, Any]:
        """生成洞察和建议"""
        insights = {
            'insights': {},
            'recommendations': []
        }
        
        # 分析原子贡献
        atomic_insights = self._analyze_atomic_insights(results.get('atomic_analysis', {}))
        insights['insights']['atomic_contributions'] = atomic_insights
        
        # 分析化学键贡献
        bond_insights = self._analyze_bond_insights(results.get('bond_analysis', {}))
        insights['insights']['chemical_bonds'] = bond_insights
        
        # 生成建议
        recommendations = []
        
        # 基于原子分析的建议
        if atomic_insights.get('dominant_atoms'):
            recommendations.append(
                f"关注原子 {atomic_insights['dominant_atoms']} 的局部环境，"
                f"它们对预测结果贡献最大"
            )
        
        if atomic_insights.get('important_features'):
            top_features = atomic_insights['important_features'][:3]
            recommendations.append(
                f"重点考虑特征：{', '.join(top_features)}，"
                f"这些特征对材料性质影响最显著"
            )
        
        # 基于化学键分析的建议
        if bond_insights.get('critical_bonds'):
            recommendations.append(
                f"关键化学键类型：{bond_insights['critical_bonds']}，"
                f"优化这些键的强度可能改善材料性能"
            )
        
        # 基于结构的建议
        if structure is not None:
            if len(structure) > 50:
                recommendations.append(
                    "结构较大，建议关注局部配位环境而非全局结构特征"
                )
            
            # 检查是否有特殊元素
            elements = set([site.specie.symbol for site in structure])
            if 'Li' in elements:
                recommendations.append(
                    "检测到锂离子，建议分析Li离子扩散路径和空位形成能"
                )
        
        insights['recommendations'] = recommendations
        
        return insights
    
    def _analyze_atomic_insights(self, atomic_analysis: Dict) -> Dict[str, Any]:
        """分析原子洞察"""
        insights = {}
        
        # 寻找主导原子
        for method, analysis in atomic_analysis.items():
            if 'error' in analysis:
                continue
            
            if 'atom_contributions' in analysis:
                contributions = analysis['atom_contributions']
                sorted_atoms = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
                
                insights['dominant_atoms'] = [atom for atom, _ in sorted_atoms[:3]]
                insights['contribution_range'] = {
                    'max': sorted_atoms[0][1] if sorted_atoms else 0,
                    'min': sorted_atoms[-1][1] if sorted_atoms else 0
                }
            
            if 'atom_feature_importance' in analysis:
                feature_importance = analysis['atom_feature_importance']
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                insights['important_features'] = [feat for feat, _ in sorted_features[:5]]
                insights['feature_concentration'] = len([f for f, score in sorted_features if score > 0.1])
            
            break  # 只分析第一个成功的方法
        
        return insights
    
    def _analyze_bond_insights(self, bond_analysis: Dict) -> Dict[str, Any]:
        """分析化学键洞察"""
        insights = {}
        
        if 'bond_type_contributions' in bond_analysis:
            bond_types = bond_analysis['bond_type_contributions']
            
            # 找到最重要的键类型
            if bond_types:
                sorted_bonds = sorted(bond_types.items(), 
                                    key=lambda x: x[1].get('avg_contribution', 0), reverse=True)
                
                insights['critical_bonds'] = [bond for bond, _ in sorted_bonds[:3]]
                insights['bond_diversity'] = len(bond_types)
        
        if 'distance_effect_analysis' in bond_analysis:
            distance_analysis = bond_analysis['distance_effect_analysis']
            correlation = distance_analysis.get('correlation_coefficient', 0)
            
            if abs(correlation) > 0.5:
                insights['distance_correlation'] = {
                    'strength': 'strong' if abs(correlation) > 0.7 else 'moderate',
                    'direction': 'positive' if correlation > 0 else 'negative',
                    'value': correlation
                }
        
        return insights
    
    def _save_analysis_results(self, results: Dict[str, Any], save_path: str):
        """保存分析结果"""
        import os
        import json
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 保存JSON结果（移除不可序列化的内容）
        serializable_results = self._make_serializable(results)
        
        json_path = os.path.join(save_path, 'interpretability_results.json')
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # 生成文本报告
        report = self._generate_text_report(results)
        report_path = os.path.join(save_path, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Analysis results saved to {save_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items() 
                   if not k.startswith('visualization')}  # 跳过可视化对象
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return str(obj)  # 复杂对象转为字符串
        else:
            return obj
    
    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """生成文本报告"""
        report = "=== 材料性质预测可解释性分析报告 ===\n\n"
        
        # 预测结果
        prediction = results.get('model_prediction', {})
        report += f"模型预测值: {prediction.get('value', 'N/A'):.4f}\n"
        report += f"预测置信度: {prediction.get('confidence', 'N/A')}\n\n"
        
        # 原子分析摘要
        report += "=== 原子级分析 ===\n"
        atomic_analysis = results.get('atomic_analysis', {})
        
        for method, analysis in atomic_analysis.items():
            if 'error' in analysis:
                continue
            
            report += f"\n{method.upper()} 方法分析结果:\n"
            
            if 'summary_statistics' in analysis:
                stats = analysis['summary_statistics']
                report += f"  总重要性分数: {stats.get('total_importance', 0):.4f}\n"
                report += f"  最大原子贡献: {stats.get('max_atom_contribution', 0):.4f}\n"
                report += f"  最小原子贡献: {stats.get('min_atom_contribution', 0):.4f}\n"
                
                if 'top_features' in stats:
                    report += "  前5个重要特征:\n"
                    for i, (feature, score) in enumerate(stats['top_features'][:5]):
                        report += f"    {i+1}. {feature}: {score:.4f}\n"
        
        # 化学键分析摘要
        report += "\n=== 化学键分析 ===\n"
        bond_analysis = results.get('bond_analysis', {})
        
        if 'bond_type_contributions' in bond_analysis:
            bond_types = bond_analysis['bond_type_contributions']
            report += f"识别出 {len(bond_types)} 种化学键类型\n"
            
            # 列出前3种最重要的键
            sorted_bonds = sorted(bond_types.items(), 
                                key=lambda x: x[1].get('avg_contribution', 0), reverse=True)
            
            report += "最重要的化学键类型:\n"
            for i, (bond_type, info) in enumerate(sorted_bonds[:3]):
                report += f"  {i+1}. {bond_type}: 平均贡献 {info.get('avg_contribution', 0):.4f}\n"
        
        # 洞察和建议
        report += "\n=== 主要洞察 ===\n"
        insights = results.get('summary_insights', {})
        
        if 'atomic_contributions' in insights:
            atomic_insights = insights['atomic_contributions']
            if 'dominant_atoms' in atomic_insights:
                report += f"主导原子: {', '.join(atomic_insights['dominant_atoms'])}\n"
        
        if 'chemical_bonds' in insights:
            bond_insights = insights['chemical_bonds']
            if 'critical_bonds' in bond_insights:
                report += f"关键化学键: {', '.join(bond_insights['critical_bonds'])}\n"
        
        # 建议
        recommendations = results.get('recommendations', [])
        if recommendations:
            report += "\n=== 优化建议 ===\n"
            for i, rec in enumerate(recommendations):
                report += f"{i+1}. {rec}\n"
        
        return report


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
    
    # 创建综合可解释性系统
    interpretability_system = ComprehensiveInterpretabilitySystem(model)
    
    # 模拟输入数据
    batch_size = 1
    n_atoms = 20
    max_neighbors = 12
    
    atom_fea = torch.randn(n_atoms, 92)
    nbr_fea = torch.randn(n_atoms, max_neighbors, 41)
    nbr_fea_idx = torch.randint(0, n_atoms, (n_atoms, max_neighbors))
    crystal_atom_idx = [torch.arange(n_atoms)]
    
    input_data = (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
    
    # 运行综合分析
    print("Running comprehensive interpretability analysis...")
    results = interpretability_system.comprehensive_analysis(
        input_data=input_data,
        structure=None,  # 实际使用中应提供Structure对象
        analysis_methods=['integrated_gradients', 'occlusion'],
        save_path='interpretability_results'
    )
    
    # 显示结果摘要
    print(f"\nModel Prediction: {results['model_prediction']['value']:.4f}")
    
    print("\nTop Insights:")
    for category, insights in results['summary_insights'].items():
        print(f"  {category}: {insights}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(results['recommendations']):
        print(f"  {i+1}. {rec}")
    
    print("\nDetailed results saved to 'interpretability_results' directory")


if __name__ == "__main__":
    example_usage() 