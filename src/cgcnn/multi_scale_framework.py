"""
Multi-Scale Modeling Framework

Comprehensive multi-scale modeling framework integrating atomic-level,
nanoscale, and macroscopic material properties with cross-scale coupling
effects, particularly focusing on battery cycling processes.

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
import json

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .enhanced_model import EnhancedCGCNN
from .physics import PhysicsConstrainedLoss


class AtomicScaleModule(nn.Module):
    """
    原子尺度模块
    Atomic Scale Module
    """
    
    def __init__(self, atom_fea_len: int = 64, nbr_fea_len: int = 41,
                 n_conv: int = 3, h_fea_len: int = 128):
        super(AtomicScaleModule, self).__init__()
        
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        
        # 原子特征嵌入
        self.atom_embedding = nn.Linear(92, atom_fea_len)  # 92维原子特征到64维
        self.nbr_embedding = nn.Linear(nbr_fea_len, atom_fea_len)
        
        # 图卷积层
        self.conv_layers = nn.ModuleList([
            AtomicConvLayer(atom_fea_len) for _ in range(n_conv)
        ])
        
        # 原子级特征提取
        self.atomic_feature_extractor = nn.Sequential(
            nn.Linear(atom_fea_len, h_fea_len),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h_fea_len, h_fea_len // 2)
        )
        
        # 原子级物理量预测
        self.atomic_properties = nn.ModuleDict({
            'bond_strength': nn.Linear(h_fea_len // 2, 1),
            'local_charge': nn.Linear(h_fea_len // 2, 1),
            'coordination_energy': nn.Linear(h_fea_len // 2, 1),
            'migration_barrier': nn.Linear(h_fea_len // 2, 1)
        })
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, atom_fea: torch.Tensor, nbr_fea: torch.Tensor,
                nbr_fea_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            atom_fea: 原子特征 [N_atoms, 92]
            nbr_fea: 邻居特征 [N_atoms, max_neighbors, nbr_fea_len]
            nbr_fea_idx: 邻居索引 [N_atoms, max_neighbors]
            
        Returns:
            atomic_outputs: 原子尺度输出
        """
        # 特征嵌入
        atom_embedded = self.atom_embedding(atom_fea)  # [N_atoms, atom_fea_len]
        nbr_embedded = self.nbr_embedding(nbr_fea)     # [N_atoms, max_neighbors, atom_fea_len]
        
        # 图卷积
        atom_features = atom_embedded
        for conv_layer in self.conv_layers:
            atom_features = conv_layer(atom_features, nbr_embedded, nbr_fea_idx)
        
        # 原子级特征提取
        atomic_features = self.atomic_feature_extractor(atom_features)
        
        # 预测原子级物理量
        atomic_outputs = {}
        for prop_name, predictor in self.atomic_properties.items():
            atomic_outputs[prop_name] = predictor(atomic_features)
        
        # 添加原子特征用于上层模块
        atomic_outputs['atomic_features'] = atomic_features
        atomic_outputs['raw_atom_features'] = atom_features
        
        return atomic_outputs


class AtomicConvLayer(nn.Module):
    """原子卷积层"""
    
    def __init__(self, atom_fea_len: int):
        super(AtomicConvLayer, self).__init__()
        
        self.atom_fea_len = atom_fea_len
        
        # 消息传递网络
        self.message_net = nn.Sequential(
            nn.Linear(2 * atom_fea_len, atom_fea_len),
            nn.ReLU(),
            nn.Linear(atom_fea_len, atom_fea_len)
        )
        
        # 更新网络
        self.update_net = nn.Sequential(
            nn.Linear(2 * atom_fea_len, atom_fea_len),
            nn.ReLU(),
            nn.Linear(atom_fea_len, atom_fea_len)
        )
        
        self.activation = nn.ReLU()
    
    def forward(self, atom_features: torch.Tensor, nbr_features: torch.Tensor,
                nbr_indices: torch.Tensor) -> torch.Tensor:
        """
        图卷积前向传播
        
        Args:
            atom_features: [N_atoms, atom_fea_len]
            nbr_features: [N_atoms, max_neighbors, atom_fea_len]
            nbr_indices: [N_atoms, max_neighbors]
        """
        N_atoms, max_neighbors = nbr_indices.shape
        
        # 收集邻居特征
        neighbor_atom_features = atom_features[nbr_indices.view(-1)].view(
            N_atoms, max_neighbors, -1
        )
        
        # 构建消息
        atom_expanded = atom_features.unsqueeze(1).expand(-1, max_neighbors, -1)
        combined_features = torch.cat([atom_expanded, neighbor_atom_features], dim=-1)
        
        # 消息传递
        messages = self.message_net(combined_features)  # [N_atoms, max_neighbors, atom_fea_len]
        
        # 聚合消息
        aggregated_messages = torch.sum(messages, dim=1)  # [N_atoms, atom_fea_len]
        
        # 更新原子特征
        update_input = torch.cat([atom_features, aggregated_messages], dim=-1)
        updated_features = self.update_net(update_input)
        
        # 残差连接
        output = atom_features + updated_features
        
        return self.activation(output)


class NanoscaleModule(nn.Module):
    """
    纳米尺度模块
    Nanoscale Module
    """
    
    def __init__(self, atomic_feature_dim: int = 64, nano_feature_dim: int = 128):
        super(NanoscaleModule, self).__init__()
        
        self.atomic_feature_dim = atomic_feature_dim
        self.nano_feature_dim = nano_feature_dim
        
        # 原子到纳米尺度的特征聚合
        self.atomic_to_nano = nn.Sequential(
            nn.Linear(atomic_feature_dim, nano_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 纳米尺度特征处理
        self.nano_processor = nn.ModuleList([
            NanoConvLayer(nano_feature_dim) for _ in range(2)
        ])
        
        # 纳米级物理量预测
        self.nano_properties = nn.ModuleDict({
            'grain_boundary_energy': nn.Linear(nano_feature_dim, 1),
            'phase_stability': nn.Linear(nano_feature_dim, 1),
            'interface_resistance': nn.Linear(nano_feature_dim, 1),
            'particle_size_effect': nn.Linear(nano_feature_dim, 1),
            'surface_energy': nn.Linear(nano_feature_dim, 1)
        })
        
        # 多尺度耦合层
        self.coupling_layer = MultiScaleCouplingLayer(
            atomic_dim=atomic_feature_dim,
            nano_dim=nano_feature_dim
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, atomic_outputs: Dict[str, torch.Tensor],
                crystal_atom_idx: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        纳米尺度前向传播
        
        Args:
            atomic_outputs: 原子尺度输出
            crystal_atom_idx: 晶体原子索引
            
        Returns:
            nano_outputs: 纳米尺度输出
        """
        atomic_features = atomic_outputs['atomic_features']
        
        # 将原子特征聚合到纳米尺度
        nano_features = []
        for crystal_idx in crystal_atom_idx:
            # 对每个晶体的原子特征进行聚合
            crystal_atomic_features = atomic_features[crystal_idx]
            
            # 多种聚合方式
            mean_features = torch.mean(crystal_atomic_features, dim=0)
            max_features, _ = torch.max(crystal_atomic_features, dim=0)
            std_features = torch.std(crystal_atomic_features, dim=0)
            
            # 组合聚合特征
            combined_features = torch.cat([mean_features, max_features, std_features], dim=0)
            
            # 映射到纳米特征空间
            nano_feature = self.atomic_to_nano(combined_features)
            nano_features.append(nano_feature)
        
        nano_features = torch.stack(nano_features)  # [batch_size, nano_feature_dim]
        
        # 纳米尺度特征处理
        for nano_layer in self.nano_processor:
            nano_features = nano_layer(nano_features)
        
        # 多尺度耦合
        coupled_features = self.coupling_layer(atomic_outputs, nano_features)
        nano_features = coupled_features['nano_features']
        
        # 预测纳米级物理量
        nano_outputs = {}
        for prop_name, predictor in self.nano_properties.items():
            nano_outputs[prop_name] = predictor(nano_features)
        
        # 添加特征用于宏观模块
        nano_outputs['nano_features'] = nano_features
        nano_outputs['coupling_effects'] = coupled_features['coupling_effects']
        
        return nano_outputs


class NanoConvLayer(nn.Module):
    """纳米卷积层"""
    
    def __init__(self, feature_dim: int):
        super(NanoConvLayer, self).__init__()
        
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(0.1)
        )
        
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        纳米卷积前向传播
        
        Args:
            features: [batch_size, feature_dim]
        """
        # 自注意力机制
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        attn_output, _ = self.attention(features, features, features)
        attn_output = attn_output.squeeze(1)  # [batch_size, feature_dim]
        
        # 特征变换
        transformed = self.feature_transform(attn_output)
        
        # 残差连接和归一化
        if features.dim() == 3:
            features = features.squeeze(1)
        
        output = self.norm(features + transformed)
        
        return output


class MacroscopicModule(nn.Module):
    """
    宏观尺度模块
    Macroscopic Module
    """
    
    def __init__(self, nano_feature_dim: int = 128, macro_feature_dim: int = 256):
        super(MacroscopicModule, self).__init__()
        
        self.nano_feature_dim = nano_feature_dim
        self.macro_feature_dim = macro_feature_dim
        
        # 纳米到宏观尺度的特征映射
        self.nano_to_macro = nn.Sequential(
            nn.Linear(nano_feature_dim, macro_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(macro_feature_dim, macro_feature_dim)
        )
        
        # 宏观特征处理
        self.macro_processor = nn.Sequential(
            nn.Linear(macro_feature_dim, macro_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(macro_feature_dim, macro_feature_dim // 2),
            nn.ReLU()
        )
        
        # 宏观物理量预测
        self.macro_properties = nn.ModuleDict({
            'bulk_modulus': nn.Linear(macro_feature_dim // 2, 1),
            'shear_modulus': nn.Linear(macro_feature_dim // 2, 1),
            'thermal_conductivity': nn.Linear(macro_feature_dim // 2, 1),
            'electrical_conductivity': nn.Linear(macro_feature_dim // 2, 1),
            'thermal_expansion': nn.Linear(macro_feature_dim // 2, 1),
            'density': nn.Linear(macro_feature_dim // 2, 1)
        })
        
        # 电池性能预测（特殊模块）
        self.battery_performance = BatteryPerformancePredictor(macro_feature_dim // 2)
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, nano_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        宏观尺度前向传播
        
        Args:
            nano_outputs: 纳米尺度输出
            
        Returns:
            macro_outputs: 宏观尺度输出
        """
        nano_features = nano_outputs['nano_features']
        
        # 纳米到宏观特征映射
        macro_features = self.nano_to_macro(nano_features)
        
        # 宏观特征处理
        processed_features = self.macro_processor(macro_features)
        
        # 预测宏观物理量
        macro_outputs = {}
        for prop_name, predictor in self.macro_properties.items():
            macro_outputs[prop_name] = predictor(processed_features)
        
        # 电池性能预测
        battery_outputs = self.battery_performance(processed_features, nano_outputs)
        macro_outputs.update(battery_outputs)
        
        # 添加特征信息
        macro_outputs['macro_features'] = processed_features
        
        return macro_outputs


class BatteryPerformancePredictor(nn.Module):
    """
    电池性能预测器
    Battery Performance Predictor
    """
    
    def __init__(self, feature_dim: int):
        super(BatteryPerformancePredictor, self).__init__()
        
        self.feature_dim = feature_dim
        
        # 电池关键性能指标预测
        self.capacity_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        self.voltage_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        self.cycle_life_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        self.rate_capability_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # 循环过程建模
        self.cycling_dynamics = CyclingDynamicsModel(feature_dim)
    
    def forward(self, macro_features: torch.Tensor, 
                nano_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        电池性能预测
        
        Args:
            macro_features: 宏观特征
            nano_outputs: 纳米尺度输出
            
        Returns:
            battery_outputs: 电池性能输出
        """
        # 基础电池性能预测
        capacity = self.capacity_predictor(macro_features)
        voltage = self.voltage_predictor(macro_features)
        cycle_life = self.cycle_life_predictor(macro_features)
        rate_capability = self.rate_capability_predictor(macro_features)
        
        # 循环动力学建模
        cycling_outputs = self.cycling_dynamics(macro_features, nano_outputs)
        
        battery_outputs = {
            'specific_capacity': capacity,
            'average_voltage': voltage,
            'cycle_life': cycle_life,
            'rate_capability': rate_capability,
            **cycling_outputs
        }
        
        return battery_outputs


class CyclingDynamicsModel(nn.Module):
    """
    循环动力学模型
    Cycling Dynamics Model
    """
    
    def __init__(self, feature_dim: int):
        super(CyclingDynamicsModel, self).__init__()
        
        # 容量衰减建模
        self.capacity_fade_model = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 3)  # [initial_fade_rate, long_term_fade_rate, knee_point]
        )
        
        # 阻抗增长建模
        self.impedance_growth_model = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 2)  # [sei_resistance, charge_transfer_resistance]
        )
        
        # 热效应建模
        self.thermal_model = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 2)  # [heat_generation_rate, thermal_stability]
        )
    
    def forward(self, macro_features: torch.Tensor,
                nano_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        循环动力学前向传播
        
        Args:
            macro_features: 宏观特征
            nano_outputs: 纳米尺度输出
            
        Returns:
            cycling_outputs: 循环动力学输出
        """
        # 容量衰减参数
        fade_params = self.capacity_fade_model(macro_features)
        
        # 阻抗增长参数
        impedance_params = self.impedance_growth_model(macro_features)
        
        # 热效应参数
        thermal_params = self.thermal_model(macro_features)
        
        cycling_outputs = {
            'capacity_fade_rate': fade_params[:, 0:1],
            'long_term_fade_rate': fade_params[:, 1:2],
            'fade_knee_point': fade_params[:, 2:3],
            'sei_resistance_growth': impedance_params[:, 0:1],
            'charge_transfer_resistance': impedance_params[:, 1:2],
            'heat_generation_rate': thermal_params[:, 0:1],
            'thermal_stability': thermal_params[:, 1:2]
        }
        
        return cycling_outputs


class MultiScaleCouplingLayer(nn.Module):
    """
    多尺度耦合层
    Multi-Scale Coupling Layer
    """
    
    def __init__(self, atomic_dim: int, nano_dim: int):
        super(MultiScaleCouplingLayer, self).__init__()
        
        self.atomic_dim = atomic_dim
        self.nano_dim = nano_dim
        
        # 原子到纳米的耦合
        self.atomic_to_nano_coupling = nn.Sequential(
            nn.Linear(atomic_dim, nano_dim),
            nn.ReLU(),
            nn.Linear(nano_dim, nano_dim)
        )
        
        # 纳米到原子的反馈
        self.nano_to_atomic_feedback = nn.Sequential(
            nn.Linear(nano_dim, atomic_dim),
            nn.ReLU(),
            nn.Linear(atomic_dim, atomic_dim)
        )
        
        # 耦合强度学习
        self.coupling_strength = nn.Parameter(torch.ones(1))
        
        # 注意力机制用于选择性耦合
        self.coupling_attention = nn.MultiheadAttention(
            embed_dim=max(atomic_dim, nano_dim),
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, atomic_outputs: Dict[str, torch.Tensor],
                nano_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        多尺度耦合前向传播
        
        Args:
            atomic_outputs: 原子尺度输出
            nano_features: 纳米特征
            
        Returns:
            coupling_results: 耦合结果
        """
        atomic_features = atomic_outputs['atomic_features']
        
        # 计算耦合效应
        # 原子影响纳米
        atomic_influence = self.atomic_to_nano_coupling(
            torch.mean(atomic_features, dim=0, keepdim=True)
        )
        
        # 纳米反馈到原子
        nano_feedback = self.nano_to_atomic_feedback(nano_features)
        nano_feedback_expanded = nano_feedback.unsqueeze(1).expand(-1, atomic_features.size(0), -1)
        
        # 应用耦合
        coupled_nano_features = nano_features + self.coupling_strength * atomic_influence.squeeze(0)
        
        # 计算耦合效应强度
        coupling_effects = torch.norm(atomic_influence, dim=-1) + torch.norm(nano_feedback, dim=-1)
        
        coupling_results = {
            'nano_features': coupled_nano_features,
            'coupling_effects': coupling_effects,
            'atomic_influence': atomic_influence,
            'nano_feedback': nano_feedback
        }
        
        return coupling_results


class MultiScaleFramework(nn.Module):
    """
    多尺度建模框架
    Multi-Scale Modeling Framework
    """
    
    def __init__(self, orig_atom_fea_len: int = 92, nbr_fea_len: int = 41,
                 atom_fea_len: int = 64, nano_feature_dim: int = 128,
                 macro_feature_dim: int = 256, n_conv: int = 3):
        super(MultiScaleFramework, self).__init__()
        
        # 各尺度模块
        self.atomic_module = AtomicScaleModule(
            atom_fea_len=atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            n_conv=n_conv
        )
        
        self.nanoscale_module = NanoscaleModule(
            atomic_feature_dim=atom_fea_len // 2,
            nano_feature_dim=nano_feature_dim
        )
        
        self.macroscopic_module = MacroscopicModule(
            nano_feature_dim=nano_feature_dim,
            macro_feature_dim=macro_feature_dim
        )
        
        # 多尺度融合层
        self.scale_fusion = MultiScaleFusionLayer(
            atomic_dim=atom_fea_len // 2,
            nano_dim=nano_feature_dim,
            macro_dim=macro_feature_dim // 2
        )
        
        # 物理约束
        self.physics_constraints = PhysicsConstrainedLoss()
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, atom_fea: torch.Tensor, nbr_fea: torch.Tensor,
                nbr_fea_idx: torch.Tensor, crystal_atom_idx: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        多尺度前向传播
        
        Args:
            atom_fea: 原子特征
            nbr_fea: 邻居特征
            nbr_fea_idx: 邻居索引
            crystal_atom_idx: 晶体原子索引
            
        Returns:
            multi_scale_outputs: 多尺度输出
        """
        # 1. 原子尺度建模
        atomic_outputs = self.atomic_module(atom_fea, nbr_fea, nbr_fea_idx)
        
        # 2. 纳米尺度建模
        nano_outputs = self.nanoscale_module(atomic_outputs, crystal_atom_idx)
        
        # 3. 宏观尺度建模
        macro_outputs = self.macroscopic_module(nano_outputs)
        
        # 4. 多尺度融合
        fused_outputs = self.scale_fusion(atomic_outputs, nano_outputs, macro_outputs)
        
        # 整合所有输出
        multi_scale_outputs = {
            'atomic_scale': atomic_outputs,
            'nanoscale': nano_outputs,
            'macroscopic': macro_outputs,
            'fused_predictions': fused_outputs,
            'primary_prediction': fused_outputs['final_prediction']
        }
        
        return multi_scale_outputs
    
    def compute_multi_scale_loss(self, outputs: Dict[str, torch.Tensor],
                                targets: Dict[str, torch.Tensor],
                                loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        计算多尺度损失
        
        Args:
            outputs: 模型输出
            targets: 目标值
            loss_weights: 损失权重
            
        Returns:
            loss_dict: 损失字典
        """
        if loss_weights is None:
            loss_weights = {
                'atomic': 0.3,
                'nano': 0.3,
                'macro': 0.4,
                'physics': 0.1
            }
        
        loss_dict = {}
        total_loss = 0.0
        
        # 原子尺度损失
        if 'atomic_targets' in targets:
            atomic_loss = self._compute_atomic_loss(
                outputs['atomic_scale'], targets['atomic_targets']
            )
            loss_dict['atomic_loss'] = atomic_loss
            total_loss += loss_weights['atomic'] * atomic_loss
        
        # 纳米尺度损失
        if 'nano_targets' in targets:
            nano_loss = self._compute_nano_loss(
                outputs['nanoscale'], targets['nano_targets']
            )
            loss_dict['nano_loss'] = nano_loss
            total_loss += loss_weights['nano'] * nano_loss
        
        # 宏观尺度损失
        if 'macro_targets' in targets:
            macro_loss = self._compute_macro_loss(
                outputs['macroscopic'], targets['macro_targets']
            )
            loss_dict['macro_loss'] = macro_loss
            total_loss += loss_weights['macro'] * macro_loss
        
        # 物理约束损失
        physics_loss, _ = self.physics_constraints(
            outputs['primary_prediction'],
            targets.get('primary_target', torch.zeros_like(outputs['primary_prediction'])),
            self,
            (None, None, None, None)  # 简化的输入
        )
        loss_dict['physics_loss'] = physics_loss
        total_loss += loss_weights['physics'] * physics_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def _compute_atomic_loss(self, atomic_outputs: Dict[str, torch.Tensor],
                           atomic_targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算原子尺度损失"""
        atomic_loss = 0.0
        n_properties = 0
        
        for prop_name in ['bond_strength', 'local_charge', 'coordination_energy']:
            if prop_name in atomic_outputs and prop_name in atomic_targets:
                loss = F.mse_loss(atomic_outputs[prop_name], atomic_targets[prop_name])
                atomic_loss += loss
                n_properties += 1
        
        return atomic_loss / max(n_properties, 1)
    
    def _compute_nano_loss(self, nano_outputs: Dict[str, torch.Tensor],
                          nano_targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算纳米尺度损失"""
        nano_loss = 0.0
        n_properties = 0
        
        for prop_name in ['grain_boundary_energy', 'phase_stability', 'interface_resistance']:
            if prop_name in nano_outputs and prop_name in nano_targets:
                loss = F.mse_loss(nano_outputs[prop_name], nano_targets[prop_name])
                nano_loss += loss
                n_properties += 1
        
        return nano_loss / max(n_properties, 1)
    
    def _compute_macro_loss(self, macro_outputs: Dict[str, torch.Tensor],
                           macro_targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算宏观尺度损失"""
        macro_loss = 0.0
        n_properties = 0
        
        for prop_name in ['bulk_modulus', 'thermal_conductivity', 'specific_capacity']:
            if prop_name in macro_outputs and prop_name in macro_targets:
                loss = F.mse_loss(macro_outputs[prop_name], macro_targets[prop_name])
                macro_loss += loss
                n_properties += 1
        
        return macro_loss / max(n_properties, 1)


class MultiScaleFusionLayer(nn.Module):
    """
    多尺度融合层
    Multi-Scale Fusion Layer
    """
    
    def __init__(self, atomic_dim: int, nano_dim: int, macro_dim: int):
        super(MultiScaleFusionLayer, self).__init__()
        
        self.atomic_dim = atomic_dim
        self.nano_dim = nano_dim
        self.macro_dim = macro_dim
        
        # 特征维度统一
        total_dim = atomic_dim + nano_dim + macro_dim
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(total_dim // 2, total_dim // 4),
            nn.ReLU()
        )
        
        # 最终预测层
        self.final_predictor = nn.Linear(total_dim // 4, 1)
        
        # 尺度权重学习
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)  # [atomic, nano, macro]
        
        # 注意力融合
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=total_dim // 4,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, atomic_outputs: Dict[str, torch.Tensor],
                nano_outputs: Dict[str, torch.Tensor],
                macro_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        多尺度融合前向传播
        
        Args:
            atomic_outputs: 原子尺度输出
            nano_outputs: 纳米尺度输出
            macro_outputs: 宏观尺度输出
            
        Returns:
            fusion_outputs: 融合输出
        """
        # 提取各尺度特征
        atomic_features = torch.mean(atomic_outputs['atomic_features'], dim=0, keepdim=True)
        nano_features = nano_outputs['nano_features']
        macro_features = macro_outputs['macro_features']
        
        # 确保批次维度一致
        batch_size = nano_features.size(0)
        atomic_features = atomic_features.expand(batch_size, -1)
        
        # 特征拼接
        fused_features = torch.cat([atomic_features, nano_features, macro_features], dim=-1)
        
        # 特征融合
        processed_features = self.feature_fusion(fused_features)
        
        # 注意力融合（可选）
        if processed_features.dim() == 2:
            processed_features = processed_features.unsqueeze(1)
        
        attended_features, attention_weights = self.attention_fusion(
            processed_features, processed_features, processed_features
        )
        attended_features = attended_features.squeeze(1)
        
        # 最终预测
        final_prediction = self.final_predictor(attended_features)
        
        # 计算尺度贡献
        scale_contributions = {
            'atomic_weight': self.scale_weights[0].item(),
            'nano_weight': self.scale_weights[1].item(),
            'macro_weight': self.scale_weights[2].item()
        }
        
        fusion_outputs = {
            'final_prediction': final_prediction,
            'fused_features': attended_features,
            'scale_contributions': scale_contributions,
            'attention_weights': attention_weights.squeeze(1) if attention_weights.dim() > 2 else attention_weights
        }
        
        return fusion_outputs


class MultiScaleAnalyzer:
    """
    多尺度分析器
    Multi-Scale Analyzer
    """
    
    def __init__(self, framework: MultiScaleFramework):
        self.framework = framework
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_scale_contributions(self, input_data: Tuple,
                                  targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        分析各尺度贡献
        
        Args:
            input_data: 输入数据
            targets: 目标值（可选）
            
        Returns:
            analysis_results: 分析结果
        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 获取多尺度输出
        with torch.no_grad():
            outputs = self.framework(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        
        analysis_results = {
            'scale_predictions': {},
            'scale_contributions': {},
            'coupling_analysis': {},
            'property_correlations': {}
        }
        
        # 1. 各尺度预测分析
        analysis_results['scale_predictions'] = self._analyze_scale_predictions(outputs)
        
        # 2. 尺度贡献分析
        analysis_results['scale_contributions'] = self._analyze_scale_contributions(outputs)
        
        # 3. 耦合效应分析
        analysis_results['coupling_analysis'] = self._analyze_coupling_effects(outputs)
        
        # 4. 属性相关性分析
        analysis_results['property_correlations'] = self._analyze_property_correlations(outputs)
        
        return analysis_results
    
    def _analyze_scale_predictions(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析各尺度预测"""
        scale_predictions = {}
        
        # 原子尺度预测
        atomic_preds = {}
        for prop, values in outputs['atomic_scale'].items():
            if prop != 'atomic_features' and prop != 'raw_atom_features':
                atomic_preds[prop] = {
                    'mean': torch.mean(values).item(),
                    'std': torch.std(values).item(),
                    'range': (torch.min(values).item(), torch.max(values).item())
                }
        scale_predictions['atomic'] = atomic_preds
        
        # 纳米尺度预测
        nano_preds = {}
        for prop, values in outputs['nanoscale'].items():
            if prop not in ['nano_features', 'coupling_effects']:
                nano_preds[prop] = {
                    'value': values.item() if values.numel() == 1 else torch.mean(values).item(),
                    'magnitude': torch.norm(values).item()
                }
        scale_predictions['nanoscale'] = nano_preds
        
        # 宏观尺度预测
        macro_preds = {}
        for prop, values in outputs['macroscopic'].items():
            if prop != 'macro_features':
                macro_preds[prop] = {
                    'value': values.item() if values.numel() == 1 else torch.mean(values).item(),
                    'magnitude': torch.norm(values).item()
                }
        scale_predictions['macroscopic'] = macro_preds
        
        return scale_predictions
    
    def _analyze_scale_contributions(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析尺度贡献"""
        contributions = outputs['fused_predictions']['scale_contributions']
        
        contribution_analysis = {
            'weights': contributions,
            'dominant_scale': max(contributions.items(), key=lambda x: x[1])[0],
            'weight_distribution': {
                'entropy': self._calculate_entropy(list(contributions.values())),
                'balance': min(contributions.values()) / max(contributions.values())
            }
        }
        
        return contribution_analysis
    
    def _analyze_coupling_effects(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析耦合效应"""
        coupling_effects = outputs['nanoscale']['coupling_effects']
        
        coupling_analysis = {
            'coupling_strength': torch.mean(coupling_effects).item(),
            'coupling_variability': torch.std(coupling_effects).item(),
            'max_coupling': torch.max(coupling_effects).item(),
            'coupling_distribution': coupling_effects.detach().cpu().numpy().tolist()
        }
        
        return coupling_analysis
    
    def _analyze_property_correlations(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析属性相关性"""
        # 收集所有数值属性
        all_properties = {}
        
        # 原子尺度属性
        for prop, values in outputs['atomic_scale'].items():
            if prop not in ['atomic_features', 'raw_atom_features']:
                all_properties[f'atomic_{prop}'] = torch.mean(values).item()
        
        # 纳米尺度属性
        for prop, values in outputs['nanoscale'].items():
            if prop not in ['nano_features', 'coupling_effects']:
                all_properties[f'nano_{prop}'] = values.item() if values.numel() == 1 else torch.mean(values).item()
        
        # 宏观尺度属性
        for prop, values in outputs['macroscopic'].items():
            if prop != 'macro_features':
                all_properties[f'macro_{prop}'] = values.item() if values.numel() == 1 else torch.mean(values).item()
        
        # 计算相关性（简化版本）
        correlations = {}
        property_names = list(all_properties.keys())
        
        for i, prop1 in enumerate(property_names):
            for prop2 in property_names[i+1:]:
                # 简化的相关性计算（实际应该用多个样本）
                corr_key = f"{prop1}_vs_{prop2}"
                correlations[corr_key] = abs(all_properties[prop1] - all_properties[prop2])
        
        correlation_analysis = {
            'property_values': all_properties,
            'correlations': correlations,
            'n_properties': len(all_properties)
        }
        
        return correlation_analysis
    
    def _calculate_entropy(self, values: List[float]) -> float:
        """计算熵"""
        values = np.array(values)
        values = values / np.sum(values)  # 归一化
        entropy = -np.sum(values * np.log(values + 1e-8))
        return entropy
    
    def generate_multi_scale_report(self, analysis_results: Dict[str, Any],
                                  save_path: Optional[str] = None) -> str:
        """生成多尺度分析报告"""
        report = "=== 多尺度材料建模分析报告 ===\n\n"
        
        # 1. 各尺度预测摘要
        report += "=== 各尺度预测结果 ===\n"
        
        scale_preds = analysis_results['scale_predictions']
        
        report += "\n原子尺度预测:\n"
        for prop, stats in scale_preds.get('atomic', {}).items():
            report += f"  {prop}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}\n"
        
        report += "\n纳米尺度预测:\n"
        for prop, stats in scale_preds.get('nanoscale', {}).items():
            report += f"  {prop}: 值={stats['value']:.4f}, 幅度={stats['magnitude']:.4f}\n"
        
        report += "\n宏观尺度预测:\n"
        for prop, stats in scale_preds.get('macroscopic', {}).items():
            report += f"  {prop}: 值={stats['value']:.4f}, 幅度={stats['magnitude']:.4f}\n"
        
        # 2. 尺度贡献分析
        report += "\n=== 尺度贡献分析 ===\n"
        contrib = analysis_results['scale_contributions']
        
        report += f"主导尺度: {contrib['dominant_scale']}\n"
        report += "各尺度权重:\n"
        for scale, weight in contrib['weights'].items():
            report += f"  {scale}: {weight:.4f}\n"
        
        report += f"权重熵: {contrib['weight_distribution']['entropy']:.4f}\n"
        report += f"权重平衡度: {contrib['weight_distribution']['balance']:.4f}\n"
        
        # 3. 耦合效应分析
        report += "\n=== 多尺度耦合分析 ===\n"
        coupling = analysis_results['coupling_analysis']
        
        report += f"平均耦合强度: {coupling['coupling_strength']:.4f}\n"
        report += f"耦合变异性: {coupling['coupling_variability']:.4f}\n"
        report += f"最大耦合强度: {coupling['max_coupling']:.4f}\n"
        
        # 4. 属性相关性
        report += "\n=== 跨尺度属性相关性 ===\n"
        correlations = analysis_results['property_correlations']
        
        report += f"总属性数量: {correlations['n_properties']}\n"
        report += "主要属性值:\n"
        
        # 显示前10个属性
        sorted_props = sorted(correlations['property_values'].items(), 
                            key=lambda x: abs(x[1]), reverse=True)[:10]
        
        for prop, value in sorted_props:
            report += f"  {prop}: {value:.4f}\n"
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Multi-scale analysis report saved to {save_path}")
        
        return report


# 使用示例
def example_usage():
    """使用示例"""
    # 创建多尺度框架
    framework = MultiScaleFramework(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=64,
        nano_feature_dim=128,
        macro_feature_dim=256,
        n_conv=3
    )
    
    # 模拟输入数据
    batch_size = 2
    n_atoms = 30
    max_neighbors = 12
    
    atom_fea = torch.randn(n_atoms, 92)
    nbr_fea = torch.randn(n_atoms, max_neighbors, 41)
    nbr_fea_idx = torch.randint(0, n_atoms, (n_atoms, max_neighbors))
    crystal_atom_idx = [torch.arange(15), torch.arange(15, 30)]
    
    input_data = (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
    
    # 前向传播
    print("Running multi-scale forward pass...")
    outputs = framework(*input_data)
    
    # 显示输出结构
    print(f"Primary prediction: {outputs['primary_prediction'].item():.4f}")
    print(f"Atomic scale properties: {len(outputs['atomic_scale'])}")
    print(f"Nanoscale properties: {len(outputs['nanoscale'])}")
    print(f"Macroscopic properties: {len(outputs['macroscopic'])}")
    
    # 多尺度分析
    analyzer = MultiScaleAnalyzer(framework)
    
    print("\nPerforming multi-scale analysis...")
    analysis_results = analyzer.analyze_scale_contributions(input_data)
    
    # 生成报告
    report = analyzer.generate_multi_scale_report(analysis_results)
    print("\nMulti-scale Analysis Report:")
    print(report[:1000] + "..." if len(report) > 1000 else report)
    
    # 损失计算示例
    print("\nTesting multi-scale loss computation...")
    targets = {
        'primary_target': torch.randn(batch_size, 1),
        'macro_targets': {
            'bulk_modulus': torch.randn(batch_size, 1),
            'thermal_conductivity': torch.randn(batch_size, 1)
        }
    }
    
    loss_dict = framework.compute_multi_scale_loss(outputs, targets)
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    for loss_name, loss_value in loss_dict.items():
        if loss_name != 'total_loss':
            print(f"  {loss_name}: {loss_value.item():.4f}")


if __name__ == "__main__":
    example_usage() 