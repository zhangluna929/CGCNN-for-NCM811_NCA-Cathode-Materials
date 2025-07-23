"""
Enhanced CGCNN Model Architecture

Detailed implementation of CGCNN with attention mechanisms,
multi-scale feature fusion, and comprehensive design rationale
for crystal structure processing.

Author: lunazhang
Date: 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import math

from .model import ConvLayer


class GraphAttentionLayer(nn.Module):
    """
    图注意力层
    Graph Attention Layer for Crystal Structures
    
    设计理念：
    - 不同原子对材料性质的贡献不同，需要自适应的注意力权重
    - 化学键的强度和类型影响信息传播，注意力机制可以学习这些关系
    - 多头注意力可以捕获不同类型的原子间相互作用
    """
    
    def __init__(self, atom_fea_len: int, nbr_fea_len: int, n_heads: int = 8):
        super(GraphAttentionLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.n_heads = n_heads
        self.head_dim = atom_fea_len // n_heads
        
        assert atom_fea_len % n_heads == 0, "atom_fea_len must be divisible by n_heads"
        
        # 多头注意力的线性变换
        self.query_transform = nn.Linear(atom_fea_len, atom_fea_len)
        self.key_transform = nn.Linear(atom_fea_len, atom_fea_len)
        self.value_transform = nn.Linear(atom_fea_len, atom_fea_len)
        
        # 邻居特征的整合
        self.neighbor_transform = nn.Linear(nbr_fea_len, atom_fea_len)
        
        # 输出投影
        self.output_projection = nn.Linear(atom_fea_len, atom_fea_len)
        
        # 归一化和激活
        self.layer_norm = nn.LayerNorm(atom_fea_len)
        self.dropout = nn.Dropout(0.1)
        
        # 门控机制用于残差连接
        self.gate = nn.Linear(atom_fea_len * 2, 1)
        
    def forward(self, atom_fea: torch.Tensor, nbr_fea: torch.Tensor, 
                nbr_fea_idx: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            atom_fea: 原子特征 [N, atom_fea_len]
            nbr_fea: 邻居特征 [N, M, nbr_fea_len]
            nbr_fea_idx: 邻居索引 [N, M]
            
        Returns:
            output: 更新后的原子特征
        """
        N, M = nbr_fea_idx.shape
        residual = atom_fea
        
        # 1. 计算查询、键、值
        queries = self.query_transform(atom_fea)  # [N, atom_fea_len]
        keys = self.key_transform(atom_fea)       # [N, atom_fea_len]
        values = self.value_transform(atom_fea)   # [N, atom_fea_len]
        
        # 2. 重塑为多头格式
        queries = queries.view(N, self.n_heads, self.head_dim)  # [N, n_heads, head_dim]
        keys = keys.view(-1, self.n_heads, self.head_dim)       # [N, n_heads, head_dim]
        values = values.view(-1, self.n_heads, self.head_dim)   # [N, n_heads, head_dim]
        
        # 3. 获取邻居的键和值
        neighbor_keys = keys[nbr_fea_idx]     # [N, M, n_heads, head_dim]
        neighbor_values = values[nbr_fea_idx] # [N, M, n_heads, head_dim]
        
        # 4. 整合邻居特征信息
        neighbor_fea_transformed = self.neighbor_transform(nbr_fea)  # [N, M, atom_fea_len]
        neighbor_fea_reshaped = neighbor_fea_transformed.view(N, M, self.n_heads, self.head_dim)
        
        # 将邻居化学信息融入键和值
        neighbor_keys = neighbor_keys + neighbor_fea_reshaped
        neighbor_values = neighbor_values + neighbor_fea_reshaped
        
        # 5. 计算注意力分数
        queries_expanded = queries.unsqueeze(2)  # [N, n_heads, 1, head_dim]
        attention_scores = torch.sum(queries_expanded * neighbor_keys, dim=-1)  # [N, n_heads, M]
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # 6. 注意力归一化
        attention_weights = F.softmax(attention_scores, dim=-1)  # [N, n_heads, M]
        attention_weights = self.dropout(attention_weights)
        
        # 7. 加权聚合
        attended_values = torch.sum(
            attention_weights.unsqueeze(-1) * neighbor_values, dim=2
        )  # [N, n_heads, head_dim]
        
        # 8. 重塑并投影
        attended_output = attended_values.view(N, self.atom_fea_len)
        output = self.output_projection(attended_output)
        
        # 9. 门控残差连接
        gate_input = torch.cat([output, residual], dim=-1)
        gate_weight = torch.sigmoid(self.gate(gate_input))
        output = gate_weight * output + (1 - gate_weight) * residual
        
        # 10. 层归一化
        output = self.layer_norm(output)
        
        return output


class MultiScaleConvLayer(nn.Module):
    """
    多尺度卷积层
    Multi-Scale Convolution Layer
    
    设计理念：
    - 材料性质受多个尺度的结构特征影响：局部配位、中程有序、长程周期性
    - 不同的卷积核可以捕获不同距离范围的原子间相互作用
    - 多尺度特征融合可以提供更全面的结构表示
    """
    
    def __init__(self, atom_fea_len: int, nbr_fea_len: int, 
                 scales: List[int] = [1, 2, 3]):
        super(MultiScaleConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.scales = scales
        
        # 为每个尺度创建卷积层
        self.scale_convs = nn.ModuleList([
            ConvLayer(atom_fea_len, nbr_fea_len) for _ in scales
        ])
        
        # 尺度融合层
        self.scale_fusion = nn.Linear(len(scales) * atom_fea_len, atom_fea_len)
        self.fusion_activation = nn.ReLU()
        
        # 自适应权重学习
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        
    def forward(self, atom_fea: torch.Tensor, nbr_fea: torch.Tensor, 
                nbr_fea_idx: torch.Tensor) -> torch.Tensor:
        """
        多尺度前向传播
        
        Args:
            atom_fea: 原子特征
            nbr_fea: 邻居特征
            nbr_fea_idx: 邻居索引
            
        Returns:
            fused_output: 多尺度融合后的特征
        """
        scale_outputs = []
        
        # 对每个尺度进行卷积
        for i, (scale, conv_layer) in enumerate(zip(self.scales, self.scale_convs)):
            # 根据尺度调整邻居特征
            if scale > 1:
                # 实现多尺度邻居采样的简化版本
                # 实际应用中可能需要更复杂的多尺度邻居构建
                scaled_nbr_fea = self._apply_scale_transform(nbr_fea, scale)
            else:
                scaled_nbr_fea = nbr_fea
            
            # 卷积操作
            scale_output = conv_layer(atom_fea, scaled_nbr_fea, nbr_fea_idx)
            scale_outputs.append(scale_output)
        
        # 加权融合不同尺度的特征
        weighted_outputs = []
        scale_weights_normalized = F.softmax(self.scale_weights, dim=0)
        
        for i, output in enumerate(scale_outputs):
            weighted_outputs.append(scale_weights_normalized[i] * output)
        
        # 特征连接和融合
        concatenated = torch.cat(scale_outputs, dim=-1)
        fused_output = self.scale_fusion(concatenated)
        fused_output = self.fusion_activation(fused_output)
        
        return fused_output
    
    def _apply_scale_transform(self, nbr_fea: torch.Tensor, scale: int) -> torch.Tensor:
        """
        应用尺度变换到邻居特征
        
        Args:
            nbr_fea: 原始邻居特征
            scale: 尺度因子
            
        Returns:
            scaled_nbr_fea: 变换后的邻居特征
        """
        # 简化的尺度变换：使用不同的权重矩阵
        if not hasattr(self, f'_scale_transform_{scale}'):
            setattr(self, f'_scale_transform_{scale}', 
                   nn.Linear(nbr_fea.size(-1), nbr_fea.size(-1)).to(nbr_fea.device))
        
        transform = getattr(self, f'_scale_transform_{scale}')
        return transform(nbr_fea)


class HierarchicalPooling(nn.Module):
    """
    分层池化模块
    Hierarchical Pooling Module
    
    设计理念：
    - 传统的全局平均池化丢失了局部结构信息
    - 分层池化可以保留不同层次的结构特征
    - 结合注意力机制可以自适应地聚合重要的原子特征
    """
    
    def __init__(self, atom_fea_len: int, hierarchy_levels: int = 3):
        super(HierarchicalPooling, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.hierarchy_levels = hierarchy_levels
        
        # 每个层次的注意力模块
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(atom_fea_len, atom_fea_len // 2),
                nn.ReLU(),
                nn.Linear(atom_fea_len // 2, 1)
            ) for _ in range(hierarchy_levels)
        ])
        
        # 层次融合
        self.hierarchy_fusion = nn.Linear(hierarchy_levels * atom_fea_len, atom_fea_len)
        
    def forward(self, atom_fea: torch.Tensor, 
                crystal_atom_idx: List[torch.Tensor]) -> torch.Tensor:
        """
        分层池化前向传播
        
        Args:
            atom_fea: 原子特征
            crystal_atom_idx: 晶体原子索引
            
        Returns:
            pooled_features: 池化后的晶体特征
        """
        batch_features = []
        
        for idx_map in crystal_atom_idx:
            crystal_atom_fea = atom_fea[idx_map]  # [n_atoms, atom_fea_len]
            
            hierarchy_features = []
            
            # 对每个层次进行池化
            for level in range(self.hierarchy_levels):
                # 计算注意力权重
                attention_scores = self.attention_layers[level](crystal_atom_fea)  # [n_atoms, 1]
                attention_weights = F.softmax(attention_scores, dim=0)
                
                # 加权池化
                weighted_feature = torch.sum(attention_weights * crystal_atom_fea, dim=0)
                hierarchy_features.append(weighted_feature)
            
            # 融合不同层次的特征
            concatenated = torch.cat(hierarchy_features, dim=0)
            fused_feature = self.hierarchy_fusion(concatenated)
            
            batch_features.append(fused_feature.unsqueeze(0))
        
        return torch.cat(batch_features, dim=0)


class EnhancedCGCNN(nn.Module):
    """
    增强的晶体图卷积神经网络
    Enhanced Crystal Graph Convolutional Neural Network
    
    架构设计理念：
    1. 原子嵌入：将原子特征映射到高维表示空间，便于后续处理
    2. 多尺度图卷积：捕获不同距离范围的原子间相互作用
    3. 图注意力机制：自适应地关注重要的原子和化学键
    4. 分层池化：保留多层次的结构信息
    5. 物理约束：确保预测结果符合物理规律
    """
    
    def __init__(self, orig_atom_fea_len: int, nbr_fea_len: int,
                 atom_fea_len: int = 64, n_conv: int = 3, h_fea_len: int = 128,
                 n_h: int = 1, classification: bool = False,
                 use_attention: bool = True, use_multiscale: bool = True,
                 attention_heads: int = 8, multiscale_levels: List[int] = [1, 2, 3]):
        super(EnhancedCGCNN, self).__init__()
        
        self.classification = classification
        self.use_attention = use_attention
        self.use_multiscale = use_multiscale
        
        # 1. 原子特征嵌入
        # 理论依据：原子的初始表示可能不是最优的，需要学习更好的表示
        self.embedding = nn.Sequential(
            nn.Linear(orig_atom_fea_len, atom_fea_len),
            nn.BatchNorm1d(atom_fea_len),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. 图卷积层
        self.convs = nn.ModuleList()
        
        for i in range(n_conv):
            if use_multiscale:
                # 多尺度卷积层
                conv_layer = MultiScaleConvLayer(atom_fea_len, nbr_fea_len, multiscale_levels)
            else:
                # 标准卷积层
                conv_layer = ConvLayer(atom_fea_len, nbr_fea_len)
            
            self.convs.append(conv_layer)
            
            # 可选的注意力层
            if use_attention:
                attention_layer = GraphAttentionLayer(atom_fea_len, nbr_fea_len, attention_heads)
                self.convs.append(attention_layer)
        
        # 3. 池化层
        if use_attention:
            self.pooling = HierarchicalPooling(atom_fea_len, hierarchy_levels=3)
        else:
            self.pooling = self._standard_pooling
        
        # 4. 特征变换
        self.conv_to_fc = nn.Sequential(
            nn.Linear(atom_fea_len, h_fea_len),
            nn.BatchNorm1d(h_fea_len),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 5. 全连接层
        if n_h > 1:
            self.fcs = nn.ModuleList()
            self.fc_norms = nn.ModuleList()
            
            for _ in range(n_h - 1):
                self.fcs.append(nn.Linear(h_fea_len, h_fea_len))
                self.fc_norms.append(nn.BatchNorm1d(h_fea_len))
        
        # 6. 输出层
        if classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.output_activation = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
            self.output_activation = None
        
        # 7. 正则化
        self.dropout = nn.Dropout(0.1)
        
        # 8. 权重初始化
        self._initialize_weights()
    
    def forward(self, atom_fea: torch.Tensor, nbr_fea: torch.Tensor, 
                nbr_fea_idx: torch.Tensor, crystal_atom_idx: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            atom_fea: 原子特征 [N, orig_atom_fea_len]
            nbr_fea: 邻居特征 [N, M, nbr_fea_len]
            nbr_fea_idx: 邻居索引 [N, M]
            crystal_atom_idx: 晶体原子索引映射
            
        Returns:
            output: 预测结果
        """
        # 1. 原子特征嵌入
        atom_fea = self.embedding(atom_fea)
        
        # 2. 图卷积处理
        for conv_layer in self.convs:
            if isinstance(conv_layer, (MultiScaleConvLayer, ConvLayer)):
                atom_fea = conv_layer(atom_fea, nbr_fea, nbr_fea_idx)
            elif isinstance(conv_layer, GraphAttentionLayer):
                atom_fea = conv_layer(atom_fea, nbr_fea, nbr_fea_idx)
        
        # 3. 池化操作
        if callable(self.pooling):
            crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        else:
            crys_fea = self._standard_pooling(atom_fea, crystal_atom_idx)
        
        # 4. 特征变换
        crys_fea = self.conv_to_fc(crys_fea)
        
        # 5. 全连接层处理
        if hasattr(self, 'fcs'):
            for fc, norm in zip(self.fcs, self.fc_norms):
                crys_fea = F.relu(norm(fc(crys_fea)))
                crys_fea = self.dropout(crys_fea)
        
        # 6. 输出层
        output = self.fc_out(crys_fea)
        
        if self.output_activation:
            output = self.output_activation(output)
        
        return output
    
    def _standard_pooling(self, atom_fea: torch.Tensor, 
                         crystal_atom_idx: List[torch.Tensor]) -> torch.Tensor:
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.size(0)
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def get_attention_weights(self, atom_fea: torch.Tensor, nbr_fea: torch.Tensor, 
                            nbr_fea_idx: torch.Tensor) -> List[torch.Tensor]:
        """
        获取注意力权重用于可解释性分析
        Get attention weights for interpretability
        
        Returns:
            attention_weights: 各层的注意力权重
        """
        attention_weights = []
        
        if not self.use_attention:
            return attention_weights
        
        # 前向传播并收集注意力权重
        atom_fea = self.embedding(atom_fea)
        
        for conv_layer in self.convs:
            if isinstance(conv_layer, GraphAttentionLayer):
                # 临时修改前向传播以返回注意力权重
                # 这里需要修改GraphAttentionLayer的实现来支持返回注意力权重
                pass
            else:
                atom_fea = conv_layer(atom_fea, nbr_fea, nbr_fea_idx)
        
        return attention_weights


class PhysicsInformedLoss(nn.Module):
    """
    物理约束损失函数
    Physics-Informed Loss Function
    
    设计理念：
    - 确保预测结果符合基本物理定律
    - 利用已知的物理关系约束模型学习
    - 提高模型在数据稀缺区域的泛化能力
    """
    
    def __init__(self, base_loss_weight: float = 1.0, 
                 physics_loss_weight: float = 0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.base_loss_weight = base_loss_weight
        self.physics_loss_weight = physics_loss_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                property_type: str = 'formation_energy') -> torch.Tensor:
        """
        计算物理约束损失
        
        Args:
            predictions: 预测值
            targets: 真实值
            property_type: 性质类型
            
        Returns:
            total_loss: 总损失
        """
        # 基础损失
        base_loss = F.mse_loss(predictions, targets)
        
        # 物理约束损失
        physics_loss = self._calculate_physics_constraints(predictions, property_type)
        
        # 总损失
        total_loss = (self.base_loss_weight * base_loss + 
                     self.physics_loss_weight * physics_loss)
        
        return total_loss
    
    def _calculate_physics_constraints(self, predictions: torch.Tensor, 
                                     property_type: str) -> torch.Tensor:
        if property_type == 'formation_energy':
            # 形成能的物理约束：绝大多数情况下应为负值
            positive_penalty = torch.relu(predictions).mean()
            return positive_penalty
        
        elif property_type == 'band_gap':
            # 带隙约束：应为非负值
            negative_penalty = torch.relu(-predictions).mean()
            return negative_penalty
        
        elif property_type == 'elastic_moduli':
            # 弹性模量约束：应为正值
            negative_penalty = torch.relu(-predictions).mean()
            return negative_penalty
        
        else:
            return torch.tensor(0.0, device=predictions.device)


# 使用示例和测试
def example_usage():
    # 创建增强的CGCNN模型
    model = EnhancedCGCNN(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=2,
        classification=False,
        use_attention=True,
        use_multiscale=True,
        attention_heads=8,
        multiscale_levels=[1, 2, 3]
    )
    
    # 创建模拟数据
    batch_size = 4
    n_atoms = 20
    max_neighbors = 12
    
    atom_fea = torch.randn(n_atoms, 92)
    nbr_fea = torch.randn(n_atoms, max_neighbors, 41)
    nbr_fea_idx = torch.randint(0, n_atoms, (n_atoms, max_neighbors))
    crystal_atom_idx = [torch.arange(5 * i, 5 * (i + 1)) for i in range(batch_size)]
    
    # 前向传播
    output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
    print(f"Model output shape: {output.shape}")
    
    # 使用物理约束损失
    physics_loss = PhysicsInformedLoss()
    targets = torch.randn(batch_size, 1)
    loss = physics_loss(output, targets, 'formation_energy')
    print(f"Physics-informed loss: {loss.item():.4f}")
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    example_usage() 