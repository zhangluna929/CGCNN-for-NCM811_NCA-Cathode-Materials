"""
Multimodal Fusion for Materials Property Prediction

Integration of multiple data modalities (structure, composition, sequence)
for enhanced materials property prediction using cross-modal attention
and fusion strategies.

Author: lunazhang
Date: 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

try:
    from pymatgen.core.structure import Structure
    from pymatgen.core.composition import Composition
    PYMATGEN_AVAILABLE = True
except ImportError:
    warnings.warn("PyMatGen not available. Some features may be limited.")
    PYMATGEN_AVAILABLE = False
    Structure = None
    Composition = None

from .model import CrystalGraphConvNet, ConvLayer
from .features import ChemicalDescriptors


class ModalityEncoder(nn.Module):
    """
    模态编码器基类
    Base class for modality encoders
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super(ModalityEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StructureEncoder(ModalityEncoder):
    """
    结构模态编码器
    Structure modality encoder using CGCNN
    """
    
    def __init__(self, orig_atom_fea_len: int, nbr_fea_len: int, 
                 atom_fea_len: int = 64, n_conv: int = 3, output_dim: int = 128):
        super(StructureEncoder, self).__init__(atom_fea_len, output_dim)
        
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.pooling_fc = nn.Linear(atom_fea_len, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, atom_fea: torch.Tensor, nbr_fea: torch.Tensor, 
                nbr_fea_idx: torch.Tensor, crystal_atom_idx: List[torch.Tensor]) -> torch.Tensor:
        # 图卷积
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        # 全局池化
        crystal_fea = self.pooling(atom_fea, crystal_atom_idx)
        
        # 输出编码
        encoded = self.activation(self.pooling_fc(crystal_fea))
        return encoded
    
    def pooling(self, atom_fea: torch.Tensor, crystal_atom_idx: List[torch.Tensor]) -> torch.Tensor:
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


class CompositionEncoder(ModalityEncoder):
    """
    组成模态编码器
    Composition modality encoder
    """
    
    def __init__(self, composition_dim: int, output_dim: int = 128, 
                 hidden_dim: int = 64):
        super(CompositionEncoder, self).__init__(composition_dim, output_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(composition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, composition_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(composition_features)


class SequenceEncoder(ModalityEncoder):
    """
    序列模态编码器（用于元素序列）
    Sequence modality encoder for element sequences
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, 
                 output_dim: int = 128, num_layers: int = 2):
        super(SequenceEncoder, self).__init__(embed_dim, output_dim)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.1, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.activation = nn.ReLU()
    
    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            sequences: [batch_size, max_len] 元素序列
            lengths: [batch_size] 序列长度
        """
        # 嵌入
        embedded = self.embedding(sequences)
        
        # 打包序列
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), 
                                                  batch_first=True, enforce_sorted=False)
        
        # LSTM
        output, (hidden, _) = self.lstm(packed)
        
        # 使用最后一个隐藏状态
        # hidden: [num_layers*num_directions, batch, hidden_dim]
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # 连接双向
        
        # 输出编码
        encoded = self.activation(self.fc(final_hidden))
        return encoded


class PhysicalPropertyEncoder(ModalityEncoder):
    """
    物理性质编码器
    Physical property encoder
    """
    
    def __init__(self, property_dim: int, output_dim: int = 128):
        super(PhysicalPropertyEncoder, self).__init__(property_dim, output_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(property_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    
    def forward(self, properties: torch.Tensor) -> torch.Tensor:
        return self.encoder(properties)


class AttentionFusion(nn.Module):
    """
    注意力融合模块
    Attention-based fusion module
    """
    
    def __init__(self, modality_dims: List[int], output_dim: int):
        super(AttentionFusion, self).__init__()
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        
        # 注意力权重计算
        self.attention_weights = nn.ModuleList([
            nn.Linear(dim, 1) for dim in modality_dims
        ])
        
        # 融合后的变换
        total_dim = sum(modality_dims)
        self.fusion_fc = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        注意力融合
        
        Args:
            modality_features: 各模态特征列表
        
        Returns:
            fused_features: 融合后的特征
        """
        # 计算注意力权重
        attention_scores = []
        for i, features in enumerate(modality_features):
            score = self.attention_weights[i](features)
            attention_scores.append(score)
        
        # Softmax归一化
        attention_weights = F.softmax(torch.cat(attention_scores, dim=1), dim=1)
        
        # 加权融合
        weighted_features = []
        for i, features in enumerate(modality_features):
            weight = attention_weights[:, i:i+1]
            weighted_features.append(weight * features)
        
        # 连接所有特征
        concatenated = torch.cat(weighted_features, dim=1)
        
        # 最终变换
        fused = self.fusion_fc(concatenated)
        return fused


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块
    Cross-modal attention module
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super(CrossModalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, query_features: torch.Tensor, key_features: torch.Tensor, 
                value_features: torch.Tensor) -> torch.Tensor:
        """
        跨模态注意力计算
        
        Args:
            query_features: 查询特征 [batch_size, feature_dim]
            key_features: 键特征 [batch_size, feature_dim]
            value_features: 值特征 [batch_size, feature_dim]
        """
        batch_size = query_features.shape[0]
        
        # 线性变换
        Q = self.query(query_features).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(key_features).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(value_features).view(batch_size, self.num_heads, self.head_dim)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        attended = attended.view(batch_size, self.feature_dim)
        
        # 输出变换
        output = self.fc_out(attended)
        return output


class MultiModalCGCNN(nn.Module):
    """
    多模态CGCNN
    Multi-modal Crystal Graph Convolutional Neural Network
    """
    
    def __init__(self, 
                 # 结构编码器参数
                 orig_atom_fea_len: int, nbr_fea_len: int,
                 atom_fea_len: int = 64, n_conv: int = 3,
                 # 其他模态参数
                 composition_dim: int = 20,
                 sequence_vocab_size: int = 100,
                 property_dim: int = 10,
                 # 融合参数
                 fusion_dim: int = 256,
                 output_dim: int = 1,
                 use_attention_fusion: bool = True,
                 classification: bool = False):
        
        super(MultiModalCGCNN, self).__init__()
        self.classification = classification
        self.use_attention_fusion = use_attention_fusion
        
        # 模态编码器
        self.structure_encoder = StructureEncoder(
            orig_atom_fea_len, nbr_fea_len, atom_fea_len, n_conv, fusion_dim//4
        )
        self.composition_encoder = CompositionEncoder(composition_dim, fusion_dim//4)
        self.sequence_encoder = SequenceEncoder(sequence_vocab_size, output_dim=fusion_dim//4)
        self.property_encoder = PhysicalPropertyEncoder(property_dim, fusion_dim//4)
        
        # 融合模块
        modality_dims = [fusion_dim//4] * 4
        if use_attention_fusion:
            self.fusion = AttentionFusion(modality_dims, fusion_dim)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(sum(modality_dims), fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # 跨模态注意力
        self.cross_attention = CrossModalAttention(fusion_dim//4)
        
        # 输出层
        if classification:
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(fusion_dim//2, output_dim),
                nn.LogSoftmax(dim=1)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(fusion_dim//2, output_dim)
            )
    
    def forward(self, 
                # 结构数据
                atom_fea: torch.Tensor, nbr_fea: torch.Tensor, 
                nbr_fea_idx: torch.Tensor, crystal_atom_idx: List[torch.Tensor],
                # 其他模态数据
                composition_features: torch.Tensor,
                element_sequences: torch.Tensor, sequence_lengths: torch.Tensor,
                physical_properties: torch.Tensor) -> torch.Tensor:
        """
        多模态前向传播
        
        Args:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx: 结构数据
            composition_features: 组成特征 [batch_size, composition_dim]
            element_sequences: 元素序列 [batch_size, max_len]
            sequence_lengths: 序列长度 [batch_size]
            physical_properties: 物理性质 [batch_size, property_dim]
        """
        # 各模态编码
        structure_features = self.structure_encoder(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        composition_features = self.composition_encoder(composition_features)
        sequence_features = self.sequence_encoder(element_sequences, sequence_lengths)
        property_features = self.property_encoder(physical_properties)
        
        # 跨模态注意力增强
        enhanced_structure = self.cross_attention(structure_features, composition_features, composition_features)
        enhanced_composition = self.cross_attention(composition_features, structure_features, structure_features)
        
        # 特征融合
        if self.use_attention_fusion:
            fused_features = self.fusion([enhanced_structure, enhanced_composition, 
                                        sequence_features, property_features])
        else:
            concatenated = torch.cat([enhanced_structure, enhanced_composition, 
                                    sequence_features, property_features], dim=1)
            fused_features = self.fusion(concatenated)
        
        # 输出预测
        if self.classification:
            output = self.classifier(fused_features)
        else:
            output = self.regressor(fused_features)
        
        return output


class StructureSequenceCGCNN(nn.Module):
    """
    结构-序列融合CGCNN
    Structure-Sequence fusion CGCNN
    """
    
    def __init__(self, 
                 orig_atom_fea_len: int, nbr_fea_len: int,
                 sequence_vocab_size: int, max_sequence_length: int,
                 atom_fea_len: int = 64, n_conv: int = 3,
                 sequence_embed_dim: int = 32, sequence_hidden_dim: int = 64,
                 fusion_dim: int = 128, output_dim: int = 1,
                 classification: bool = False):
        
        super(StructureSequenceCGCNN, self).__init__()
        self.classification = classification
        
        # 结构分支（基于CGCNN）
        self.atom_embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.structure_convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                             nbr_fea_len=nbr_fea_len)
                                             for _ in range(n_conv)])
        self.structure_fc = nn.Linear(atom_fea_len, fusion_dim//2)
        
        # 序列分支（基于LSTM）
        self.sequence_embedding = nn.Embedding(sequence_vocab_size, sequence_embed_dim)
        self.sequence_lstm = nn.LSTM(sequence_embed_dim, sequence_hidden_dim, 
                                   batch_first=True, bidirectional=True)
        self.sequence_fc = nn.Linear(sequence_hidden_dim * 2, fusion_dim//2)
        
        # 融合层
        self.fusion_attention = nn.MultiheadAttention(fusion_dim//2, num_heads=4, batch_first=True)
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 输出层
        if classification:
            self.output_layer = nn.Sequential(
                nn.Linear(fusion_dim, output_dim),
                nn.LogSoftmax(dim=1)
            )
        else:
            self.output_layer = nn.Linear(fusion_dim, output_dim)
    
    def forward(self, 
                atom_fea: torch.Tensor, nbr_fea: torch.Tensor, 
                nbr_fea_idx: torch.Tensor, crystal_atom_idx: List[torch.Tensor],
                element_sequences: torch.Tensor) -> torch.Tensor:
        """
        结构-序列融合前向传播
        
        Args:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx: 结构数据
            element_sequences: 元素序列 [batch_size, max_len]
        """
        # 结构分支
        atom_fea_embedded = self.atom_embedding(atom_fea)
        for conv in self.structure_convs:
            atom_fea_embedded = conv(atom_fea_embedded, nbr_fea, nbr_fea_idx)
        
        # 结构池化
        structure_features = self.pooling(atom_fea_embedded, crystal_atom_idx)
        structure_encoded = F.relu(self.structure_fc(structure_features))
        
        # 序列分支
        sequence_embedded = self.sequence_embedding(element_sequences)
        sequence_output, _ = self.sequence_lstm(sequence_embedded)
        
        # 使用最后一个时间步的输出
        sequence_final = sequence_output[:, -1, :]
        sequence_encoded = F.relu(self.sequence_fc(sequence_final))
        
        # 注意力融合
        structure_expanded = structure_encoded.unsqueeze(1)  # [batch, 1, dim]
        sequence_expanded = sequence_encoded.unsqueeze(1)    # [batch, 1, dim]
        
        fused_structure, _ = self.fusion_attention(structure_expanded, sequence_expanded, sequence_expanded)
        fused_sequence, _ = self.fusion_attention(sequence_expanded, structure_expanded, structure_expanded)
        
        # 特征连接和变换
        fused_features = torch.cat([fused_structure.squeeze(1), fused_sequence.squeeze(1)], dim=1)
        fused_features = self.fusion_fc(fused_features)
        
        # 输出预测
        output = self.output_layer(fused_features)
        return output
    
    def pooling(self, atom_fea: torch.Tensor, crystal_atom_idx: List[torch.Tensor]) -> torch.Tensor:
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


class MultiModalDataLoader:
    """
    多模态数据加载器
    Multi-modal data loader
    """
    
    def __init__(self):
        self.chemical_desc = ChemicalDescriptors()
    
    def prepare_multimodal_data(self, structure, element_vocab: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        准备多模态数据
        Prepare multi-modal data from structure
        
        Args:
            structure: pymatgen Structure object
            element_vocab: 元素到索引的映射
        
        Returns:
            multimodal_data: 多模态数据字典
        """
        # 组成特征
        composition_features = self.chemical_desc.compute_chemical_fingerprint(structure)
        
        # 元素序列
        elements = [str(site.specie) for site in structure]
        element_sequence = [element_vocab.get(elem, 0) for elem in elements]  # 0为未知元素
        
        # 物理性质（示例）
        physical_properties = np.array([
            structure.density,
            structure.volume / len(structure),  # 每原子体积
            len(structure),  # 原子数
            structure.lattice.a,
            structure.lattice.b,
            structure.lattice.c,
            structure.lattice.alpha,
            structure.lattice.beta,
            structure.lattice.gamma,
            structure.get_space_group_info()[1]  # 空间群编号
        ])
        
        return {
            'composition_features': torch.tensor(composition_features, dtype=torch.float32),
            'element_sequence': torch.tensor(element_sequence, dtype=torch.long),
            'sequence_length': torch.tensor(len(element_sequence), dtype=torch.long),
            'physical_properties': torch.tensor(physical_properties, dtype=torch.float32)
        }


# 使用示例
def example_usage():
    # 创建多模态模型
    model = MultiModalCGCNN(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        composition_dim=20,
        sequence_vocab_size=100,
        property_dim=10,
        fusion_dim=256,
        output_dim=1,
        classification=False
    )
    
    # 或者创建结构-序列融合模型
    # seq_model = StructureSequenceCGCNN(
    #     orig_atom_fea_len=92,
    #     nbr_fea_len=41,
    #     sequence_vocab_size=100,
    #     max_sequence_length=50,
    #     output_dim=1
    # )
    
    pass 