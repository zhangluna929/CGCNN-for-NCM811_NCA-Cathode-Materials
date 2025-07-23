"""
Data Imbalance Handling Module

Comprehensive solutions for handling data imbalance in materials datasets,
including SMOTE augmentation, weight adjustment, and physics-constrained
data generation for rare defect configurations and extreme conditions.

Author: lunazhang
Date: 2023
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from collections import Counter, defaultdict
import logging

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek
    from sklearn.neighbors import NearestNeighbors
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not available. Some features may be limited.")

try:
    from pymatgen.core.structure import Structure
    from pymatgen.transformations.standard_transformations import *
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


class MaterialsSMOTE:
    """
    材料专用SMOTE算法
    Materials-Specific SMOTE Algorithm
    """
    
    def __init__(self, k_neighbors: int = 5, random_state: int = 42,
                 feature_constraints: Optional[Dict[str, Tuple[float, float]]] = None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.feature_constraints = feature_constraints or {}
        
        np.random.seed(random_state)
        self.logger = logging.getLogger(__name__)
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray, 
                    structure_info: Optional[List[Dict]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        材料数据的SMOTE重采样
        SMOTE resampling for materials data
        
        Args:
            X: 特征矩阵
            y: 标签
            structure_info: 结构信息（用于物理约束）
            
        Returns:
            X_resampled, y_resampled: 重采样后的数据
        """
        # 分析类别分布
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        minority_classes = [cls for cls in class_counts if cls != majority_class]
        
        self.logger.info(f"Original class distribution: {class_counts}")
        
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        # 为每个少数类生成合成样本
        for minority_class in minority_classes:
            minority_indices = np.where(y == minority_class)[0]
            minority_X = X[minority_indices]
            
            # 计算需要生成的样本数
            target_samples = class_counts[majority_class] - class_counts[minority_class]
            
            if len(minority_indices) < self.k_neighbors:
                # 如果样本太少，使用所有样本作为邻居
                effective_k = len(minority_indices) - 1
                if effective_k <= 0:
                    self.logger.warning(f"Too few samples for class {minority_class}, skipping SMOTE")
                    continue
            else:
                effective_k = self.k_neighbors
            
            # 生成合成样本
            synthetic_samples = self._generate_synthetic_samples(
                minority_X, target_samples, effective_k, structure_info
            )
            
            # 添加到重采样数据中
            X_resampled = np.vstack([X_resampled, synthetic_samples])
            y_resampled = np.hstack([y_resampled, [minority_class] * len(synthetic_samples)])
        
        # 打乱数据
        shuffle_indices = np.random.permutation(len(X_resampled))
        X_resampled = X_resampled[shuffle_indices]
        y_resampled = y_resampled[shuffle_indices]
        
        new_class_counts = Counter(y_resampled)
        self.logger.info(f"Resampled class distribution: {new_class_counts}")
        
        return X_resampled, y_resampled
    
    def _generate_synthetic_samples(self, minority_X: np.ndarray, n_samples: int,
                                  k_neighbors: int, structure_info: Optional[List[Dict]]) -> np.ndarray:
        """生成合成样本"""
        if len(minority_X) < 2:
            return np.array([]).reshape(0, minority_X.shape[1])
        
        # 找到k近邻
        nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(minority_X)))
        nn.fit(minority_X)
        
        synthetic_samples = []
        
        for _ in range(n_samples):
            # 随机选择一个少数类样本
            random_idx = np.random.randint(0, len(minority_X))
            sample = minority_X[random_idx]
            
            # 找到其邻居
            distances, indices = nn.kneighbors([sample])
            neighbor_indices = indices[0][1:]  # 排除自己
            
            if len(neighbor_indices) == 0:
                # 如果没有邻居，直接复制原样本并添加噪声
                synthetic_sample = sample + np.random.normal(0, 0.01, sample.shape)
            else:
                # 随机选择一个邻居
                neighbor_idx = np.random.choice(neighbor_indices)
                neighbor = minority_X[neighbor_idx]
                
                # 在样本和邻居之间插值
                alpha = np.random.random()
                synthetic_sample = sample + alpha * (neighbor - sample)
            
            # 应用物理约束
            synthetic_sample = self._apply_physical_constraints(synthetic_sample)
            
            synthetic_samples.append(synthetic_sample)
        
        return np.array(synthetic_samples)
    
    def _apply_physical_constraints(self, sample: np.ndarray) -> np.ndarray:
        """应用物理约束"""
        constrained_sample = sample.copy()
        
        # 应用特征约束
        for feature_idx, (min_val, max_val) in self.feature_constraints.items():
            if isinstance(feature_idx, int) and 0 <= feature_idx < len(sample):
                constrained_sample[feature_idx] = np.clip(sample[feature_idx], min_val, max_val)
        
        # 材料特定的约束
        constrained_sample = self._apply_materials_constraints(constrained_sample)
        
        return constrained_sample
    
    def _apply_materials_constraints(self, sample: np.ndarray) -> np.ndarray:
        """应用材料特定约束"""
        # 这里可以添加材料科学的具体约束
        # 例如：确保原子比例合理、键长在合理范围内等
        
        # 示例：确保某些特征的和为1（如果它们代表原子分数）
        # composition_features = sample[:5]  # 假设前5个特征是成分
        # if np.sum(composition_features) > 0:
        #     sample[:5] = composition_features / np.sum(composition_features)
        
        return sample


class PhysicsConstrainedDataGenerator:
    """
    基于物理约束的数据生成器
    Physics-Constrained Data Generator
    """
    
    def __init__(self, base_structures: Optional[List[Structure]] = None):
        self.base_structures = base_structures or []
        self.generation_rules = self._default_generation_rules()
        
        self.logger = logging.getLogger(__name__)
    
    def _default_generation_rules(self) -> Dict[str, Any]:
        """默认生成规则"""
        return {
            'formation_energy': {
                'min_value': -6.0,
                'max_value': 0.0,
                'stability_constraint': True
            },
            'band_gap': {
                'min_value': 0.0,
                'max_value': 5.0,
                'semiconductor_bias': True
            },
            'elastic_moduli': {
                'min_value': 50.0,
                'max_value': 500.0,
                'positive_constraint': True
            },
            'composition': {
                'sum_to_one': True,
                'non_negative': True
            }
        }
    
    def generate_rare_defect_samples(self, n_samples: int, defect_type: str,
                                   base_material: str = 'NCM811') -> List[Dict[str, Any]]:
        """
        生成稀有缺陷样本
        Generate rare defect configuration samples
        
        Args:
            n_samples: 生成样本数量
            defect_type: 缺陷类型
            base_material: 基础材料
            
        Returns:
            generated_samples: 生成的样本列表
        """
        generated_samples = []
        
        for i in range(n_samples):
            sample = self._generate_single_defect_sample(defect_type, base_material, i)
            generated_samples.append(sample)
        
        self.logger.info(f"Generated {len(generated_samples)} {defect_type} samples for {base_material}")
        
        return generated_samples
    
    def _generate_single_defect_sample(self, defect_type: str, base_material: str, 
                                     sample_id: int) -> Dict[str, Any]:
        """生成单个缺陷样本"""
        # 基础结构参数
        base_params = self._get_base_material_params(base_material)
        
        # 缺陷特定修改
        if defect_type == 'Li_vacancy':
            sample = self._generate_li_vacancy_sample(base_params, sample_id)
        elif defect_type == 'Ni_migration':
            sample = self._generate_ni_migration_sample(base_params, sample_id)
        elif defect_type == 'O_vacancy':
            sample = self._generate_o_vacancy_sample(base_params, sample_id)
        elif defect_type == 'complex_defect':
            sample = self._generate_complex_defect_sample(base_params, sample_id)
        else:
            raise ValueError(f"Unknown defect type: {defect_type}")
        
        # 应用物理约束
        sample = self._apply_physics_constraints(sample, defect_type)
        
        return sample
    
    def _get_base_material_params(self, material: str) -> Dict[str, Any]:
        """获取基础材料参数"""
        base_params = {
            'NCM811': {
                'lattice_a': 2.87,
                'lattice_c': 14.2,
                'formation_energy_base': -4.1,
                'band_gap_base': 0.35,
                'bulk_modulus_base': 170.0
            },
            'NCA': {
                'lattice_a': 2.86,
                'lattice_c': 14.18,
                'formation_energy_base': -3.8,
                'band_gap_base': 0.50,
                'bulk_modulus_base': 175.0
            }
        }
        
        return base_params.get(material, base_params['NCM811'])
    
    def _generate_li_vacancy_sample(self, base_params: Dict[str, Any], 
                                  sample_id: int) -> Dict[str, Any]:
        """生成锂空位样本"""
        # 基于已知的物理关系生成特征
        vacancy_concentration = np.random.uniform(0.05, 0.3)  # 5-30%空位浓度
        
        # 形成能变化（空位增加形成能）
        formation_energy = base_params['formation_energy_base'] + \
                          vacancy_concentration * np.random.uniform(0.2, 0.8)
        
        # 带隙变化（空位可能改变电子结构）
        band_gap = base_params['band_gap_base'] + \
                  np.random.normal(0, 0.1) + vacancy_concentration * 0.1
        
        # 弹性模量变化（空位降低模量）
        bulk_modulus = base_params['bulk_modulus_base'] * \
                      (1 - vacancy_concentration * np.random.uniform(0.1, 0.3))
        
        # 生成原子特征（简化）
        atom_features = self._generate_atom_features_with_vacancy(
            base_params, vacancy_concentration
        )
        
        # 生成邻居特征
        neighbor_features = self._generate_neighbor_features_with_defect(
            base_params, 'Li_vacancy', vacancy_concentration
        )
        
        return {
            'sample_id': f"Li_vac_{sample_id}",
            'defect_type': 'Li_vacancy',
            'formation_energy': formation_energy,
            'band_gap': max(0, band_gap),
            'bulk_modulus': max(50, bulk_modulus),
            'vacancy_concentration': vacancy_concentration,
            'atom_features': atom_features,
            'neighbor_features': neighbor_features,
            'metadata': {
                'generated': True,
                'base_material': 'NCM811',
                'generation_method': 'physics_constrained'
            }
        }
    
    def _generate_ni_migration_sample(self, base_params: Dict[str, Any], 
                                    sample_id: int) -> Dict[str, Any]:
        """生成镍迁移样本"""
        migration_fraction = np.random.uniform(0.01, 0.15)  # 1-15%迁移
        
        # 形成能大幅增加（迁移是高能过程）
        formation_energy = base_params['formation_energy_base'] + \
                          migration_fraction * np.random.uniform(0.5, 1.2)
        
        # 带隙变化（可能形成缺陷态）
        band_gap = base_params['band_gap_base'] + \
                  np.random.normal(0, 0.15) - migration_fraction * 0.2
        
        # 弹性性质变化
        bulk_modulus = base_params['bulk_modulus_base'] * \
                      (1 - migration_fraction * np.random.uniform(0.2, 0.5))
        
        atom_features = self._generate_atom_features_with_migration(
            base_params, migration_fraction
        )
        
        neighbor_features = self._generate_neighbor_features_with_defect(
            base_params, 'Ni_migration', migration_fraction
        )
        
        return {
            'sample_id': f"Ni_mig_{sample_id}",
            'defect_type': 'Ni_migration',
            'formation_energy': formation_energy,
            'band_gap': max(0, band_gap),
            'bulk_modulus': max(50, bulk_modulus),
            'migration_fraction': migration_fraction,
            'atom_features': atom_features,
            'neighbor_features': neighbor_features,
            'metadata': {
                'generated': True,
                'base_material': 'NCM811',
                'generation_method': 'physics_constrained'
            }
        }
    
    def _generate_o_vacancy_sample(self, base_params: Dict[str, Any], 
                                 sample_id: int) -> Dict[str, Any]:
        """生成氧空位样本"""
        vacancy_concentration = np.random.uniform(0.01, 0.1)  # 1-10%氧空位
        
        # 氧空位对性质的影响
        formation_energy = base_params['formation_energy_base'] + \
                          vacancy_concentration * np.random.uniform(0.8, 1.5)
        
        # 氧空位可能引入缺陷态，减小带隙
        band_gap = base_params['band_gap_base'] - \
                  vacancy_concentration * np.random.uniform(0.1, 0.3)
        
        bulk_modulus = base_params['bulk_modulus_base'] * \
                      (1 - vacancy_concentration * np.random.uniform(0.15, 0.4))
        
        atom_features = self._generate_atom_features_with_o_vacancy(
            base_params, vacancy_concentration
        )
        
        neighbor_features = self._generate_neighbor_features_with_defect(
            base_params, 'O_vacancy', vacancy_concentration
        )
        
        return {
            'sample_id': f"O_vac_{sample_id}",
            'defect_type': 'O_vacancy',
            'formation_energy': formation_energy,
            'band_gap': max(0, band_gap),
            'bulk_modulus': max(50, bulk_modulus),
            'vacancy_concentration': vacancy_concentration,
            'atom_features': atom_features,
            'neighbor_features': neighbor_features,
            'metadata': {
                'generated': True,
                'base_material': 'NCM811',
                'generation_method': 'physics_constrained'
            }
        }
    
    def _generate_complex_defect_sample(self, base_params: Dict[str, Any], 
                                      sample_id: int) -> Dict[str, Any]:
        """生成复合缺陷样本"""
        # 复合缺陷：同时包含多种缺陷类型
        li_vacancy_conc = np.random.uniform(0.02, 0.15)
        ni_migration_conc = np.random.uniform(0.005, 0.05)
        o_vacancy_conc = np.random.uniform(0.005, 0.03)
        
        # 复合效应：可能非线性
        defect_interaction_factor = np.random.uniform(0.8, 1.3)
        
        formation_energy = base_params['formation_energy_base'] + \
                          defect_interaction_factor * (
                              li_vacancy_conc * 0.3 + 
                              ni_migration_conc * 0.8 + 
                              o_vacancy_conc * 1.0
                          )
        
        band_gap = base_params['band_gap_base'] + \
                  np.random.normal(0, 0.2) - \
                  (o_vacancy_conc * 0.2 - li_vacancy_conc * 0.05)
        
        bulk_modulus = base_params['bulk_modulus_base'] * \
                      (1 - (li_vacancy_conc * 0.15 + 
                           ni_migration_conc * 0.3 + 
                           o_vacancy_conc * 0.25) * defect_interaction_factor)
        
        return {
            'sample_id': f"complex_{sample_id}",
            'defect_type': 'complex_defect',
            'formation_energy': formation_energy,
            'band_gap': max(0, band_gap),
            'bulk_modulus': max(50, bulk_modulus),
            'li_vacancy_conc': li_vacancy_conc,
            'ni_migration_conc': ni_migration_conc,
            'o_vacancy_conc': o_vacancy_conc,
            'defect_interaction_factor': defect_interaction_factor,
            'metadata': {
                'generated': True,
                'base_material': 'NCM811',
                'generation_method': 'physics_constrained'
            }
        }
    
    def _generate_atom_features_with_vacancy(self, base_params: Dict[str, Any], 
                                           vacancy_conc: float) -> np.ndarray:
        """生成带空位的原子特征"""
        # 简化的原子特征生成
        n_atoms = 20  # 假设超胞中有20个原子
        n_features = 92  # 原子特征维度
        
        # 基础特征
        base_features = np.random.normal(0, 0.1, (n_atoms, n_features))
        
        # 根据空位浓度调整特征
        vacancy_atoms = int(n_atoms * vacancy_conc)
        if vacancy_atoms > 0:
            # 某些原子位置变为空位（特征置零或特殊值）
            vacancy_indices = np.random.choice(n_atoms, vacancy_atoms, replace=False)
            base_features[vacancy_indices] *= 0.1  # 大幅减小特征值模拟空位
        
        return base_features
    
    def _generate_atom_features_with_migration(self, base_params: Dict[str, Any], 
                                             migration_frac: float) -> np.ndarray:
        """生成带迁移的原子特征"""
        n_atoms = 20
        n_features = 92
        
        base_features = np.random.normal(0, 0.1, (n_atoms, n_features))
        
        # 模拟原子迁移：某些原子的特征发生显著变化
        migrated_atoms = int(n_atoms * migration_frac)
        if migrated_atoms > 0:
            migrated_indices = np.random.choice(n_atoms, migrated_atoms, replace=False)
            # 迁移原子的特征发生变化
            base_features[migrated_indices] += np.random.normal(0, 0.3, (migrated_atoms, n_features))
        
        return base_features
    
    def _generate_atom_features_with_o_vacancy(self, base_params: Dict[str, Any], 
                                             vacancy_conc: float) -> np.ndarray:
        """生成带氧空位的原子特征"""
        n_atoms = 20
        n_features = 92
        
        base_features = np.random.normal(0, 0.1, (n_atoms, n_features))
        
        # 氧空位影响周围原子的电子结构
        o_vacancy_atoms = int(n_atoms * vacancy_conc)
        if o_vacancy_atoms > 0:
            vacancy_indices = np.random.choice(n_atoms, o_vacancy_atoms, replace=False)
            
            # 空位位置
            base_features[vacancy_indices] *= 0.05
            
            # 影响邻近原子
            for idx in vacancy_indices:
                neighbor_indices = [i for i in range(max(0, idx-2), min(n_atoms, idx+3)) if i != idx]
                for neighbor_idx in neighbor_indices:
                    base_features[neighbor_idx] += np.random.normal(0, 0.15, n_features)
        
        return base_features
    
    def _generate_neighbor_features_with_defect(self, base_params: Dict[str, Any], 
                                              defect_type: str, defect_conc: float) -> np.ndarray:
        """生成带缺陷的邻居特征"""
        n_atoms = 20
        max_neighbors = 12
        n_nbr_features = 41
        
        # 基础邻居特征
        neighbor_features = np.random.normal(0, 0.1, (n_atoms, max_neighbors, n_nbr_features))
        
        # 根据缺陷类型调整邻居特征
        if defect_type == 'Li_vacancy':
            # 锂空位会改变局部配位环境
            affected_fraction = defect_conc * 2  # 影响范围
            n_affected = int(n_atoms * max_neighbors * affected_fraction)
            
            # 随机选择受影响的邻居对
            for _ in range(n_affected):
                atom_idx = np.random.randint(0, n_atoms)
                nbr_idx = np.random.randint(0, max_neighbors)
                neighbor_features[atom_idx, nbr_idx] *= np.random.uniform(0.5, 1.5)
        
        elif defect_type == 'Ni_migration':
            # 镍迁移改变键长和配位
            affected_fraction = defect_conc * 3
            n_affected = int(n_atoms * max_neighbors * affected_fraction)
            
            for _ in range(n_affected):
                atom_idx = np.random.randint(0, n_atoms)
                nbr_idx = np.random.randint(0, max_neighbors)
                # 更大的变化幅度
                neighbor_features[atom_idx, nbr_idx] += np.random.normal(0, 0.3, n_nbr_features)
        
        elif defect_type == 'O_vacancy':
            # 氧空位影响金属-氧键
            affected_fraction = defect_conc * 4  # 氧缺失影响多个金属原子
            n_affected = int(n_atoms * max_neighbors * affected_fraction)
            
            for _ in range(n_affected):
                atom_idx = np.random.randint(0, n_atoms)
                nbr_idx = np.random.randint(0, max_neighbors)
                # 模拟键的断裂或重组
                neighbor_features[atom_idx, nbr_idx] *= np.random.uniform(0.2, 0.8)
        
        return neighbor_features
    
    def _apply_physics_constraints(self, sample: Dict[str, Any], 
                                 defect_type: str) -> Dict[str, Any]:
        """应用物理约束"""
        # 确保形成能合理
        if sample['formation_energy'] < -6.0:
            sample['formation_energy'] = -6.0 + np.random.uniform(0, 0.5)
        elif sample['formation_energy'] > 2.0:
            sample['formation_energy'] = 2.0 - np.random.uniform(0, 0.5)
        
        # 确保带隙非负
        sample['band_gap'] = max(0, sample['band_gap'])
        
        # 确保弹性模量为正
        sample['bulk_modulus'] = max(50, sample['bulk_modulus'])
        
        # 缺陷特定约束
        if defect_type == 'Li_vacancy' and 'vacancy_concentration' in sample:
            sample['vacancy_concentration'] = np.clip(sample['vacancy_concentration'], 0, 0.5)
        
        return sample


class WeightedLossHandler:
    """
    加权损失处理器
    Weighted Loss Handler for Imbalanced Data
    """
    
    def __init__(self, class_weights: Optional[Dict[Any, float]] = None,
                 auto_weight: bool = True):
        self.class_weights = class_weights
        self.auto_weight = auto_weight
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_class_weights(self, y: np.ndarray, method: str = 'balanced') -> Dict[Any, float]:
        """
        计算类别权重
        Calculate class weights for imbalanced data
        
        Args:
            y: 标签数组
            method: 权重计算方法
            
        Returns:
            class_weights: 类别权重字典
        """
        from collections import Counter
        
        class_counts = Counter(y)
        total_samples = len(y)
        n_classes = len(class_counts)
        
        if method == 'balanced':
            # sklearn风格的平衡权重
            weights = {}
            for class_label, count in class_counts.items():
                weights[class_label] = total_samples / (n_classes * count)
        
        elif method == 'inverse_frequency':
            # 逆频率权重
            weights = {}
            for class_label, count in class_counts.items():
                weights[class_label] = 1.0 / count
        
        elif method == 'effective_number':
            # 有效样本数方法
            beta = 0.99
            weights = {}
            for class_label, count in class_counts.items():
                effective_num = (1 - beta ** count) / (1 - beta)
                weights[class_label] = 1.0 / effective_num
        
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # 归一化权重
        weight_sum = sum(weights.values())
        normalized_weights = {k: v / weight_sum * n_classes for k, v in weights.items()}
        
        self.logger.info(f"Calculated class weights ({method}): {normalized_weights}")
        
        return normalized_weights
    
    def create_weighted_loss_function(self, class_weights: Dict[Any, float],
                                    base_loss: str = 'mse') -> nn.Module:
        """创建加权损失函数"""
        if base_loss == 'mse':
            return WeightedMSELoss(class_weights)
        elif base_loss == 'cross_entropy':
            return WeightedCrossEntropyLoss(class_weights)
        elif base_loss == 'focal':
            return FocalLoss(class_weights)
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")


class WeightedMSELoss(nn.Module):
    """加权MSE损失"""
    
    def __init__(self, class_weights: Dict[Any, float]):
        super(WeightedMSELoss, self).__init__()
        self.class_weights = class_weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        mse_loss = (predictions - targets) ** 2
        
        if class_labels is not None:
            # 应用类别权重
            weights = torch.ones_like(mse_loss)
            for class_label, weight in self.class_weights.items():
                mask = (class_labels == class_label)
                weights[mask] = weight
            
            weighted_loss = mse_loss * weights
            return weighted_loss.mean()
        else:
            return mse_loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """加权交叉熵损失"""
    
    def __init__(self, class_weights: Dict[Any, float]):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 构建权重张量
        weight_tensor = torch.ones(len(self.class_weights))
        for class_label, weight in self.class_weights.items():
            weight_tensor[class_label] = weight
        
        return nn.CrossEntropyLoss(weight=weight_tensor)(predictions, targets)


class FocalLoss(nn.Module):
    """Focal Loss用于处理极度不平衡数据"""
    
    def __init__(self, class_weights: Dict[Any, float], alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(reduction='none')(predictions, targets)
        pt = torch.exp(-ce_loss)
        
        # 应用focal权重
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # 应用类别权重
        class_weight = torch.ones_like(ce_loss)
        for class_label, weight in self.class_weights.items():
            mask = (targets == class_label)
            class_weight[mask] = weight
        
        focal_loss = focal_weight * class_weight * ce_loss
        
        return focal_loss.mean()


class DataBalanceOrchestrator:
    """
    数据平衡协调器
    Data Balance Orchestrator
    """
    
    def __init__(self, strategy: str = 'combined'):
        self.strategy = strategy
        
        # 组件初始化
        self.materials_smote = MaterialsSMOTE()
        self.physics_generator = PhysicsConstrainedDataGenerator()
        self.weight_handler = WeightedLossHandler()
        
        self.logger = logging.getLogger(__name__)
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray,
                       structure_info: Optional[List[Dict]] = None,
                       target_balance_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        平衡数据集
        Balance dataset using multiple strategies
        
        Args:
            X: 特征矩阵
            y: 标签
            structure_info: 结构信息
            target_balance_ratio: 目标平衡比例
            
        Returns:
            X_balanced, y_balanced: 平衡后的数据
        """
        self.logger.info(f"Starting dataset balancing with strategy: {self.strategy}")
        
        # 分析初始分布
        initial_distribution = Counter(y)
        self.logger.info(f"Initial distribution: {initial_distribution}")
        
        if self.strategy == 'smote_only':
            X_balanced, y_balanced = self.materials_smote.fit_resample(X, y, structure_info)
        
        elif self.strategy == 'generation_only':
            X_balanced, y_balanced = self._generation_only_balance(X, y, target_balance_ratio)
        
        elif self.strategy == 'combined':
            X_balanced, y_balanced = self._combined_balance(X, y, structure_info, target_balance_ratio)
        
        else:
            raise ValueError(f"Unknown balancing strategy: {self.strategy}")
        
        # 分析最终分布
        final_distribution = Counter(y_balanced)
        self.logger.info(f"Final distribution: {final_distribution}")
        
        return X_balanced, y_balanced
    
    def _generation_only_balance(self, X: np.ndarray, y: np.ndarray,
                               target_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """仅使用生成方法平衡"""
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        target_samples = int(class_counts[majority_class] * target_ratio)
        
        X_balanced = X.copy()
        y_balanced = y.copy()
        
        for class_label, count in class_counts.items():
            if class_label != majority_class and count < target_samples:
                needed_samples = target_samples - count
                
                # 根据类别生成样本
                if 'vacancy' in str(class_label).lower():
                    defect_type = 'Li_vacancy'
                elif 'migration' in str(class_label).lower():
                    defect_type = 'Ni_migration'
                else:
                    defect_type = 'Li_vacancy'  # 默认
                
                generated_samples = self.physics_generator.generate_rare_defect_samples(
                    needed_samples, defect_type
                )
                
                # 转换为特征矩阵格式
                generated_X = self._convert_generated_to_features(generated_samples)
                generated_y = np.full(len(generated_samples), class_label)
                
                X_balanced = np.vstack([X_balanced, generated_X])
                y_balanced = np.hstack([y_balanced, generated_y])
        
        return X_balanced, y_balanced
    
    def _combined_balance(self, X: np.ndarray, y: np.ndarray,
                        structure_info: Optional[List[Dict]],
                        target_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """组合方法平衡"""
        # 第一步：使用SMOTE
        X_smote, y_smote = self.materials_smote.fit_resample(X, y, structure_info)
        
        # 第二步：检查是否还需要生成
        smote_distribution = Counter(y_smote)
        majority_count = max(smote_distribution.values())
        
        X_final = X_smote.copy()
        y_final = y_smote.copy()
        
        for class_label, count in smote_distribution.items():
            target_count = int(majority_count * target_ratio)
            if count < target_count:
                needed_samples = target_count - count
                
                # 生成物理约束样本
                defect_type = self._infer_defect_type_from_label(class_label)
                generated_samples = self.physics_generator.generate_rare_defect_samples(
                    needed_samples, defect_type
                )
                
                generated_X = self._convert_generated_to_features(generated_samples)
                generated_y = np.full(len(generated_samples), class_label)
                
                X_final = np.vstack([X_final, generated_X])
                y_final = np.hstack([y_final, generated_y])
        
        return X_final, y_final
    
    def _convert_generated_to_features(self, generated_samples: List[Dict]) -> np.ndarray:
        """将生成的样本转换为特征矩阵"""
        # 简化的转换过程
        feature_matrix = []
        
        for sample in generated_samples:
            # 提取关键特征
            features = [
                sample.get('formation_energy', 0),
                sample.get('band_gap', 0),
                sample.get('bulk_modulus', 100),
                sample.get('vacancy_concentration', 0),
                sample.get('migration_fraction', 0),
            ]
            
            # 添加原子特征的统计信息
            if 'atom_features' in sample:
                atom_features = sample['atom_features']
                features.extend([
                    np.mean(atom_features),
                    np.std(atom_features),
                    np.max(atom_features),
                    np.min(atom_features)
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # 补充到目标维度（这里假设需要匹配原始特征维度）
            while len(features) < 50:  # 假设目标维度为50
                features.append(np.random.normal(0, 0.01))
            
            feature_matrix.append(features[:50])  # 截断到目标维度
        
        return np.array(feature_matrix)
    
    def _infer_defect_type_from_label(self, class_label: Any) -> str:
        """从类别标签推断缺陷类型"""
        label_str = str(class_label).lower()
        
        if 'li' in label_str and 'vac' in label_str:
            return 'Li_vacancy'
        elif 'ni' in label_str and 'mig' in label_str:
            return 'Ni_migration'
        elif 'o' in label_str and 'vac' in label_str:
            return 'O_vacancy'
        else:
            return 'Li_vacancy'  # 默认


# 使用示例
def example_usage():
    """使用示例"""
    # 创建模拟不平衡数据
    np.random.seed(42)
    
    # 创建不平衡数据集
    n_majority = 800
    n_minority1 = 50
    n_minority2 = 20
    
    X_majority = np.random.normal(0, 1, (n_majority, 10))
    X_minority1 = np.random.normal(2, 1, (n_minority1, 10))
    X_minority2 = np.random.normal(-2, 1, (n_minority2, 10))
    
    X = np.vstack([X_majority, X_minority1, X_minority2])
    y = np.hstack([
        np.zeros(n_majority),
        np.ones(n_minority1),
        np.full(n_minority2, 2)
    ])
    
    print(f"Original distribution: {Counter(y)}")
    
    # 1. 使用Materials SMOTE
    materials_smote = MaterialsSMOTE(k_neighbors=3)
    X_smote, y_smote = materials_smote.fit_resample(X, y)
    print(f"After SMOTE: {Counter(y_smote)}")
    
    # 2. 使用物理约束生成器
    physics_generator = PhysicsConstrainedDataGenerator()
    rare_samples = physics_generator.generate_rare_defect_samples(
        n_samples=30, defect_type='Li_vacancy'
    )
    print(f"Generated {len(rare_samples)} rare defect samples")
    
    # 3. 计算类别权重
    weight_handler = WeightedLossHandler()
    class_weights = weight_handler.calculate_class_weights(y, method='balanced')
    print(f"Class weights: {class_weights}")
    
    # 4. 使用数据平衡协调器
    orchestrator = DataBalanceOrchestrator(strategy='combined')
    X_balanced, y_balanced = orchestrator.balance_dataset(X, y, target_balance_ratio=0.5)
    print(f"Final balanced distribution: {Counter(y_balanced)}")
    
    # 5. 创建加权损失函数
    weighted_loss = weight_handler.create_weighted_loss_function(class_weights, 'focal')
    print(f"Created weighted loss function: {type(weighted_loss).__name__}")


if __name__ == "__main__":
    example_usage() 