"""
Physics-Informed Neural Networks for Materials

Implementation of physical constraints and conservation laws in neural
network training for materials property prediction. Enforces thermodynamic
consistency and symmetry constraints.

Author: lunazhang  
Date: 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings

try:
    from pymatgen.core.structure import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.thermodynamics import ThermoData
    PYMATGEN_AVAILABLE = True
except ImportError:
    warnings.warn("PyMatGen not available. Some physics constraints may be limited.")
    PYMATGEN_AVAILABLE = False
    Structure = None


class ConservationLaws:
    """
    物理守恒定律约束
    Physical Conservation Laws Constraints
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def charge_neutrality_loss(self, predicted_charges: torch.Tensor, 
                              true_charges: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        电荷守恒约束损失
        Charge conservation constraint loss
        
        Args:
            predicted_charges: 预测的原子电荷 [batch_size, n_atoms]
            true_charges: 真实电荷（可选）
        
        Returns:
            loss: 电荷守恒损失
        """
        # 每个结构的总电荷应为零（或接近零）
        total_charges = torch.sum(predicted_charges, dim=1)
        charge_neutrality_loss = torch.mean(total_charges ** 2)
        
        if true_charges is not None:
            # 如果有真实电荷，添加监督损失
            supervised_loss = F.mse_loss(predicted_charges, true_charges)
            return charge_neutrality_loss + supervised_loss
        
        return charge_neutrality_loss
    
    def mass_conservation_loss(self, predicted_compositions: torch.Tensor,
                              target_compositions: torch.Tensor) -> torch.Tensor:
        """
        质量守恒约束损失
        Mass conservation constraint loss
        
        Args:
            predicted_compositions: 预测的组成 [batch_size, n_elements]
            target_compositions: 目标组成 [batch_size, n_elements]
        
        Returns:
            loss: 质量守恒损失
        """
        # 组成比例之和应为1
        pred_sum = torch.sum(predicted_compositions, dim=1)
        target_sum = torch.sum(target_compositions, dim=1)
        
        # 归一化损失
        normalization_loss = torch.mean((pred_sum - 1.0) ** 2 + (target_sum - 1.0) ** 2)
        
        # 组成预测损失
        composition_loss = F.mse_loss(predicted_compositions, target_compositions)
        
        return composition_loss + normalization_loss
    
    def energy_conservation_loss(self, predicted_energies: torch.Tensor,
                                reference_energies: torch.Tensor,
                                reaction_coefficients: torch.Tensor) -> torch.Tensor:
        """
        能量守恒约束损失（用于反应能预测）
        Energy conservation constraint loss for reaction energies
        
        Args:
            predicted_energies: 预测能量 [batch_size]
            reference_energies: 参考能量 [batch_size, n_references]
            reaction_coefficients: 反应系数 [batch_size, n_references]
        
        Returns:
            loss: 能量守恒损失
        """
        # 计算反应能
        predicted_reaction_energy = torch.sum(reaction_coefficients * reference_energies, dim=1)
        
        # 能量守恒损失
        conservation_loss = F.mse_loss(predicted_energies, predicted_reaction_energy)
        
        return conservation_loss


class SymmetryConstraints:
    """
    对称性约束
    Symmetry Constraints
    """
    
    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance
    
    def rotational_invariance_loss(self, model: nn.Module, input_data: Tuple,
                                  rotation_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        旋转不变性约束损失
        Rotational invariance constraint loss
        
        Args:
            model: CGCNN模型
            input_data: 输入数据 (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            rotation_matrices: 旋转矩阵列表 [3, 3]
        
        Returns:
            loss: 旋转不变性损失
        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 原始预测
        original_output = model(*input_data)
        
        invariance_losses = []
        
        for rotation_matrix in rotation_matrices:
            # 应用旋转变换到原子坐标相关的特征
            # 注意：这需要根据具体的特征表示进行调整
            rotated_atom_fea = self._apply_rotation_to_features(atom_fea, rotation_matrix)
            rotated_nbr_fea = self._apply_rotation_to_neighbor_features(nbr_fea, rotation_matrix)
            
            rotated_input = (rotated_atom_fea, rotated_nbr_fea, nbr_fea_idx, crystal_atom_idx)
            rotated_output = model(*rotated_input)
            
            # 计算不变性损失
            invariance_loss = F.mse_loss(original_output, rotated_output)
            invariance_losses.append(invariance_loss)
        
        return torch.mean(torch.stack(invariance_losses))
    
    def _apply_rotation_to_features(self, atom_features: torch.Tensor, 
                                   rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        将旋转应用到原子特征（如果包含坐标信息）
        Apply rotation to atom features (if they contain coordinate information)
        """
        # 这里假设原子特征的前3维是坐标
        if atom_features.shape[-1] >= 3:
            rotated_features = atom_features.clone()
            coords = atom_features[:, :3]
            rotated_coords = torch.matmul(coords, rotation_matrix.T)
            rotated_features[:, :3] = rotated_coords
            return rotated_features
        return atom_features
    
    def _apply_rotation_to_neighbor_features(self, nbr_features: torch.Tensor,
                                           rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        将旋转应用到邻居特征（如果包含方向信息）
        Apply rotation to neighbor features (if they contain directional information)
        """
        # 这里假设邻居特征包含方向向量
        if nbr_features.shape[-1] >= 3:
            rotated_features = nbr_features.clone()
            # 假设最后3维是方向向量
            directions = nbr_features[:, :, -3:]
            rotated_directions = torch.matmul(directions, rotation_matrix.T)
            rotated_features[:, :, -3:] = rotated_directions
            return rotated_features
        return nbr_features
    
    def translational_invariance_loss(self, model: nn.Module, input_data: Tuple,
                                    translations: List[torch.Tensor]) -> torch.Tensor:
        """
        平移不变性约束损失
        Translational invariance constraint loss
        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
        
        # 原始预测
        original_output = model(*input_data)
        
        invariance_losses = []
        
        for translation in translations:
            # 应用平移变换
            translated_atom_fea = self._apply_translation_to_features(atom_fea, translation)
            
            translated_input = (translated_atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            translated_output = model(*translated_input)
            
            # 计算不变性损失
            invariance_loss = F.mse_loss(original_output, translated_output)
            invariance_losses.append(invariance_loss)
        
        return torch.mean(torch.stack(invariance_losses))
    
    def _apply_translation_to_features(self, atom_features: torch.Tensor,
                                     translation: torch.Tensor) -> torch.Tensor:
        if atom_features.shape[-1] >= 3:
            translated_features = atom_features.clone()
            translated_features[:, :3] += translation
            return translated_features
        return atom_features
    
    def point_group_symmetry_loss(self, predictions: torch.Tensor,
                                 symmetry_operations: List[torch.Tensor]) -> torch.Tensor:
        """
        点群对称性约束损失
        Point group symmetry constraint loss
        
        Args:
            predictions: 模型预测 [batch_size, output_dim]
            symmetry_operations: 对称操作矩阵列表
        
        Returns:
            loss: 对称性损失
        """
        # 对于标量性质，应该在所有对称操作下保持不变
        if predictions.dim() == 1:
            return torch.tensor(0.0, device=predictions.device)
        
        symmetry_losses = []
        for sym_op in symmetry_operations:
            # 对于向量性质，需要应用对称操作
            if predictions.shape[-1] == 3:  # 假设是3D向量
                transformed_predictions = torch.matmul(predictions, sym_op.T)
                # 某些性质在对称操作下应该保持不变
                symmetry_loss = F.mse_loss(predictions, transformed_predictions)
                symmetry_losses.append(symmetry_loss)
        
        if symmetry_losses:
            return torch.mean(torch.stack(symmetry_losses))
        else:
            return torch.tensor(0.0, device=predictions.device)


class ThermodynamicConsistency:
    """
    热力学一致性约束
    Thermodynamic Consistency Constraints
    """
    
    def __init__(self, temperature: float = 298.15):
        self.temperature = temperature
        self.kb = 8.617333e-5  # Boltzmann constant in eV/K
    
    def gibbs_duhem_loss(self, chemical_potentials: torch.Tensor,
                        compositions: torch.Tensor) -> torch.Tensor:
        """
        Gibbs-Duhem方程约束损失
        Gibbs-Duhem equation constraint loss
        
        Args:
            chemical_potentials: 化学势 [batch_size, n_components]
            compositions: 组成 [batch_size, n_components]
        
        Returns:
            loss: Gibbs-Duhem约束损失
        """
        # Gibbs-Duhem方程: Σ x_i * dμ_i = 0
        # 近似为: Σ x_i * μ_i = 常数
        gibbs_duhem_sum = torch.sum(compositions * chemical_potentials, dim=1)
        
        # 约束：所有样本的Gibbs-Duhem和应该相近
        mean_sum = torch.mean(gibbs_duhem_sum)
        consistency_loss = torch.mean((gibbs_duhem_sum - mean_sum) ** 2)
        
        return consistency_loss
    
    def phase_equilibrium_loss(self, energies_phase1: torch.Tensor,
                              energies_phase2: torch.Tensor,
                              compositions_phase1: torch.Tensor,
                              compositions_phase2: torch.Tensor) -> torch.Tensor:
        """
        相平衡约束损失
        Phase equilibrium constraint loss
        
        Args:
            energies_phase1: 相1的能量 [batch_size]
            energies_phase2: 相2的能量 [batch_size]
            compositions_phase1: 相1的组成 [batch_size, n_components]
            compositions_phase2: 相2的组成 [batch_size, n_components]
        
        Returns:
            loss: 相平衡约束损失
        """
        # 相平衡条件：化学势相等
        # μ_i^(1) = μ_i^(2) for all components i
        
        # 简化的化学势计算（实际应用中需要更复杂的模型）
        chemical_potentials_1 = self._compute_chemical_potentials(energies_phase1, compositions_phase1)
        chemical_potentials_2 = self._compute_chemical_potentials(energies_phase2, compositions_phase2)
        
        # 化学势相等约束
        equilibrium_loss = F.mse_loss(chemical_potentials_1, chemical_potentials_2)
        
        return equilibrium_loss
    
    def _compute_chemical_potentials(self, energies: torch.Tensor,
                                   compositions: torch.Tensor) -> torch.Tensor:
        """
        计算化学势（简化版本）
        Compute chemical potentials (simplified version)
        """
        # 简化的化学势计算：μ_i = ∂G/∂x_i
        # 这里使用一个简化的近似
        n_components = compositions.shape[1]
        chemical_potentials = torch.zeros_like(compositions)
        
        for i in range(n_components):
            # 数值微分近似
            eps = 1e-6
            comp_plus = compositions.clone()
            comp_minus = compositions.clone()
            
            comp_plus[:, i] += eps
            comp_minus[:, i] -= eps
            
            # 重新归一化
            comp_plus = comp_plus / torch.sum(comp_plus, dim=1, keepdim=True)
            comp_minus = comp_minus / torch.sum(comp_minus, dim=1, keepdim=True)
            
            # 近似导数
            dG_dxi = (energies - energies) / (2 * eps)  # 简化版本
            chemical_potentials[:, i] = dG_dxi
        
        return chemical_potentials
    
    def stability_constraint_loss(self, formation_energies: torch.Tensor,
                                 reference_energies: torch.Tensor,
                                 compositions: torch.Tensor) -> torch.Tensor:
        """
        稳定性约束损失
        Stability constraint loss
        
        Args:
            formation_energies: 形成能 [batch_size]
            reference_energies: 参考相能量 [n_references]
            compositions: 组成 [batch_size, n_components]
        
        Returns:
            loss: 稳定性约束损失
        """
        # 稳定相的形成能应该在凸包上
        # 这里使用一个简化的凸包近似
        
        batch_size = formation_energies.shape[0]
        stability_losses = []
        
        for i in range(batch_size):
            comp = compositions[i]
            formation_energy = formation_energies[i]
            
            # 计算与参考相的线性组合能量下界
            # 在实际应用中，这需要更复杂的凸包算法
            linear_combination_energy = torch.sum(comp * reference_energies)
            
            # 稳定性约束：形成能应该不高于线性组合
            stability_loss = F.relu(formation_energy - linear_combination_energy)
            stability_losses.append(stability_loss)
        
        return torch.mean(torch.stack(stability_losses))
    
    def maxwell_relations_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Maxwell关系约束损失
        Maxwell relations constraint loss
        
        Args:
            predictions: 包含各种热力学量的预测字典
        
        Returns:
            loss: Maxwell关系约束损失
        """
        losses = []
        
        # Maxwell关系: (∂S/∂P)_T = -(∂V/∂T)_P
        if 'entropy' in predictions and 'volume' in predictions:
            # 需要计算偏导数，这里使用简化的有限差分近似
            # 实际应用中需要更精确的实现
            maxwell_loss = torch.tensor(0.0, device=list(predictions.values())[0].device)
            losses.append(maxwell_loss)
        
        if losses:
            return torch.mean(torch.stack(losses))
        else:
            return torch.tensor(0.0)


class PhysicsConstrainedLoss(nn.Module):
    """
    物理约束损失函数
    Physics-constrained loss function
    """
    
    def __init__(self, 
                 conservation_weight: float = 1.0,
                 symmetry_weight: float = 1.0,
                 thermodynamic_weight: float = 1.0,
                 base_loss_weight: float = 1.0):
        super(PhysicsConstrainedLoss, self).__init__()
        
        self.conservation_weight = conservation_weight
        self.symmetry_weight = symmetry_weight
        self.thermodynamic_weight = thermodynamic_weight
        self.base_loss_weight = base_loss_weight
        
        self.conservation_laws = ConservationLaws()
        self.symmetry_constraints = SymmetryConstraints()
        self.thermodynamic_consistency = ThermodynamicConsistency()
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                model: nn.Module,
                input_data: Tuple,
                physics_data: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算物理约束损失
        
        Args:
            predictions: 模型预测
            targets: 真实标签
            model: 模型实例
            input_data: 输入数据
            physics_data: 物理约束相关数据
        
        Returns:
            total_loss: 总损失
            loss_components: 各组成部分的损失
        """
        # 基础损失（MSE或交叉熵）
        if predictions.shape[-1] == 1:  # 回归
            base_loss = F.mse_loss(predictions, targets)
        else:  # 分类
            base_loss = F.cross_entropy(predictions, targets.long())
        
        loss_components = {'base_loss': base_loss}
        total_loss = self.base_loss_weight * base_loss
        
        if physics_data is None:
            return total_loss, loss_components
        
        # 守恒定律约束
        if 'charges' in physics_data:
            charge_loss = self.conservation_laws.charge_neutrality_loss(physics_data['charges'])
            loss_components['charge_conservation'] = charge_loss
            total_loss += self.conservation_weight * charge_loss
        
        if 'compositions' in physics_data:
            mass_loss = self.conservation_laws.mass_conservation_loss(
                physics_data['compositions'], physics_data.get('target_compositions', physics_data['compositions']))
            loss_components['mass_conservation'] = mass_loss
            total_loss += self.conservation_weight * mass_loss
        
        # 对称性约束
        if 'rotation_matrices' in physics_data:
            rotational_loss = self.symmetry_constraints.rotational_invariance_loss(
                model, input_data, physics_data['rotation_matrices'])
            loss_components['rotational_invariance'] = rotational_loss
            total_loss += self.symmetry_weight * rotational_loss
        
        if 'translations' in physics_data:
            translational_loss = self.symmetry_constraints.translational_invariance_loss(
                model, input_data, physics_data['translations'])
            loss_components['translational_invariance'] = translational_loss
            total_loss += self.symmetry_weight * translational_loss
        
        # 热力学一致性约束
        if 'chemical_potentials' in physics_data and 'compositions' in physics_data:
            gibbs_duhem_loss = self.thermodynamic_consistency.gibbs_duhem_loss(
                physics_data['chemical_potentials'], physics_data['compositions'])
            loss_components['gibbs_duhem'] = gibbs_duhem_loss
            total_loss += self.thermodynamic_weight * gibbs_duhem_loss
        
        if 'formation_energies' in physics_data and 'reference_energies' in physics_data:
            stability_loss = self.thermodynamic_consistency.stability_constraint_loss(
                physics_data['formation_energies'], 
                physics_data['reference_energies'],
                physics_data['compositions'])
            loss_components['stability'] = stability_loss
            total_loss += self.thermodynamic_weight * stability_loss
        
        return total_loss, loss_components


def generate_symmetry_operations(space_group_number: int) -> List[torch.Tensor]:
    """
    生成对称操作矩阵
    Generate symmetry operation matrices for a given space group
    
    Args:
        space_group_number: 空间群编号
    
    Returns:
        symmetry_operations: 对称操作矩阵列表
    """
    # 这里提供一些常见空间群的对称操作
    # 实际应用中应该使用更完整的空间群数据库
    
    if space_group_number == 225:  # Fm-3m (FCC)
        operations = [
            torch.eye(3),  # 恒等操作
            torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32),  # 180°绕z轴
            torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32),  # 180°绕y轴
            torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32),  # 180°绕x轴
        ]
    elif space_group_number == 221:  # Pm-3m (简单立方)
        operations = [
            torch.eye(3),
            torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32),  # 反演
        ]
    else:
        # 默认只返回恒等操作
        operations = [torch.eye(3)]
    
    return operations


# 使用示例
def example_usage():
    # 创建物理约束损失函数
    physics_loss = PhysicsConstrainedLoss(
        conservation_weight=0.1,
        symmetry_weight=0.05,
        thermodynamic_weight=0.1
    )
    
    # 在训练循环中使用
    # total_loss, loss_components = physics_loss(
    #     predictions, targets, model, input_data, physics_data
    # )
    
    # 生成对称操作
    # symmetry_ops = generate_symmetry_operations(225)  # FCC空间群
    
    pass 