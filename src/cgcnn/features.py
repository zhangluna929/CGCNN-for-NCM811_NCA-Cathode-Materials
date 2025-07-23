"""
Feature Engineering for Crystal Structures

Advanced descriptors and feature extraction methods for crystalline materials.
Implements structural, chemical, and topological descriptors for enhanced
materials property prediction.

Author: lunazhang
Date: 2023
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

try:
    from pymatgen.core.structure import Structure
    from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.core.periodic_table import Element
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    Structure = None

from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
import warnings


class StructuralDescriptors:
    """
    结构描述符计算器
    Structural Descriptors Calculator
    """
    
    def __init__(self, cutoff_radius: float = 8.0, max_neighbors: int = 12):
        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors
        self.voronoi_nn = VoronoiNN()
        self.crystal_nn = CrystalNN()
    
    def compute_radial_distribution_function(self, structure: Structure, 
                                           r_max: float = 10.0, dr: float = 0.1) -> np.ndarray:
        """
        计算径向分布函数 (RDF)
        Compute Radial Distribution Function
        """
        distances = structure.distance_matrix.flatten()
        distances = distances[distances > 0]  # 排除自身距离
        distances = distances[distances <= r_max]
        
        r_bins = np.arange(0, r_max + dr, dr)
        hist, _ = np.histogram(distances, bins=r_bins)
        
        # 归一化
        volume = structure.volume
        n_atoms = len(structure)
        density = n_atoms / volume
        
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        shell_volumes = 4 * np.pi * r_centers**2 * dr
        expected_counts = density * shell_volumes * n_atoms
        
        rdf = hist / (expected_counts + 1e-10)
        return rdf
    
    def compute_angular_distribution_function(self, structure: Structure, 
                                            site_idx: int) -> np.ndarray:
        """
        计算角度分布函数 (ADF)
        Compute Angular Distribution Function for a specific site
        """
        neighbors = structure.get_neighbors(structure[site_idx], self.cutoff_radius)
        
        if len(neighbors) < 2:
            return np.zeros(180)  # 返回零向量如果邻居太少
        
        angles = []
        center = structure[site_idx].coords
        
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                vec1 = neighbors[i].coords - center
                vec2 = neighbors[j].coords - center
                
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1, 1)  # 数值稳定性
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
        
        # 创建角度分布直方图
        hist, _ = np.histogram(angles, bins=np.arange(0, 181, 1))
        return hist / (np.sum(hist) + 1e-10)  # 归一化
    
    def compute_coordination_features(self, structure: Structure) -> Dict[str, np.ndarray]:
        """
        计算配位环境特征
        Compute coordination environment features
        """
        features = {
            'coordination_numbers': [],
            'bond_lengths_mean': [],
            'bond_lengths_std': [],
            'bond_angles_mean': [],
            'bond_angles_std': [],
            'polyhedron_volumes': []
        }
        
        for i, site in enumerate(structure):
            try:
                # 使用Voronoi方法计算配位数
                voronoi_neighbors = self.voronoi_nn.get_nn_info(structure, i)
                coordination_number = len(voronoi_neighbors)
                
                # 计算键长统计
                bond_lengths = [neighbor['weight'] for neighbor in voronoi_neighbors]
                bond_lengths_mean = np.mean(bond_lengths) if bond_lengths else 0
                bond_lengths_std = np.std(bond_lengths) if len(bond_lengths) > 1 else 0
                
                # 计算键角统计
                angles = self.compute_angular_distribution_function(structure, i)
                angle_indices = np.where(angles > 0)[0]
                if len(angle_indices) > 0:
                    weighted_angles = angle_indices * angles[angle_indices]
                    bond_angles_mean = np.sum(weighted_angles) / np.sum(angles[angle_indices])
                    # 计算加权标准差
                    variance = np.sum(angles[angle_indices] * (angle_indices - bond_angles_mean)**2) / np.sum(angles[angle_indices])
                    bond_angles_std = np.sqrt(variance)
                else:
                    bond_angles_mean = 0
                    bond_angles_std = 0
                
                # 计算多面体体积（近似）
                if len(voronoi_neighbors) >= 4:
                    neighbor_coords = np.array([neighbor['site'].coords for neighbor in voronoi_neighbors])
                    center_coords = site.coords
                    # 使用平均距离的立方作为体积的近似
                    avg_distance = np.mean(bond_lengths)
                    polyhedron_volume = (4/3) * np.pi * avg_distance**3
                else:
                    polyhedron_volume = 0
                
                features['coordination_numbers'].append(coordination_number)
                features['bond_lengths_mean'].append(bond_lengths_mean)
                features['bond_lengths_std'].append(bond_lengths_std)
                features['bond_angles_mean'].append(bond_angles_mean)
                features['bond_angles_std'].append(bond_angles_std)
                features['polyhedron_volumes'].append(polyhedron_volume)
                
            except Exception as e:
                warnings.warn(f"Error computing coordination features for site {i}: {e}")
                # 填充默认值
                features['coordination_numbers'].append(0)
                features['bond_lengths_mean'].append(0)
                features['bond_lengths_std'].append(0)
                features['bond_angles_mean'].append(0)
                features['bond_angles_std'].append(0)
                features['polyhedron_volumes'].append(0)
        
        # 转换为numpy数组
        for key in features:
            features[key] = np.array(features[key])
        
        return features
    
    def compute_packing_efficiency(self, structure: Structure) -> float:
        """
        计算堆积效率
        Compute packing efficiency
        """
        try:
            # 简化计算：使用离子半径估算
            total_atomic_volume = 0
            for site in structure:
                # 使用简单的离子半径估计
                ionic_radius = site.specie.ionic_radius if hasattr(site.specie, 'ionic_radius') and site.specie.ionic_radius else 1.0
                total_atomic_volume += (4/3) * np.pi * ionic_radius**3
            
            packing_efficiency = total_atomic_volume / structure.volume
            return min(packing_efficiency, 1.0)  # 限制在0-1之间
        except:
            return 0.0
    
    def compute_structural_fingerprint(self, structure: Structure) -> np.ndarray:
        """
        计算结构指纹
        Compute structural fingerprint combining multiple descriptors
        """
        # RDF特征
        rdf = self.compute_radial_distribution_function(structure)
        rdf_features = [
            np.max(rdf),  # 最大峰值
            np.argmax(rdf) * 0.1,  # 第一个峰位置
            np.sum(rdf[:10]),  # 短程有序
            np.sum(rdf[10:]),  # 长程有序
        ]
        
        # 配位环境特征
        coord_features = self.compute_coordination_features(structure)
        coord_stats = [
            np.mean(coord_features['coordination_numbers']),
            np.std(coord_features['coordination_numbers']),
            np.mean(coord_features['bond_lengths_mean']),
            np.std(coord_features['bond_lengths_mean']),
            np.mean(coord_features['bond_angles_mean']),
            np.std(coord_features['bond_angles_mean']),
        ]
        
        # 堆积效率
        packing_eff = self.compute_packing_efficiency(structure)
        
        # 其他结构特征
        lattice_features = [
            structure.volume / len(structure),  # 每原子体积
            structure.density,  # 密度
            np.mean([structure.lattice.a, structure.lattice.b, structure.lattice.c]),  # 平均晶格参数
            np.std([structure.lattice.a, structure.lattice.b, structure.lattice.c]),  # 晶格参数变化
        ]
        
        # 组合所有特征
        fingerprint = np.array(rdf_features + coord_stats + [packing_eff] + lattice_features)
        return fingerprint


class ChemicalDescriptors:
    """
    化学描述符计算器
    Chemical Descriptors Calculator
    """
    
    def __init__(self):
        self.element_properties = self._load_element_properties()
    
    def _load_element_properties(self) -> Dict[str, Dict[str, float]]:
        """
        加载元素属性数据
        Load element properties data
        """
        # 简化的元素属性，实际应用中可以使用更全面的数据库
        properties = {
            'electronegativity': {
                'H': 2.20, 'Li': 0.98, 'C': 2.55, 'N': 3.04, 'O': 3.44,
                'F': 3.98, 'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90,
                'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'K': 0.82, 'Ca': 1.00,
                'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83,
                'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65
            },
            'atomic_radius': {
                'H': 0.37, 'Li': 1.52, 'C': 0.70, 'N': 0.65, 'O': 0.60,
                'F': 0.50, 'Na': 1.86, 'Mg': 1.60, 'Al': 1.43, 'Si': 1.11,
                'P': 1.07, 'S': 1.05, 'Cl': 0.99, 'K': 2.27, 'Ca': 1.97,
                'Ti': 1.47, 'V': 1.34, 'Cr': 1.28, 'Mn': 1.27, 'Fe': 1.26,
                'Co': 1.25, 'Ni': 1.24, 'Cu': 1.28, 'Zn': 1.34
            },
            'ionization_energy': {
                'H': 13.60, 'Li': 5.39, 'C': 11.26, 'N': 14.53, 'O': 13.62,
                'F': 17.42, 'Na': 5.14, 'Mg': 7.65, 'Al': 5.99, 'Si': 8.15,
                'P': 10.49, 'S': 10.36, 'Cl': 12.97, 'K': 4.34, 'Ca': 6.11,
                'Ti': 6.83, 'V': 6.75, 'Cr': 6.77, 'Mn': 7.43, 'Fe': 7.90,
                'Co': 7.88, 'Ni': 7.64, 'Cu': 7.73, 'Zn': 9.39
            }
        }
        return properties
    
    def compute_composition_features(self, structure: Structure) -> Dict[str, float]:
        """
        计算组成特征
        Compute composition features
        """
        composition = structure.composition
        elements = [str(el) for el in composition.elements]
        fractions = [composition.get_atomic_fraction(el) for el in elements]
        
        features = {}
        
        # 基本组成统计
        features['n_elements'] = len(elements)
        features['composition_entropy'] = -sum(f * np.log(f) for f in fractions if f > 0)
        
        # 元素属性统计
        for prop_name, prop_dict in self.element_properties.items():
            values = []
            weights = []
            for element, fraction in zip(elements, fractions):
                if element in prop_dict:
                    values.append(prop_dict[element])
                    weights.append(fraction)
            
            if values:
                weighted_values = np.array(values) * np.array(weights)
                features[f'{prop_name}_mean'] = np.average(values, weights=weights)
                features[f'{prop_name}_std'] = np.sqrt(np.average((values - features[f'{prop_name}_mean'])**2, weights=weights))
                features[f'{prop_name}_min'] = np.min(values)
                features[f'{prop_name}_max'] = np.max(values)
                features[f'{prop_name}_range'] = features[f'{prop_name}_max'] - features[f'{prop_name}_min']
            else:
                # 如果没有数据，填充零值
                for suffix in ['_mean', '_std', '_min', '_max', '_range']:
                    features[f'{prop_name}{suffix}'] = 0.0
        
        return features
    
    def compute_bond_features(self, structure: Structure) -> Dict[str, List[float]]:
        """
        计算键特征
        Compute bond features
        """
        bond_features = {
            'bond_lengths': [],
            'electronegativity_differences': [],
            'ionic_character': [],
            'covalent_character': []
        }
        
        # 计算所有原子对的键特征
        for i, site_i in enumerate(structure):
            neighbors = structure.get_neighbors(site_i, 4.0)  # 4Å截断
            
            for neighbor in neighbors:
                site_j = neighbor.site
                distance = neighbor.nn_distance
                
                # 键长
                bond_features['bond_lengths'].append(distance)
                
                # 电负性差异
                element_i = str(site_i.specie)
                element_j = str(site_j.specie)
                
                if (element_i in self.element_properties['electronegativity'] and 
                    element_j in self.element_properties['electronegativity']):
                    
                    en_i = self.element_properties['electronegativity'][element_i]
                    en_j = self.element_properties['electronegativity'][element_j]
                    en_diff = abs(en_i - en_j)
                    bond_features['electronegativity_differences'].append(en_diff)
                    
                    # 离子性程度 (Pauling estimate)
                    ionic_char = 1 - np.exp(-0.25 * en_diff**2)
                    covalent_char = 1 - ionic_char
                    
                    bond_features['ionic_character'].append(ionic_char)
                    bond_features['covalent_character'].append(covalent_char)
                else:
                    bond_features['electronegativity_differences'].append(0)
                    bond_features['ionic_character'].append(0)
                    bond_features['covalent_character'].append(0)
        
        return bond_features
    
    def compute_oxidation_state_features(self, structure: Structure) -> Dict[str, float]:
        """
        计算氧化态特征
        Compute oxidation state features
        """
        try:
            bv_analyzer = BVAnalyzer()
            oxidation_states = bv_analyzer.get_valences(structure)
            
            features = {
                'oxidation_state_mean': np.mean(np.abs(oxidation_states)),
                'oxidation_state_std': np.std(oxidation_states),
                'oxidation_state_range': np.max(oxidation_states) - np.min(oxidation_states),
                'charge_balance': np.sum(oxidation_states) / len(oxidation_states)
            }
        except:
            # 如果无法计算氧化态，返回默认值
            features = {
                'oxidation_state_mean': 0,
                'oxidation_state_std': 0,
                'oxidation_state_range': 0,
                'charge_balance': 0
            }
        
        return features
    
    def compute_chemical_fingerprint(self, structure: Structure) -> np.ndarray:
        """
        计算化学指纹
        Compute chemical fingerprint
        """
        # 组成特征
        comp_features = self.compute_composition_features(structure)
        comp_values = list(comp_features.values())
        
        # 键特征统计
        bond_features = self.compute_bond_features(structure)
        bond_stats = []
        for feature_name, feature_values in bond_features.items():
            if feature_values:
                bond_stats.extend([
                    np.mean(feature_values),
                    np.std(feature_values),
                    np.min(feature_values),
                    np.max(feature_values)
                ])
            else:
                bond_stats.extend([0, 0, 0, 0])
        
        # 氧化态特征
        ox_features = self.compute_oxidation_state_features(structure)
        ox_values = list(ox_features.values())
        
        # 组合所有特征
        fingerprint = np.array(comp_values + bond_stats + ox_values)
        return fingerprint


class AdvancedFeatureExtractor:
    """
    高级特征提取器
    Advanced Feature Extractor combining structural and chemical descriptors
    """
    
    def __init__(self, cutoff_radius: float = 8.0, max_neighbors: int = 12):
        self.structural_desc = StructuralDescriptors(cutoff_radius, max_neighbors)
        self.chemical_desc = ChemicalDescriptors()
    
    def extract_features(self, structure: Structure) -> Dict[str, np.ndarray]:
        """
        提取完整的特征集
        Extract complete feature set
        """
        features = {}
        
        # 结构特征
        features['structural_fingerprint'] = self.structural_desc.compute_structural_fingerprint(structure)
        features['coordination_features'] = self.structural_desc.compute_coordination_features(structure)
        
        # 化学特征
        features['chemical_fingerprint'] = self.chemical_desc.compute_chemical_fingerprint(structure)
        
        # 组合特征
        combined_fingerprint = np.concatenate([
            features['structural_fingerprint'],
            features['chemical_fingerprint']
        ])
        features['combined_fingerprint'] = combined_fingerprint
        
        return features
    
    def extract_site_features(self, structure: Structure, site_idx: int) -> np.ndarray:
        """
        提取特定位点的特征
        Extract features for a specific site
        """
        site = structure[site_idx]
        
        # 局部结构特征
        local_features = []
        
        # 配位数
        neighbors = structure.get_neighbors(site, self.structural_desc.cutoff_radius)
        coordination_number = len(neighbors)
        local_features.append(coordination_number)
        
        # 平均键长
        if neighbors:
            avg_bond_length = np.mean([n.nn_distance for n in neighbors])
            bond_length_std = np.std([n.nn_distance for n in neighbors])
        else:
            avg_bond_length = 0
            bond_length_std = 0
        local_features.extend([avg_bond_length, bond_length_std])
        
        # 局部化学环境
        element = str(site.specie)
        if element in self.chemical_desc.element_properties['electronegativity']:
            electronegativity = self.chemical_desc.element_properties['electronegativity'][element]
            atomic_radius = self.chemical_desc.element_properties['atomic_radius'][element]
            ionization_energy = self.chemical_desc.element_properties['ionization_energy'][element]
        else:
            electronegativity = 0
            atomic_radius = 0
            ionization_energy = 0
        
        local_features.extend([electronegativity, atomic_radius, ionization_energy])
        
        # 邻居元素统计
        if neighbors:
            neighbor_elements = [str(n.site.specie) for n in neighbors]
            unique_neighbors = len(set(neighbor_elements))
            local_features.append(unique_neighbors)
            
            # 邻居电负性统计
            neighbor_en = []
            for elem in neighbor_elements:
                if elem in self.chemical_desc.element_properties['electronegativity']:
                    neighbor_en.append(self.chemical_desc.element_properties['electronegativity'][elem])
            
            if neighbor_en:
                local_features.extend([np.mean(neighbor_en), np.std(neighbor_en)])
            else:
                local_features.extend([0, 0])
        else:
            local_features.extend([0, 0, 0])
        
        return np.array(local_features)


# 使用示例
def example_usage():
    """使用示例"""
    # 创建特征提取器
    extractor = AdvancedFeatureExtractor()
    
    # 假设有一个Structure对象
    # structure = Structure.from_file("example.cif")
    
    # 提取特征
    # features = extractor.extract_features(structure)
    # print(f"结构指纹维度: {features['structural_fingerprint'].shape}")
    # print(f"化学指纹维度: {features['chemical_fingerprint'].shape}")
    # print(f"组合指纹维度: {features['combined_fingerprint'].shape}")
    
    pass 