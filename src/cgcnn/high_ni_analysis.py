"""
High-Ni Cathode Materials Analysis Module

Specialized analysis tools for quantifying structural instabilities,
ionic conductivity degradation, and thermal expansion in high-Ni
layered cathode materials (NCM811/NCA).

Author: lunazhang
Date: 2023
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import warnings

try:
    from pymatgen.core.structure import Structure
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer
    from pymatgen.analysis.thermodynamics import ThermoData
    PYMATGEN_AVAILABLE = True
except ImportError:
    warnings.warn("PyMatGen not available. Some analysis features may be limited.")
    PYMATGEN_AVAILABLE = False
    Structure = None


class StructuralInstabilityAnalyzer:
    """
    结构不稳定性分析器
    Structural Instability Analyzer for High-Ni Cathodes
    """
    
    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance
        self.reference_bond_lengths = {
            'Li-O': 2.1,  # 标准Li-O键长 (Å)
            'Ni-O': 1.9,  # 标准Ni-O键长 (Å)
            'Co-O': 1.9,  # 标准Co-O键长 (Å)
            'Mn-O': 1.95, # 标准Mn-O键长 (Å)
            'Al-O': 1.8   # 标准Al-O键长 (Å)
        }
    
    def analyze_li_o_bond_distortion(self, structure: Structure) -> Dict[str, float]:
        """
        分析Li-O键长畸变
        Analyze Li-O bond length distortion
        
        Args:
            structure: 晶体结构
            
        Returns:
            bond_analysis: 键长分析结果
        """
        if not PYMATGEN_AVAILABLE:
            return {'error': 'PyMatGen not available'}
        
        li_sites = [i for i, site in enumerate(structure) if site.specie.symbol == 'Li']
        o_sites = [i for i, site in enumerate(structure) if site.specie.symbol == 'O']
        
        if not li_sites or not o_sites:
            return {'li_o_bonds': 0, 'avg_distortion': 0.0, 'max_distortion': 0.0}
        
        bond_lengths = []
        distortions = []
        
        for li_idx in li_sites:
            li_site = structure[li_idx]
            # 找到最近的氧原子
            distances = []
            for o_idx in o_sites:
                o_site = structure[o_idx]
                distance = li_site.distance(o_site)
                if distance < 3.0:  # 合理的Li-O键长范围
                    distances.append(distance)
                    bond_lengths.append(distance)
                    
                    # 计算相对于标准键长的畸变
                    distortion = abs(distance - self.reference_bond_lengths['Li-O']) / self.reference_bond_lengths['Li-O']
                    distortions.append(distortion)
        
        if not bond_lengths:
            return {'li_o_bonds': 0, 'avg_distortion': 0.0, 'max_distortion': 0.0}
        
        return {
            'li_o_bonds': len(bond_lengths),
            'avg_bond_length': np.mean(bond_lengths),
            'std_bond_length': np.std(bond_lengths),
            'avg_distortion': np.mean(distortions),
            'max_distortion': np.max(distortions),
            'distortion_variance': np.var(distortions)
        }
    
    def analyze_transition_metal_migration(self, structure: Structure) -> Dict[str, float]:
        """
        分析过渡金属离子迁移
        Analyze transition metal ion migration
        
        Args:
            structure: 晶体结构
            
        Returns:
            migration_analysis: 迁移分析结果
        """
        if not PYMATGEN_AVAILABLE:
            return {'error': 'PyMatGen not available'}
        
        tm_elements = ['Ni', 'Co', 'Mn', 'Al']
        li_sites = [i for i, site in enumerate(structure) if site.specie.symbol == 'Li']
        tm_sites = [i for i, site in enumerate(structure) 
                   if site.specie.symbol in tm_elements]
        
        if not li_sites or not tm_sites:
            return {'tm_in_li_layer': 0, 'migration_probability': 0.0}
        
        # 分析过渡金属是否出现在锂层中
        tm_in_li_layer = 0
        min_tm_li_distance = float('inf')
        
        for tm_idx in tm_sites:
            tm_site = structure[tm_idx]
            for li_idx in li_sites:
                li_site = structure[li_idx]
                distance = tm_site.distance(li_site)
                min_tm_li_distance = min(min_tm_li_distance, distance)
                
                # 如果过渡金属离子与锂离子距离过近，可能发生了迁移
                if distance < 2.5:  # 临界距离
                    tm_in_li_layer += 1
        
        # 计算迁移概率指标
        migration_probability = tm_in_li_layer / len(tm_sites) if tm_sites else 0.0
        
        return {
            'tm_in_li_layer': tm_in_li_layer,
            'total_tm_sites': len(tm_sites),
            'migration_probability': migration_probability,
            'min_tm_li_distance': min_tm_li_distance if min_tm_li_distance != float('inf') else 0.0
        }
    
    def analyze_oxygen_framework_stability(self, structure: Structure) -> Dict[str, float]:
        """
        分析氧骨架稳定性
        Analyze oxygen framework stability
        
        Args:
            structure: 晶体结构
            
        Returns:
            framework_analysis: 氧骨架分析结果
        """
        if not PYMATGEN_AVAILABLE:
            return {'error': 'PyMatGen not available'}
        
        o_sites = [i for i, site in enumerate(structure) if site.specie.symbol == 'O']
        
        if len(o_sites) < 3:
            return {'o_o_bonds': 0, 'framework_distortion': 0.0}
        
        # 分析O-O距离分布
        o_o_distances = []
        for i, o1_idx in enumerate(o_sites):
            o1_site = structure[o1_idx]
            for o2_idx in o_sites[i+1:]:
                o2_site = structure[o2_idx]
                distance = o1_site.distance(o2_site)
                if 2.5 < distance < 4.0:  # 合理的O-O距离范围
                    o_o_distances.append(distance)
        
        if not o_o_distances:
            return {'o_o_bonds': 0, 'framework_distortion': 0.0}
        
        # 理想O-O距离（基于理想层状结构）
        ideal_o_o_distance = 2.9  # Å
        
        distortions = [abs(d - ideal_o_o_distance) / ideal_o_o_distance for d in o_o_distances]
        
        return {
            'o_o_bonds': len(o_o_distances),
            'avg_o_o_distance': np.mean(o_o_distances),
            'std_o_o_distance': np.std(o_o_distances),
            'framework_distortion': np.mean(distortions),
            'max_framework_distortion': np.max(distortions)
        }
    
    def calculate_layering_parameter(self, structure: Structure) -> float:
        """
        计算层状结构参数
        Calculate layering parameter for structural characterization
        
        Args:
            structure: 晶体结构
            
        Returns:
            layering_parameter: 层状结构参数 (0-1, 1为完美层状)
        """
        if not PYMATGEN_AVAILABLE:
            return 0.0
        
        # 获取晶格参数
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        
        # 层状结构的特征：c/a比值和角度
        c_over_a = c / a
        ideal_c_over_a = 4.9  # NCM811的理想c/a比值
        
        # 角度偏离六方对称的程度
        angle_deviation = abs(gamma - 120.0) / 120.0
        
        # 层状参数计算
        c_a_score = 1.0 - abs(c_over_a - ideal_c_over_a) / ideal_c_over_a
        angle_score = 1.0 - angle_deviation
        
        layering_parameter = (c_a_score + angle_score) / 2.0
        
        return max(0.0, min(1.0, layering_parameter))


class IonicConductivityAnalyzer:
    """
    离子导电性分析器
    Ionic Conductivity Analyzer
    """
    
    def __init__(self):
        self.li_diffusion_pathways = []
    
    def analyze_li_diffusion_barriers(self, structure: Structure, 
                                    vacancy_structure: Structure) -> Dict[str, float]:
        """
        分析锂离子扩散势垒
        Analyze Li-ion diffusion barriers
        
        Args:
            structure: 原始结构
            vacancy_structure: 含空位结构
            
        Returns:
            diffusion_analysis: 扩散分析结果
        """
        if not PYMATGEN_AVAILABLE:
            return {'error': 'PyMatGen not available'}
        
        # 识别锂离子扩散路径
        li_sites_orig = [i for i, site in enumerate(structure) if site.specie.symbol == 'Li']
        li_sites_vac = [i for i, site in enumerate(vacancy_structure) if site.specie.symbol == 'Li']
        
        # 计算平均Li-Li距离（代表扩散路径长度）
        li_li_distances = []
        for i, li1_idx in enumerate(li_sites_orig):
            li1_site = structure[li1_idx]
            for li2_idx in li_sites_orig[i+1:]:
                li2_site = structure[li2_idx]
                distance = li1_site.distance(li2_site)
                if 2.0 < distance < 5.0:  # 合理的Li-Li扩散距离
                    li_li_distances.append(distance)
        
        # 计算扩散路径的拥挤程度
        vacancy_density = (len(li_sites_orig) - len(li_sites_vac)) / len(li_sites_orig)
        
        # 估算扩散势垒指标
        if li_li_distances:
            avg_diffusion_distance = np.mean(li_li_distances)
            min_diffusion_distance = np.min(li_li_distances)
            
            # 经验公式：距离越短，势垒越高
            diffusion_barrier_index = 1.0 / min_diffusion_distance if min_diffusion_distance > 0 else float('inf')
        else:
            avg_diffusion_distance = 0.0
            min_diffusion_distance = 0.0
            diffusion_barrier_index = float('inf')
        
        return {
            'li_sites_original': len(li_sites_orig),
            'li_sites_vacancy': len(li_sites_vac),
            'vacancy_density': vacancy_density,
            'avg_diffusion_distance': avg_diffusion_distance,
            'min_diffusion_distance': min_diffusion_distance,
            'diffusion_barrier_index': diffusion_barrier_index,
            'pathway_accessibility': 1.0 / (1.0 + diffusion_barrier_index) if diffusion_barrier_index != float('inf') else 0.0
        }
    
    def calculate_percolation_threshold(self, structure: Structure) -> float:
        """
        计算渗流阈值
        Calculate percolation threshold for Li-ion transport
        
        Args:
            structure: 晶体结构
            
        Returns:
            percolation_threshold: 渗流阈值
        """
        if not PYMATGEN_AVAILABLE:
            return 0.0
        
        li_sites = [i for i, site in enumerate(structure) if site.specie.symbol == 'Li']
        total_li_sites = len([site for site in structure if site.specie.symbol == 'Li'])
        
        if total_li_sites == 0:
            return 1.0  # 完全阻塞
        
        # 简化的渗流分析：基于Li密度和连通性
        li_density = len(li_sites) / len(structure)
        
        # 理论渗流阈值（经验值）
        theoretical_threshold = 0.16  # 3D情况下的渗流阈值
        
        # 实际阈值考虑结构因素
        actual_threshold = theoretical_threshold / li_density if li_density > 0 else 1.0
        
        return min(1.0, actual_threshold)
    
    def analyze_bottleneck_sites(self, structure: Structure) -> Dict[str, Any]:
        """
        分析扩散瓶颈位点
        Analyze diffusion bottleneck sites
        
        Args:
            structure: 晶体结构
            
        Returns:
            bottleneck_analysis: 瓶颈分析结果
        """
        if not PYMATGEN_AVAILABLE:
            return {'error': 'PyMatGen not available'}
        
        li_sites = [i for i, site in enumerate(structure) if site.specie.symbol == 'Li']
        tm_sites = [i for i, site in enumerate(structure) 
                   if site.specie.symbol in ['Ni', 'Co', 'Mn', 'Al']]
        
        bottleneck_sites = []
        for li_idx in li_sites:
            li_site = structure[li_idx]
            
            # 计算与过渡金属的最近距离
            min_tm_distance = float('inf')
            for tm_idx in tm_sites:
                tm_site = structure[tm_idx]
                distance = li_site.distance(tm_site)
                min_tm_distance = min(min_tm_distance, distance)
            
            # 如果Li离子被过渡金属包围得太紧密，形成瓶颈
            if min_tm_distance < 2.8:  # 临界距离
                bottleneck_sites.append({
                    'site_index': li_idx,
                    'min_tm_distance': min_tm_distance,
                    'coordination': len([tm_idx for tm_idx in tm_sites 
                                       if structure[li_idx].distance(structure[tm_idx]) < 3.5])
                })
        
        if bottleneck_sites:
            avg_bottleneck_distance = np.mean([site['min_tm_distance'] for site in bottleneck_sites])
            max_coordination = max([site['coordination'] for site in bottleneck_sites])
        else:
            avg_bottleneck_distance = 0.0
            max_coordination = 0
        
        return {
            'total_bottleneck_sites': len(bottleneck_sites),
            'bottleneck_fraction': len(bottleneck_sites) / len(li_sites) if li_sites else 0.0,
            'avg_bottleneck_distance': avg_bottleneck_distance,
            'max_coordination': max_coordination,
            'bottleneck_details': bottleneck_sites
        }


class ThermalExpansionAnalyzer:
    """
    热膨胀系数分析器
    Thermal Expansion Coefficient Analyzer
    """
    
    def __init__(self):
        self.reference_temp = 298.15  # K
    
    def calculate_anisotropic_expansion(self, structure_low_t: Structure,
                                      structure_high_t: Structure,
                                      temp_low: float, temp_high: float) -> Dict[str, float]:
        """
        计算各向异性热膨胀系数
        Calculate anisotropic thermal expansion coefficients
        
        Args:
            structure_low_t: 低温结构
            structure_high_t: 高温结构
            temp_low: 低温 (K)
            temp_high: 高温 (K)
            
        Returns:
            expansion_coefficients: 热膨胀系数
        """
        if not PYMATGEN_AVAILABLE:
            return {'error': 'PyMatGen not available'}
        
        # 获取晶格参数
        a_low, b_low, c_low = structure_low_t.lattice.abc
        a_high, b_high, c_high = structure_high_t.lattice.abc
        
        vol_low = structure_low_t.lattice.volume
        vol_high = structure_high_t.lattice.volume
        
        delta_t = temp_high - temp_low
        
        if delta_t == 0:
            return {'alpha_a': 0.0, 'alpha_b': 0.0, 'alpha_c': 0.0, 'alpha_vol': 0.0}
        
        # 线膨胀系数 α = (1/L) * (dL/dT)
        alpha_a = (a_high - a_low) / (a_low * delta_t)
        alpha_b = (b_high - b_low) / (b_low * delta_t)
        alpha_c = (c_high - c_low) / (c_low * delta_t)
        
        # 体膨胀系数
        alpha_vol = (vol_high - vol_low) / (vol_low * delta_t)
        
        # 各向异性指标
        anisotropy_ab = abs(alpha_a - alpha_b) / max(abs(alpha_a), abs(alpha_b), 1e-10)
        anisotropy_ac = abs(alpha_a - alpha_c) / max(abs(alpha_a), abs(alpha_c), 1e-10)
        anisotropy_bc = abs(alpha_b - alpha_c) / max(abs(alpha_b), abs(alpha_c), 1e-10)
        
        return {
            'alpha_a': alpha_a,
            'alpha_b': alpha_b,
            'alpha_c': alpha_c,
            'alpha_vol': alpha_vol,
            'anisotropy_ab': anisotropy_ab,
            'anisotropy_ac': anisotropy_ac,
            'anisotropy_bc': anisotropy_bc,
            'max_anisotropy': max(anisotropy_ab, anisotropy_ac, anisotropy_bc)
        }
    
    def predict_thermal_stress(self, expansion_coefficients: Dict[str, float],
                             elastic_moduli: Dict[str, float]) -> Dict[str, float]:
        """
        预测热应力
        Predict thermal stress based on expansion and elastic properties
        
        Args:
            expansion_coefficients: 热膨胀系数
            elastic_moduli: 弹性模量
            
        Returns:
            thermal_stress: 热应力分析
        """
        # 简化的热应力计算：σ = E * α * ΔT
        delta_t = 100.0  # 假设温度变化100K
        
        bulk_modulus = elastic_moduli.get('bulk_modulus', 150.0)  # GPa
        shear_modulus = elastic_moduli.get('shear_modulus', 60.0)  # GPa
        
        alpha_a = expansion_coefficients.get('alpha_a', 0.0)
        alpha_c = expansion_coefficients.get('alpha_c', 0.0)
        alpha_vol = expansion_coefficients.get('alpha_vol', 0.0)
        
        # 各方向的热应力
        stress_a = bulk_modulus * alpha_a * delta_t
        stress_c = bulk_modulus * alpha_c * delta_t
        
        # 剪切应力（由于各向异性膨胀）
        anisotropy = abs(alpha_a - alpha_c)
        shear_stress = shear_modulus * anisotropy * delta_t
        
        # 体积应力
        volume_stress = bulk_modulus * alpha_vol * delta_t
        
        # 热应力风险指标
        max_stress = max(abs(stress_a), abs(stress_c), abs(shear_stress))
        critical_stress = 100.0  # MPa，临界应力
        
        stress_risk = max_stress / critical_stress
        
        return {
            'stress_a': stress_a,
            'stress_c': stress_c,
            'shear_stress': shear_stress,
            'volume_stress': volume_stress,
            'max_stress': max_stress,
            'stress_risk': stress_risk,
            'risk_level': 'high' if stress_risk > 1.0 else 'medium' if stress_risk > 0.5 else 'low'
        }


class ComprehensiveHighNiAnalyzer:
    """
    高镍材料综合分析器
    Comprehensive High-Ni Materials Analyzer
    """
    
    def __init__(self):
        self.structural_analyzer = StructuralInstabilityAnalyzer()
        self.conductivity_analyzer = IonicConductivityAnalyzer()
        self.thermal_analyzer = ThermalExpansionAnalyzer()
    
    def comprehensive_analysis(self, structure: Structure,
                             vacancy_structure: Optional[Structure] = None,
                             high_temp_structure: Optional[Structure] = None,
                             elastic_properties: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        综合分析高镍材料性能
        Comprehensive analysis of high-Ni cathode performance
        
        Args:
            structure: 基础晶体结构
            vacancy_structure: 含空位结构（可选）
            high_temp_structure: 高温结构（可选）
            elastic_properties: 弹性性质（可选）
            
        Returns:
            comprehensive_results: 综合分析结果
        """
        results = {}
        
        # 1. 结构稳定性分析
        results['structural_stability'] = {
            'li_o_bonds': self.structural_analyzer.analyze_li_o_bond_distortion(structure),
            'tm_migration': self.structural_analyzer.analyze_transition_metal_migration(structure),
            'oxygen_framework': self.structural_analyzer.analyze_oxygen_framework_stability(structure),
            'layering_parameter': self.structural_analyzer.calculate_layering_parameter(structure)
        }
        
        # 2. 离子导电性分析
        if vacancy_structure:
            results['ionic_conductivity'] = {
                'diffusion_barriers': self.conductivity_analyzer.analyze_li_diffusion_barriers(
                    structure, vacancy_structure),
                'percolation_threshold': self.conductivity_analyzer.calculate_percolation_threshold(structure),
                'bottleneck_sites': self.conductivity_analyzer.analyze_bottleneck_sites(structure)
            }
        
        # 3. 热膨胀分析
        if high_temp_structure:
            expansion_coeffs = self.thermal_analyzer.calculate_anisotropic_expansion(
                structure, high_temp_structure, 298.15, 398.15)
            results['thermal_expansion'] = expansion_coeffs
            
            if elastic_properties:
                thermal_stress = self.thermal_analyzer.predict_thermal_stress(
                    expansion_coeffs, elastic_properties)
                results['thermal_stress'] = thermal_stress
        
        # 4. 综合性能指标
        results['performance_indicators'] = self._calculate_performance_indicators(results)
        
        return results
    
    def _calculate_performance_indicators(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """
        计算综合性能指标
        Calculate comprehensive performance indicators
        """
        indicators = {}
        
        # 结构稳定性指标 (0-1, 1为最稳定)
        if 'structural_stability' in analysis_results:
            structural = analysis_results['structural_stability']
            
            # Li-O键稳定性
            li_o_stability = 1.0 - min(1.0, structural['li_o_bonds'].get('avg_distortion', 0.0))
            
            # 过渡金属迁移风险
            tm_stability = 1.0 - structural['tm_migration'].get('migration_probability', 0.0)
            
            # 层状结构完整性
            layering_quality = structural.get('layering_parameter', 0.0)
            
            # 综合结构稳定性
            indicators['structural_stability_index'] = (li_o_stability + tm_stability + layering_quality) / 3.0
        
        # 离子导电性指标 (0-1, 1为最佳导电性)
        if 'ionic_conductivity' in analysis_results:
            conductivity = analysis_results['ionic_conductivity']
            
            # 扩散路径可达性
            pathway_score = conductivity['diffusion_barriers'].get('pathway_accessibility', 0.0)
            
            # 瓶颈位点影响
            bottleneck_impact = 1.0 - conductivity['bottleneck_sites'].get('bottleneck_fraction', 0.0)
            
            # 综合导电性
            indicators['ionic_conductivity_index'] = (pathway_score + bottleneck_impact) / 2.0
        
        # 热稳定性指标 (0-1, 1为最佳热稳定性)
        if 'thermal_stress' in analysis_results:
            thermal = analysis_results['thermal_stress']
            stress_risk = thermal.get('stress_risk', 0.0)
            indicators['thermal_stability_index'] = 1.0 / (1.0 + stress_risk)
        
        # 总体性能评分
        if indicators:
            indicators['overall_performance_score'] = np.mean(list(indicators.values()))
        
        return indicators
    
    def material_comparison(self, ncm811_results: Dict[str, Any],
                          nca_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        材料对比分析
        Comparative analysis between NCM811 and NCA
        
        Args:
            ncm811_results: NCM811分析结果
            nca_results: NCA分析结果
            
        Returns:
            comparison: 对比分析结果
        """
        comparison = {
            'material_advantages': {},
            'performance_gaps': {},
            'improvement_recommendations': {}
        }
        
        # 性能指标对比
        ncm_indicators = ncm811_results.get('performance_indicators', {})
        nca_indicators = nca_results.get('performance_indicators', {})
        
        for indicator in ncm_indicators:
            if indicator in nca_indicators:
                ncm_score = ncm_indicators[indicator]
                nca_score = nca_indicators[indicator]
                
                if ncm_score > nca_score:
                    comparison['material_advantages'][f'NCM811_{indicator}'] = ncm_score - nca_score
                else:
                    comparison['material_advantages'][f'NCA_{indicator}'] = nca_score - ncm_score
                
                comparison['performance_gaps'][indicator] = abs(ncm_score - nca_score)
        
        # 改进建议
        if ncm_indicators.get('structural_stability_index', 0) < 0.7:
            comparison['improvement_recommendations']['NCM811'] = [
                '优化合成条件减少Ni在Li层的占位',
                '掺杂稳定元素提高结构稳定性',
                '表面包覆减少副反应'
            ]
        
        if nca_indicators.get('ionic_conductivity_index', 0) < 0.6:
            comparison['improvement_recommendations']['NCA'] = [
                '优化Li/Ni混排减少扩散障碍',
                '控制Al掺杂浓度平衡稳定性和导电性',
                '纳米化设计缩短扩散路径'
            ]
        
        return comparison


# 使用示例
def example_usage():
    """使用示例"""
    if not PYMATGEN_AVAILABLE:
        print("PyMatGen not available. Please install: pip install pymatgen")
        return
    
    # 创建分析器
    analyzer = ComprehensiveHighNiAnalyzer()
    
    # 加载结构文件
    # structure = Structure.from_file('NCM811.cif')
    # vacancy_structure = Structure.from_file('NCM811_LiVac0.cif')
    
    # 进行综合分析
    # results = analyzer.comprehensive_analysis(
    #     structure=structure,
    #     vacancy_structure=vacancy_structure,
    #     elastic_properties={'bulk_modulus': 160.0, 'shear_modulus': 65.0}
    # )
    
    # print("High-Ni Material Analysis Results:")
    # print(f"Structural Stability Index: {results['performance_indicators']['structural_stability_index']:.3f}")
    # print(f"Ionic Conductivity Index: {results['performance_indicators']['ionic_conductivity_index']:.3f}")
    # print(f"Overall Performance Score: {results['performance_indicators']['overall_performance_score']:.3f}")
    
    pass


if __name__ == "__main__":
    example_usage() 