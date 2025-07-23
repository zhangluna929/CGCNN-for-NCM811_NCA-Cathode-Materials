"""
Materials Project API Integration

Interface for accessing and comparing with Materials Project database.
Provides functions for data retrieval, validation, and benchmark creation
for CGCNN model development.

Author: lunazhang
Date: 2023
"""

import requests
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import pandas as pd

try:
    from mp_api.client import MPRester
    from pymatgen.core.structure import Structure
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    MP_API_AVAILABLE = True
except ImportError:
    warnings.warn("Materials Project API not available. Install with: pip install mp-api")
    MP_API_AVAILABLE = False
    MPRester = None


class MaterialsProjectAPI:
    """
    Materials Project API接口类
    Materials Project API Interface Class
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化Materials Project API接口
        
        Args:
            api_key: Materials Project API密钥
        """
        if not MP_API_AVAILABLE:
            raise ImportError("Materials Project API not available. Install with: pip install mp-api")
        
        self.api_key = api_key
        if api_key:
            self.mpr = MPRester(api_key)
        else:
            warnings.warn("No API key provided. Some functions may be limited.")
            self.mpr = None
    
    def search_structures(self, 
                         elements: Optional[List[str]] = None,
                         formula: Optional[str] = None,
                         energy_above_hull: float = 0.1,
                         is_stable: bool = True,
                         max_results: int = 100) -> List[Dict]:
        """
        搜索Materials Project中的结构
        Search structures in Materials Project
        
        Args:
            elements: 元素列表
            formula: 化学式
            energy_above_hull: 距离凸包的最大能量差
            is_stable: 是否只返回稳定相
            max_results: 最大结果数量
        
        Returns:
            structures_data: 结构数据列表
        """
        if self.mpr is None:
            raise ValueError("API key required for structure search")
        
        # 构建查询条件
        criteria = {}
        if elements:
            criteria['elements'] = elements
        if formula:
            criteria['formula'] = formula
        if is_stable:
            criteria['energy_above_hull'] = (0, energy_above_hull)
        
        try:
            # 查询结构
            results = self.mpr.materials.summary.search(
                **criteria,
                fields=["material_id", "formula_pretty", "structure", 
                       "energy_above_hull", "formation_energy_per_atom",
                       "band_gap", "density", "total_magnetization"]
            )
            
            # 限制结果数量
            if len(results) > max_results:
                results = results[:max_results]
            
            structures_data = []
            for result in results:
                structure_data = {
                    'material_id': result.material_id,
                    'formula': result.formula_pretty,
                    'structure': result.structure,
                    'energy_above_hull': result.energy_above_hull,
                    'formation_energy_per_atom': result.formation_energy_per_atom,
                    'band_gap': result.band_gap,
                    'density': result.density,
                    'total_magnetization': result.total_magnetization
                }
                structures_data.append(structure_data)
            
            return structures_data
            
        except Exception as e:
            warnings.warn(f"Failed to search structures: {e}")
            return []
    
    def get_structure_by_id(self, material_id: str) -> Optional[Structure]:
        """
        通过Material ID获取结构
        Get structure by Material ID
        
        Args:
            material_id: Materials Project材料ID
        
        Returns:
            structure: PyMatGen结构对象
        """
        if self.mpr is None:
            raise ValueError("API key required for structure retrieval")
        
        try:
            material = self.mpr.materials.summary.get_data_by_id(material_id)
            if material:
                return material.structure
            else:
                warnings.warn(f"No structure found for ID: {material_id}")
                return None
        except Exception as e:
            warnings.warn(f"Failed to retrieve structure {material_id}: {e}")
            return None
    
    def get_phase_diagram(self, elements: List[str]) -> Optional[PhaseDiagram]:
        """
        获取相图
        Get phase diagram
        
        Args:
            elements: 元素列表
        
        Returns:
            phase_diagram: 相图对象
        """
        if self.mpr is None:
            raise ValueError("API key required for phase diagram")
        
        try:
            # 获取相图条目
            entries = self.mpr.get_entries_in_chemsys(elements)
            
            if entries:
                phase_diagram = PhaseDiagram(entries)
                return phase_diagram
            else:
                warnings.warn(f"No entries found for system: {'-'.join(elements)}")
                return None
                
        except Exception as e:
            warnings.warn(f"Failed to get phase diagram: {e}")
            return None
    
    def compare_with_mp(self, structure: Structure, 
                       cgcnn_predictions: Dict[str, float],
                       tolerance: float = 0.1) -> Dict[str, Any]:
        """
        与Materials Project数据对比
        Compare with Materials Project data
        
        Args:
            structure: 待比较的结构
            cgcnn_predictions: CGCNN预测结果
            tolerance: 结构匹配容差
        
        Returns:
            comparison_results: 对比结果
        """
        if self.mpr is None:
            raise ValueError("API key required for comparison")
        
        try:
            # 查找相似结构
            similar_structures = self.search_structures(
                elements=list(set([str(site.specie) for site in structure])),
                max_results=50
            )
            
            best_match = None
            best_score = float('inf')
            
            for mp_data in similar_structures:
                mp_structure = mp_data['structure']
                
                # 结构相似性评估（简化版本）
                if len(structure) == len(mp_structure):
                    # 使用密度和体积作为快速筛选
                    density_diff = abs(structure.density - mp_data['density'])
                    volume_diff = abs(structure.volume/len(structure) - 
                                    mp_structure.volume/len(mp_structure))
                    
                    score = density_diff + volume_diff
                    
                    if score < best_score and score < tolerance:
                        best_score = score
                        best_match = mp_data
            
            if best_match:
                # 对比预测结果
                comparison = {
                    'mp_material_id': best_match['material_id'],
                    'mp_formula': best_match['formula'],
                    'structure_similarity_score': best_score,
                    'property_comparison': {}
                }
                
                # 对比各种性质
                if 'formation_energy_per_atom' in cgcnn_predictions:
                    mp_formation_energy = best_match['formation_energy_per_atom']
                    cgcnn_formation_energy = cgcnn_predictions['formation_energy_per_atom']
                    
                    comparison['property_comparison']['formation_energy_per_atom'] = {
                        'cgcnn_prediction': cgcnn_formation_energy,
                        'mp_value': mp_formation_energy,
                        'difference': abs(cgcnn_formation_energy - mp_formation_energy),
                        'relative_error': abs(cgcnn_formation_energy - mp_formation_energy) / abs(mp_formation_energy) if mp_formation_energy != 0 else float('inf')
                    }
                
                if 'band_gap' in cgcnn_predictions:
                    mp_band_gap = best_match['band_gap']
                    cgcnn_band_gap = cgcnn_predictions['band_gap']
                    
                    comparison['property_comparison']['band_gap'] = {
                        'cgcnn_prediction': cgcnn_band_gap,
                        'mp_value': mp_band_gap,
                        'difference': abs(cgcnn_band_gap - mp_band_gap),
                        'relative_error': abs(cgcnn_band_gap - mp_band_gap) / abs(mp_band_gap) if mp_band_gap != 0 else float('inf')
                    }
                
                return comparison
            else:
                return {'message': 'No similar structure found in Materials Project'}
                
        except Exception as e:
            warnings.warn(f"Failed to compare with MP: {e}")
            return {'error': str(e)}
    
    def download_training_data(self, 
                              elements: List[str],
                              properties: List[str] = ['formation_energy_per_atom', 'band_gap'],
                              max_structures: int = 1000,
                              save_path: Optional[str] = None) -> pd.DataFrame:
        """
        下载训练数据
        Download training data
        
        Args:
            elements: 感兴趣的元素
            properties: 要下载的性质
            max_structures: 最大结构数量
            save_path: 保存路径
        
        Returns:
            training_data: 训练数据DataFrame
        """
        if self.mpr is None:
            raise ValueError("API key required for data download")
        
        try:
            # 构建查询字段
            fields = ["material_id", "formula_pretty", "structure"] + properties
            
            # 搜索结构
            results = self.mpr.materials.summary.search(
                elements=elements,
                energy_above_hull=(0, 0.1),  # 只要相对稳定的相
                fields=fields
            )
            
            if len(results) > max_structures:
                results = results[:max_structures]
            
            # 构建DataFrame
            data_records = []
            for result in results:
                record = {
                    'material_id': result.material_id,
                    'formula': result.formula_pretty,
                    'structure': result.structure
                }
                
                # 添加性质数据
                for prop in properties:
                    if hasattr(result, prop):
                        record[prop] = getattr(result, prop)
                    else:
                        record[prop] = None
                
                data_records.append(record)
            
            training_data = pd.DataFrame(data_records)
            
            # 保存数据
            if save_path:
                training_data.to_pickle(save_path)
                print(f"Training data saved to: {save_path}")
            
            return training_data
            
        except Exception as e:
            warnings.warn(f"Failed to download training data: {e}")
            return pd.DataFrame()
    
    def validate_cgcnn_model(self, 
                            cgcnn_predict_func: callable,
                            test_material_ids: List[str],
                            property_name: str = 'formation_energy_per_atom') -> Dict[str, Any]:
        """
        验证CGCNN模型性能
        Validate CGCNN model performance
        
        Args:
            cgcnn_predict_func: CGCNN预测函数
            test_material_ids: 测试材料ID列表
            property_name: 要验证的性质名称
        
        Returns:
            validation_results: 验证结果
        """
        if self.mpr is None:
            raise ValueError("API key required for model validation")
        
        predictions = []
        mp_values = []
        material_ids = []
        
        for material_id in test_material_ids:
            try:
                # 获取MP数据
                material = self.mpr.materials.summary.get_data_by_id(material_id)
                if not material:
                    continue
                
                structure = material.structure
                mp_value = getattr(material, property_name)
                
                if mp_value is None:
                    continue
                
                # CGCNN预测
                cgcnn_prediction = cgcnn_predict_func(structure)
                
                predictions.append(cgcnn_prediction)
                mp_values.append(mp_value)
                material_ids.append(material_id)
                
            except Exception as e:
                warnings.warn(f"Failed to process {material_id}: {e}")
                continue
        
        if not predictions:
            return {'error': 'No valid predictions obtained'}
        

        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        predictions = np.array(predictions)
        mp_values = np.array(mp_values)
        
        mae = mean_absolute_error(mp_values, predictions)
        mse = mean_squared_error(mp_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(mp_values, predictions)
        
        validation_results = {
            'n_samples': len(predictions),
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'predictions': predictions.tolist(),
            'mp_values': mp_values.tolist(),
            'material_ids': material_ids
        }
        
        return validation_results
    
    def create_benchmark_dataset(self, 
                                system_types: List[str] = ['binary', 'ternary'],
                                min_structures_per_system: int = 10,
                                save_dir: str = 'mp_benchmark') -> Dict[str, str]:
        """
        创建基准测试数据集
        Create benchmark dataset
        
        Args:
            system_types: 系统类型
            min_structures_per_system: 每个系统最少结构数
            save_dir: 保存目录
        
        Returns:
            dataset_info: 数据集信息
        """
        if self.mpr is None:
            raise ValueError("API key required for benchmark creation")
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        dataset_info = {}
        
        for system_type in system_types:
            print(f"Creating {system_type} benchmark...")
            
            # 根据系统类型搜索
            if system_type == 'binary':
                # 搜索二元化合物
                common_elements = ['Li', 'Na', 'K', 'Mg', 'Ca', 'Al', 'Si', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'O', 'S', 'F', 'Cl']
                element_pairs = [(common_elements[i], common_elements[j]) 
                               for i in range(len(common_elements)) 
                               for j in range(i+1, len(common_elements))]
            else:  # ternary
                # 选择一些常见的三元系统
                element_triples = [['Li', 'Co', 'O'], ['Li', 'Ni', 'O'], ['Li', 'Mn', 'O'], 
                                 ['Li', 'Fe', 'P'], ['Na', 'Fe', 'O'], ['Mg', 'Al', 'O']]
            
            system_data = []
            
            if system_type == 'binary':
                for elements in element_pairs[:20]:  # 限制数量
                    structures = self.search_structures(
                        elements=list(elements),
                        max_results=min_structures_per_system
                    )
                    if len(structures) >= min_structures_per_system:
                        system_data.extend(structures)
            else:
                for elements in element_triples:
                    structures = self.search_structures(
                        elements=elements,
                        max_results=min_structures_per_system
                    )
                    if len(structures) >= min_structures_per_system:
                        system_data.extend(structures)
            
            # 保存数据
            if system_data:
                dataset_file = save_path / f'{system_type}_benchmark.pkl'
                pd.DataFrame(system_data).to_pickle(dataset_file)
                dataset_info[system_type] = str(dataset_file)
                print(f"Saved {len(system_data)} {system_type} structures")
        
        return dataset_info
    
    def export_structures_for_cgcnn(self, material_ids: List[str], 
                                   output_dir: str = 'cgcnn_structures') -> None:
        """
        导出结构用于CGCNN训练
        Export structures for CGCNN training
        
        Args:
            material_ids: 材料ID列表
            output_dir: 输出目录
        """
        if self.mpr is None:
            raise ValueError("API key required for structure export")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 创建id_prop.csv文件
        prop_data = []
        
        for i, material_id in enumerate(material_ids):
            try:
                material = self.mpr.materials.summary.get_data_by_id(material_id)
                if not material:
                    continue
                
                # 保存CIF文件
                cif_filename = f"{material_id}.cif"
                cif_path = output_path / cif_filename
                material.structure.to(fmt="cif", filename=str(cif_path))
                
                # 收集性质数据
                prop_data.append({
                    'cif_id': material_id,
                    'formation_energy_per_atom': material.formation_energy_per_atom,
                    'band_gap': material.band_gap
                })
                
                print(f"Exported {i+1}/{len(material_ids)}: {material_id}")
                
            except Exception as e:
                warnings.warn(f"Failed to export {material_id}: {e}")
        
        # 保存性质数据
        prop_df = pd.DataFrame(prop_data)
        prop_df.to_csv(output_path / 'id_prop.csv', index=False)
        
        print(f"Exported {len(prop_data)} structures to {output_dir}")


# 使用示例
def example_usage():
    if not MP_API_AVAILABLE:
        print("Materials Project API not available for example")
        return
    
    # 创建API接口（需要API密钥）
    # mp_api = MaterialsProjectAPI(api_key="your_api_key_here")
    
    # 搜索锂电池相关材料
    # li_materials = mp_api.search_structures(
    #     elements=['Li', 'Co', 'O'],
    #     max_results=10
    # )
    
    # 下载训练数据
    # training_data = mp_api.download_training_data(
    #     elements=['Li', 'Co', 'Ni', 'Mn', 'O'],
    #     save_path='mp_training_data.pkl'
    # )
    
    pass 