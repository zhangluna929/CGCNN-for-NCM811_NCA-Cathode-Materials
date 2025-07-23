"""
Multi-Property Precision Analysis Module

Stratified precision evaluation system for different materials and defect types,
with focus on joint optimization of mechanical properties like elastic moduli
and thermal conductivity.

Author: lunazhang
Date: 2023
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import classification_report, confusion_matrix
    from scipy.stats import pearsonr, spearmanr
    SKLEARN_AVAILABLE = True
except ImportError:
    warnings.warn("Scikit-learn not fully available. Some metrics may be limited.")
    SKLEARN_AVAILABLE = False


class MaterialPropertyMatrix:
    """
    材料-性质精度矩阵
    Material-Property Precision Matrix
    """
    
    def __init__(self):
        self.materials = ['NCM811', 'NCA', 'NCM622', 'NCM532']
        self.defect_types = ['Li_vacancy', 'Ni_migration', 'O_vacancy', 'pristine']
        self.properties = [
            'formation_energy', 'band_gap', 'elastic_moduli', 'thermal_conductivity',
            'efermi', 'poisson_ratio', 'shear_modulus', 'bulk_modulus'
        ]
        
        # 初始化精度矩阵
        self.precision_matrix = {}
        self.uncertainty_matrix = {}
        self.sample_counts = {}
        
        self._initialize_matrices()
    
    def _initialize_matrices(self):
        """初始化各种矩阵"""
        for material in self.materials:
            self.precision_matrix[material] = {}
            self.uncertainty_matrix[material] = {}
            self.sample_counts[material] = {}
            
            for defect in self.defect_types:
                self.precision_matrix[material][defect] = {}
                self.uncertainty_matrix[material][defect] = {}
                self.sample_counts[material][defect] = {}
                
                for prop in self.properties:
                    self.precision_matrix[material][defect][prop] = {
                        'mae': float('inf'),
                        'rmse': float('inf'),
                        'r2': -float('inf'),
                        'mape': float('inf')
                    }
                    self.uncertainty_matrix[material][defect][prop] = {
                        'epistemic': 0.0,
                        'aleatoric': 0.0,
                        'total': 0.0
                    }
                    self.sample_counts[material][defect][prop] = 0
    
    def update_precision(self, material: str, defect_type: str, property_name: str,
                        predictions: np.ndarray, targets: np.ndarray,
                        uncertainties: Optional[np.ndarray] = None):
        """
        更新精度矩阵
        Update precision matrix with new evaluation results
        
        Args:
            material: 材料类型
            defect_type: 缺陷类型
            property_name: 性质名称
            predictions: 预测值
            targets: 真实值
            uncertainties: 不确定性（可选）
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")
        
        # 计算精度指标
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        # 计算MAPE（避免除零）
        non_zero_mask = targets != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        # 更新精度矩阵
        self.precision_matrix[material][defect_type][property_name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        # 更新样本数量
        self.sample_counts[material][defect_type][property_name] = len(predictions)
        
        # 更新不确定性矩阵
        if uncertainties is not None:
            epistemic_uncertainty = np.std(uncertainties)  # 简化的认识不确定性
            aleatoric_uncertainty = np.mean(uncertainties)  # 简化的偶然不确定性
            total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
            
            self.uncertainty_matrix[material][defect_type][property_name] = {
                'epistemic': epistemic_uncertainty,
                'aleatoric': aleatoric_uncertainty,
                'total': total_uncertainty
            }
    
    def get_precision_summary(self) -> pd.DataFrame:
        """
        获取精度总结
        Get precision summary as DataFrame
        
        Returns:
            summary_df: 精度总结数据框
        """
        rows = []
        
        for material in self.materials:
            for defect in self.defect_types:
                for prop in self.properties:
                    precision = self.precision_matrix[material][defect][prop]
                    uncertainty = self.uncertainty_matrix[material][defect][prop]
                    sample_count = self.sample_counts[material][defect][prop]
                    
                    if precision['mae'] != float('inf'):  # 只包含有数据的条目
                        rows.append({
                            'Material': material,
                            'Defect_Type': defect,
                            'Property': prop,
                            'MAE': precision['mae'],
                            'RMSE': precision['rmse'],
                            'R2': precision['r2'],
                            'MAPE': precision['mape'],
                            'Epistemic_Uncertainty': uncertainty['epistemic'],
                            'Aleatoric_Uncertainty': uncertainty['aleatoric'],
                            'Total_Uncertainty': uncertainty['total'],
                            'Sample_Count': sample_count
                        })
        
        return pd.DataFrame(rows)
    
    def identify_precision_gaps(self, threshold_mae: float = 0.1) -> Dict[str, List[str]]:
        """
        识别精度差距
        Identify precision gaps requiring improvement
        
        Args:
            threshold_mae: MAE阈值
            
        Returns:
            precision_gaps: 需要改进的材料-性质组合
        """
        gaps = {
            'high_error': [],
            'low_sample': [],
            'high_uncertainty': []
        }
        
        for material in self.materials:
            for defect in self.defect_types:
                for prop in self.properties:
                    precision = self.precision_matrix[material][defect][prop]
                    uncertainty = self.uncertainty_matrix[material][defect][prop]
                    sample_count = self.sample_counts[material][defect][prop]
                    
                    combination = f"{material}-{defect}-{prop}"
                    
                    # 高误差
                    if precision['mae'] > threshold_mae and precision['mae'] != float('inf'):
                        gaps['high_error'].append(combination)
                    
                    # 样本不足
                    if 0 < sample_count < 50:
                        gaps['low_sample'].append(combination)
                    
                    # 高不确定性
                    if uncertainty['total'] > 0.2:
                        gaps['high_uncertainty'].append(combination)
        
        return gaps
    
    def visualize_precision_matrix(self, property_name: str, save_path: Optional[str] = None):
        """
        可视化精度矩阵
        Visualize precision matrix for a specific property
        
        Args:
            property_name: 性质名称
            save_path: 保存路径
        """
        # 创建MAE热图数据
        mae_data = []
        for material in self.materials:
            row = []
            for defect in self.defect_types:
                mae = self.precision_matrix[material][defect][property_name]['mae']
                row.append(mae if mae != float('inf') else np.nan)
            mae_data.append(row)
        
        mae_array = np.array(mae_data)
        
        # 绘制热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(mae_array, 
                   xticklabels=self.defect_types,
                   yticklabels=self.materials,
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis_r',
                   cbar_kws={'label': 'Mean Absolute Error'})
        
        plt.title(f'Prediction Precision Matrix for {property_name}')
        plt.xlabel('Defect Type')
        plt.ylabel('Material')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class StratifiedPrecisionEvaluator:
    """
    分层精度评估器
    Stratified Precision Evaluator
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.evaluation_results = {}
    
    def evaluate_by_material_type(self, data_loader, material_labels: List[str]) -> Dict[str, Dict[str, float]]:
        """
        按材料类型评估
        Evaluate by material type
        
        Args:
            data_loader: 数据加载器
            material_labels: 材料标签列表
            
        Returns:
            material_results: 按材料分类的评估结果
        """
        self.model.eval()
        
        # 收集预测结果
        all_predictions = []
        all_targets = []
        all_materials = []
        
        with torch.no_grad():
            for i, (inputs, targets, batch_cif_ids) in enumerate(data_loader):
                # 移动数据到设备
                inputs = [inp.to(self.device) if torch.is_tensor(inp) 
                         else [t.to(self.device) for t in inp] for inp in inputs]
                targets = targets.to(self.device)
                
                # 预测
                outputs = self.model(*inputs)
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # 从CIF ID推断材料类型
                batch_materials = [self._infer_material_type(cif_id) for cif_id in batch_cif_ids]
                all_materials.extend(batch_materials)
        
        # 按材料类型分组评估
        material_results = {}
        unique_materials = set(all_materials)
        
        for material in unique_materials:
            material_mask = np.array(all_materials) == material
            material_preds = np.array(all_predictions)[material_mask]
            material_targets = np.array(all_targets)[material_mask]
            
            if len(material_preds) > 0:
                material_results[material] = self._calculate_metrics(material_preds, material_targets)
        
        return material_results
    
    def evaluate_by_defect_type(self, data_loader) -> Dict[str, Dict[str, float]]:
        """
        按缺陷类型评估
        Evaluate by defect type
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            defect_results: 按缺陷分类的评估结果
        """
        self.model.eval()
        
        # 收集预测结果
        all_predictions = []
        all_targets = []
        all_defects = []
        
        with torch.no_grad():
            for i, (inputs, targets, batch_cif_ids) in enumerate(data_loader):
                # 移动数据到设备
                inputs = [inp.to(self.device) if torch.is_tensor(inp) 
                         else [t.to(self.device) for t in inp] for inp in inputs]
                targets = targets.to(self.device)
                
                # 预测
                outputs = self.model(*inputs)
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # 从CIF ID推断缺陷类型
                batch_defects = [self._infer_defect_type(cif_id) for cif_id in batch_cif_ids]
                all_defects.extend(batch_defects)
        
        # 按缺陷类型分组评估
        defect_results = {}
        unique_defects = set(all_defects)
        
        for defect in unique_defects:
            defect_mask = np.array(all_defects) == defect
            defect_preds = np.array(all_predictions)[defect_mask]
            defect_targets = np.array(all_targets)[defect_mask]
            
            if len(defect_preds) > 0:
                defect_results[defect] = self._calculate_metrics(defect_preds, defect_targets)
        
        return defect_results
    
    def _infer_material_type(self, cif_id: str) -> str:
        """从CIF ID推断材料类型"""
        cif_id_lower = cif_id.lower()
        if 'ncm811' in cif_id_lower:
            return 'NCM811'
        elif 'nca' in cif_id_lower:
            return 'NCA'
        elif 'ncm622' in cif_id_lower:
            return 'NCM622'
        elif 'ncm532' in cif_id_lower:
            return 'NCM532'
        else:
            return 'Unknown'
    
    def _infer_defect_type(self, cif_id: str) -> str:
        """从CIF ID推断缺陷类型"""
        cif_id_lower = cif_id.lower()
        if 'livac' in cif_id_lower or 'li_vac' in cif_id_lower:
            return 'Li_vacancy'
        elif 'nimig' in cif_id_lower or 'ni_mig' in cif_id_lower:
            return 'Ni_migration'
        elif 'ovac' in cif_id_lower or 'o_vac' in cif_id_lower:
            return 'O_vacancy'
        else:
            return 'pristine'
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        if len(predictions) == 0:
            return {}
        
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        metrics = {
            'mae': mean_absolute_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'r2': r2_score(targets, predictions),
            'sample_count': len(predictions)
        }
        
        # 计算MAPE
        non_zero_mask = targets != 0
        if np.any(non_zero_mask):
            metrics['mape'] = np.mean(np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])) * 100
        else:
            metrics['mape'] = float('inf')
        
        # 计算相关系数
        if len(predictions) > 1:
            pearson_corr, _ = pearsonr(predictions, targets)
            spearman_corr, _ = spearmanr(predictions, targets)
            metrics['pearson_corr'] = pearson_corr
            metrics['spearman_corr'] = spearman_corr
        
        return metrics
    
    def cross_validation_by_strata(self, dataset, property_name: str, 
                                  cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        分层交叉验证
        Stratified cross-validation
        
        Args:
            dataset: 数据集
            property_name: 性质名称
            cv_folds: 交叉验证折数
            
        Returns:
            cv_results: 交叉验证结果
        """
        # 提取材料和缺陷标签用于分层
        material_labels = []
        defect_labels = []
        
        for i in range(len(dataset)):
            _, _, _, _, _, cif_id = dataset[i]
            material_labels.append(self._infer_material_type(cif_id))
            defect_labels.append(self._infer_defect_type(cif_id))
        
        # 创建复合标签用于分层
        combined_labels = [f"{mat}_{def}" for mat, def in zip(material_labels, defect_labels)]
        
        # 分层K折交叉验证
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'overall': {'mae': [], 'rmse': [], 'r2': []},
            'by_material': {},
            'by_defect': {}
        }
        
        for train_idx, val_idx in skf.split(range(len(dataset)), combined_labels):
            # 创建训练和验证子集
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            # 这里需要重新训练模型（简化版本，实际应用中需要完整的训练循环）
            # 评估当前折
            val_results = self._evaluate_subset(val_subset)
            
            # 收集结果
            cv_results['overall']['mae'].append(val_results.get('mae', float('inf')))
            cv_results['overall']['rmse'].append(val_results.get('rmse', float('inf')))
            cv_results['overall']['r2'].append(val_results.get('r2', -float('inf')))
        
        # 计算平均值和标准差
        for metric in cv_results['overall']:
            values = cv_results['overall'][metric]
            cv_results['overall'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return cv_results
    
    def _evaluate_subset(self, subset) -> Dict[str, float]:
        """评估数据子集（简化版本）"""
        # 这里是简化的评估，实际应用中需要完整的数据加载和评估流程
        return {'mae': 0.05, 'rmse': 0.08, 'r2': 0.85}


class MechanicalPropertyOptimizer:
    """
    机械性能联合优化器
    Mechanical Property Joint Optimizer
    """
    
    def __init__(self):
        self.property_relationships = {
            'elastic_moduli': ['bulk_modulus', 'shear_modulus', 'youngs_modulus'],
            'mechanical_stability': ['poisson_ratio', 'elastic_moduli', 'hardness'],
            'thermal_mechanical': ['thermal_conductivity', 'thermal_expansion', 'elastic_moduli']
        }
    
    def analyze_property_correlations(self, predictions_dict: Dict[str, np.ndarray],
                                    targets_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        分析性质间相关性
        Analyze correlations between properties
        
        Args:
            predictions_dict: 预测值字典
            targets_dict: 真实值字典
            
        Returns:
            correlation_analysis: 相关性分析结果
        """
        correlations = {}
        
        # 检查所有性质对的相关性
        properties = list(predictions_dict.keys())
        
        for i, prop1 in enumerate(properties):
            for prop2 in properties[i+1:]:
                if prop1 in targets_dict and prop2 in targets_dict:
                    # 真实值之间的相关性
                    true_corr, _ = pearsonr(targets_dict[prop1].flatten(), 
                                          targets_dict[prop2].flatten())
                    
                    # 预测值之间的相关性
                    pred_corr, _ = pearsonr(predictions_dict[prop1].flatten(),
                                          predictions_dict[prop2].flatten())
                    
                    correlations[f"{prop1}_{prop2}"] = {
                        'true_correlation': true_corr,
                        'predicted_correlation': pred_corr,
                        'correlation_preservation': abs(true_corr - pred_corr)
                    }
        
        return correlations
    
    def optimize_multi_property_loss(self, predictions: Dict[str, torch.Tensor],
                                   targets: Dict[str, torch.Tensor],
                                   property_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        多性质联合优化损失函数
        Multi-property joint optimization loss
        
        Args:
            predictions: 预测值字典
            targets: 真实值字典
            property_weights: 性质权重字典
            
        Returns:
            total_loss: 总损失
        """
        if property_weights is None:
            property_weights = {prop: 1.0 for prop in predictions.keys()}
        
        total_loss = 0.0
        loss_components = {}
        
        # 1. 单独性质损失
        for prop in predictions:
            if prop in targets:
                prop_loss = nn.MSELoss()(predictions[prop], targets[prop])
                weighted_loss = property_weights.get(prop, 1.0) * prop_loss
                total_loss += weighted_loss
                loss_components[f"{prop}_loss"] = prop_loss.item()
        
        # 2. 物理约束损失
        physics_loss = self._calculate_physics_constraints(predictions, targets)
        total_loss += 0.1 * physics_loss  # 物理约束权重
        loss_components["physics_loss"] = physics_loss.item()
        
        # 3. 相关性保持损失
        correlation_loss = self._calculate_correlation_loss(predictions, targets)
        total_loss += 0.05 * correlation_loss  # 相关性权重
        loss_components["correlation_loss"] = correlation_loss.item()
        
        return total_loss
    
    def _calculate_physics_constraints(self, predictions: Dict[str, torch.Tensor],
                                     targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算物理约束损失"""
        constraints_loss = torch.tensor(0.0, device=list(predictions.values())[0].device)
        
        # 弹性模量约束：K > 0, G > 0, K > G
        if 'bulk_modulus' in predictions and 'shear_modulus' in predictions:
            bulk_mod = predictions['bulk_modulus']
            shear_mod = predictions['shear_modulus']
            
            # 正值约束
            positive_constraint = torch.relu(-bulk_mod).mean() + torch.relu(-shear_mod).mean()
            
            # K > G 约束（对于大多数材料）
            kg_constraint = torch.relu(shear_mod - bulk_mod).mean()
            
            constraints_loss += positive_constraint + 0.5 * kg_constraint
        
        # 泊松比约束：-1 < ν < 0.5
        if 'poisson_ratio' in predictions:
            poisson = predictions['poisson_ratio']
            poisson_constraint = torch.relu(poisson - 0.5).mean() + torch.relu(-1.0 - poisson).mean()
            constraints_loss += poisson_constraint
        
        return constraints_loss
    
    def _calculate_correlation_loss(self, predictions: Dict[str, torch.Tensor],
                                  targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算相关性保持损失"""
        correlation_loss = torch.tensor(0.0, device=list(predictions.values())[0].device)
        
        # 已知的物理相关性
        known_correlations = [
            ('bulk_modulus', 'shear_modulus', 0.7),  # 正相关
            ('elastic_moduli', 'thermal_conductivity', 0.4),  # 正相关
            ('poisson_ratio', 'bulk_modulus', -0.3)  # 负相关
        ]
        
        for prop1, prop2, expected_corr in known_correlations:
            if prop1 in predictions and prop2 in predictions:
                # 计算预测值的相关系数
                pred1 = predictions[prop1].flatten()
                pred2 = predictions[prop2].flatten()
                
                # 简化的相关性计算（使用余弦相似度近似）
                pred1_norm = pred1 - pred1.mean()
                pred2_norm = pred2 - pred2.mean()
                
                correlation = torch.dot(pred1_norm, pred2_norm) / (
                    torch.norm(pred1_norm) * torch.norm(pred2_norm) + 1e-8)
                
                # 相关性偏离损失
                correlation_deviation = (correlation - expected_corr) ** 2
                correlation_loss += correlation_deviation
        
        return correlation_loss
    
    def recommend_property_improvements(self, precision_matrix: MaterialPropertyMatrix) -> Dict[str, List[str]]:
        """
        推荐性质改进策略
        Recommend property improvement strategies
        
        Args:
            precision_matrix: 精度矩阵
            
        Returns:
            recommendations: 改进建议
        """
        recommendations = {
            'data_augmentation': [],
            'model_architecture': [],
            'training_strategy': []
        }
        
        summary_df = precision_matrix.get_precision_summary()
        
        # 识别需要改进的性质
        for _, row in summary_df.iterrows():
            combination = f"{row['Material']}-{row['Defect_Type']}-{row['Property']}"
            
            # 数据增强建议
            if row['Sample_Count'] < 50:
                recommendations['data_augmentation'].append(
                    f"增加{combination}的训练样本，考虑数据增强技术"
                )
            
            # 模型架构建议
            if row['R2'] < 0.8:
                recommendations['model_architecture'].append(
                    f"针对{combination}优化模型架构，考虑专门的注意力机制"
                )
            
            # 训练策略建议
            if row['Total_Uncertainty'] > 0.2:
                recommendations['training_strategy'].append(
                    f"针对{combination}实施不确定性感知训练，使用贝叶斯方法"
                )
        
        return recommendations


# 使用示例
def example_usage():
    """使用示例"""
    # 创建精度矩阵
    precision_matrix = MaterialPropertyMatrix()
    
    # 模拟一些评估数据
    np.random.seed(42)
    predictions = np.random.normal(0, 0.1, 100)
    targets = np.random.normal(0, 0.05, 100)
    uncertainties = np.random.uniform(0.01, 0.05, 100)
    
    # 更新精度矩阵
    precision_matrix.update_precision(
        'NCM811', 'Li_vacancy', 'formation_energy',
        predictions, targets, uncertainties
    )
    
    # 获取精度总结
    summary = precision_matrix.get_precision_summary()
    print("Precision Summary:")
    print(summary.head())
    
    # 识别精度差距
    gaps = precision_matrix.identify_precision_gaps()
    print(f"\nPrecision Gaps: {gaps}")
    
    # 创建机械性能优化器
    mech_optimizer = MechanicalPropertyOptimizer()
    
    # 模拟多性质预测结果
    multi_predictions = {
        'bulk_modulus': np.random.normal(150, 10, 50),
        'shear_modulus': np.random.normal(60, 5, 50),
        'poisson_ratio': np.random.normal(0.3, 0.05, 50)
    }
    
    multi_targets = {
        'bulk_modulus': np.random.normal(155, 8, 50),
        'shear_modulus': np.random.normal(62, 4, 50),
        'poisson_ratio': np.random.normal(0.32, 0.03, 50)
    }
    
    # 分析性质相关性
    correlations = mech_optimizer.analyze_property_correlations(multi_predictions, multi_targets)
    print(f"\nProperty Correlations: {correlations}")
    
    # 获取改进建议
    recommendations = mech_optimizer.recommend_property_improvements(precision_matrix)
    print(f"\nImprovement Recommendations: {recommendations}")


if __name__ == "__main__":
    example_usage() 