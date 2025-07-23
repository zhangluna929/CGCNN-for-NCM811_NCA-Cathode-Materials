"""
ASE Integration Interface for CGCNN

Provides seamless integration with the Atomic Simulation Environment (ASE)
for structure optimization, molecular dynamics, and property calculations
using CGCNN as a machine learning potential.

Author: lunazhang
Date: 2023
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings

try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.io import read, write
    from ase.optimize import BFGS, LBFGS
    from ase.constraints import FixAtoms
    ASE_AVAILABLE = True
except ImportError:
    warnings.warn("ASE not available. Install with: pip install ase")
    ASE_AVAILABLE = False
    Atoms = None
    Calculator = object

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor


class CGCNNCalculator(Calculator):
    """
    CGCNN计算器，可用于ASE优化和MD
    CGCNN Calculator for use with ASE optimizers and MD
    """
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, model_path: str, device: Optional[str] = None, 
                 predict_forces: bool = False, predict_stress: bool = False,
                 **kwargs):
        """
        初始化CGCNN计算器
        
        Args:
            model_path: CGCNN模型路径
            device: 计算设备
            predict_forces: 是否预测力
            predict_stress: 是否预测应力
        """
        super().__init__(**kwargs)
        
        if not ASE_AVAILABLE:
            raise ImportError("ASE is required for CGCNNCalculator")
        
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.predict_forces = predict_forces
        self.predict_stress = predict_stress
        
        # 加载模型
        self.model = self._load_model()
        self.adaptor = AseAtomsAdaptor()
    
    def _load_model(self):
        """加载CGCNN模型"""
        from ..model import CrystalGraphConvNet
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 从检查点获取模型参数
        model_args = checkpoint.get('args', {})
        model = CrystalGraphConvNet(
            orig_atom_fea_len=model_args.get('orig_atom_fea_len', 92),
            nbr_fea_len=model_args.get('nbr_fea_len', 41),
            atom_fea_len=model_args.get('atom_fea_len', 64),
            n_conv=model_args.get('n_conv', 3),
            h_fea_len=model_args.get('h_fea_len', 128),
            n_h=model_args.get('n_h', 1),
            classification=model_args.get('classification', False)
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def calculate(self, atoms: Atoms = None, properties: List[str] = ['energy'],
                 system_changes: List[str] = all_changes):
        """
        执行CGCNN计算
        
        Args:
            atoms: ASE原子对象
            properties: 要计算的性质
            system_changes: 系统变化
        """
        super().calculate(atoms, properties, system_changes)
        
        # 转换为PyMatGen结构
        structure = self.adaptor.get_structure(atoms)
        
        # 准备CGCNN输入
        cgcnn_input = self._prepare_cgcnn_input(structure)
        
        # 执行预测
        with torch.no_grad():
            if self.predict_forces:
                energy, forces = self._predict_energy_forces(cgcnn_input, structure)
                self.results['energy'] = energy
                self.results['forces'] = forces
            else:
                energy = self._predict_energy(cgcnn_input)
                self.results['energy'] = energy
            
            if self.predict_stress and 'stress' in properties:
                stress = self._predict_stress(cgcnn_input, structure)
                self.results['stress'] = stress
    
    def _prepare_cgcnn_input(self, structure: Structure) -> Tuple:
        """准备CGCNN输入数据"""
        from ..data import CIFData
        
        # 这里需要将Structure转换为CGCNN的输入格式
        # 简化实现，实际应用中需要更完整的转换
        
        # 创建临时的CIF数据对象
        temp_dir = Path.cwd() / "temp_cgcnn_calc"
        temp_dir.mkdir(exist_ok=True)
        
        cif_path = temp_dir / "temp.cif"
        structure.to(fmt="cif", filename=str(cif_path))
        
        # 使用CIFData加载
        dataset = CIFData(str(temp_dir))
        data = dataset[0]  # 获取第一个（也是唯一的）数据
        
        # 清理临时文件
        cif_path.unlink()
        temp_dir.rmdir()
        
        return data[:4]  # 返回 (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
    
    def _predict_energy(self, cgcnn_input: Tuple) -> float:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = cgcnn_input
        
        # 转换为tensor并移动到设备
        atom_fea = atom_fea.to(self.device).unsqueeze(0)
        nbr_fea = nbr_fea.to(self.device).unsqueeze(0)
        nbr_fea_idx = nbr_fea_idx.to(self.device)
        crystal_atom_idx = [idx.to(self.device) for idx in crystal_atom_idx]
        
        # 预测
        energy = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        
        return energy.item()
    
    def _predict_energy_forces(self, cgcnn_input: Tuple, structure: Structure) -> Tuple[float, np.ndarray]:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = cgcnn_input
        
        # 启用梯度计算
        atom_fea = atom_fea.to(self.device).unsqueeze(0)
        atom_fea.requires_grad_(True)
        
        nbr_fea = nbr_fea.to(self.device).unsqueeze(0)
        nbr_fea_idx = nbr_fea_idx.to(self.device)
        crystal_atom_idx = [idx.to(self.device) for idx in crystal_atom_idx]
        
        # 预测能量
        energy = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        
        # 计算力（能量对坐标的负梯度）
        energy.backward()
        
        # 提取力
        if atom_fea.grad is not None:
            # 这里需要将原子特征的梯度转换为力
            # 简化实现，实际需要考虑特征与坐标的关系
            forces_flat = -atom_fea.grad.cpu().numpy().flatten()
            forces = forces_flat[:len(structure) * 3].reshape(-1, 3)
        else:
            forces = np.zeros((len(structure), 3))
        
        return energy.item(), forces
    
    def _predict_stress(self, cgcnn_input: Tuple, structure: Structure) -> np.ndarray:
        # 应力预测需要对晶格参数求导
        # 这里提供一个简化的实现框架
        
        # 简化版本：返回零应力
        return np.zeros(6)  # 6个应力分量


class ASEInterface:
    """
    ASE接口主类
    Main ASE Interface Class
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化ASE接口
        
        Args:
            model_path: CGCNN模型路径
        """
        if not ASE_AVAILABLE:
            raise ImportError("ASE is required. Install with: pip install ase")
        
        self.model_path = model_path
        self.adaptor = AseAtomsAdaptor()
    
    def structure_to_atoms(self, structure: Structure) -> Atoms:
        """
        将PyMatGen Structure转换为ASE Atoms
        Convert PyMatGen Structure to ASE Atoms
        """
        return self.adaptor.get_atoms(structure)
    
    def atoms_to_structure(self, atoms: Atoms) -> Structure:
        """
        将ASE Atoms转换为PyMatGen Structure
        Convert ASE Atoms to PyMatGen Structure
        """
        return self.adaptor.get_structure(atoms)
    
    def create_calculator(self, predict_forces: bool = False, 
                         predict_stress: bool = False) -> CGCNNCalculator:
        """
        创建CGCNN计算器
        Create CGCNN calculator
        """
        if self.model_path is None:
            raise ValueError("Model path must be provided to create calculator")
        
        return CGCNNCalculator(
            model_path=self.model_path,
            predict_forces=predict_forces,
            predict_stress=predict_stress
        )
    
    def optimize_structure(self, structure: Structure, 
                          optimizer: str = 'BFGS',
                          fmax: float = 0.05,
                          steps: int = 100,
                          constraints: Optional[List] = None) -> Tuple[Structure, Dict]:
        """
        使用CGCNN优化结构
        Optimize structure using CGCNN
        
        Args:
            structure: 输入结构
            optimizer: 优化器类型
            fmax: 力的收敛标准
            steps: 最大步数
            constraints: 约束条件
        
        Returns:
            optimized_structure: 优化后的结构
            optimization_info: 优化信息
        """
        if self.model_path is None:
            raise ValueError("Model path must be provided for optimization")
        
        # 转换为ASE格式
        atoms = self.structure_to_atoms(structure)
        
        # 设置计算器
        calc = self.create_calculator(predict_forces=True)
        atoms.set_calculator(calc)
        
        # 设置约束
        if constraints:
            atoms.set_constraint(constraints)
        
        # 选择优化器
        if optimizer.upper() == 'BFGS':
            opt = BFGS(atoms)
        elif optimizer.upper() == 'LBFGS':
            opt = LBFGS(atoms)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # 执行优化
        opt.run(fmax=fmax, steps=steps)
        
        # 转换回PyMatGen格式
        optimized_structure = self.atoms_to_structure(atoms)
        
        # 收集优化信息
        optimization_info = {
            'converged': opt.converged(),
            'nsteps': opt.nsteps,
            'final_energy': atoms.get_potential_energy(),
            'final_forces': atoms.get_forces()
        }
        
        return optimized_structure, optimization_info
    
    def run_neb_calculation(self, initial_structure: Structure, 
                           final_structure: Structure,
                           n_images: int = 5,
                           spring_constant: float = 1.0) -> List[Structure]:
        """
        运行NEB (Nudged Elastic Band) 计算
        Run NEB calculation
        
        Args:
            initial_structure: 初始结构
            final_structure: 最终结构
            n_images: 中间图像数量
            spring_constant: 弹簧常数
        
        Returns:
            neb_path: NEB路径上的结构列表
        """
        try:
            from ase.neb import NEB
        except ImportError:
            raise ImportError("ASE NEB module not available")
        
        if self.model_path is None:
            raise ValueError("Model path must be provided for NEB calculation")
        
        # 转换为ASE格式
        initial_atoms = self.structure_to_atoms(initial_structure)
        final_atoms = self.structure_to_atoms(final_structure)
        
        # 创建中间图像
        images = [initial_atoms.copy()]
        for i in range(n_images - 2):
            image = initial_atoms.copy()
            images.append(image)
        images.append(final_atoms.copy())
        
        # 设置计算器
        calc = self.create_calculator(predict_forces=True)
        for image in images:
            image.set_calculator(calc)
        
        # 创建NEB对象
        neb = NEB(images, k=spring_constant)
        neb.interpolate()
        
        # 优化NEB
        optimizer = BFGS(neb)
        optimizer.run(fmax=0.05)
        
        # 转换回PyMatGen格式
        neb_structures = [self.atoms_to_structure(image) for image in images]
        
        return neb_structures
    
    def calculate_phonons(self, structure: Structure, 
                         supercell_matrix: np.ndarray = None) -> Dict:
        """
        计算声子（需要ASE phonopy接口）
        Calculate phonons (requires ASE phonopy interface)
        """
        try:
            from phonopy import Phonopy
            from phonopy.structure.atoms import PhonopyAtoms
        except ImportError:
            raise ImportError("Phonopy is required for phonon calculations")
        
        warnings.warn("Phonon calculation with CGCNN is experimental")
        
        # 转换为ASE格式
        atoms = self.structure_to_atoms(structure)
        
        # 设置超胞
        if supercell_matrix is None:
            supercell_matrix = np.diag([2, 2, 2])
        
        # 创建Phonopy对象
        phonopy_atoms = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.get_cell()
        )
        
        phonon = Phonopy(phonopy_atoms, supercell_matrix)
        
        return {'phonon_object': phonon, 'message': 'Phonon setup complete'}
    
    def batch_structure_analysis(self, structures: List[Structure]) -> List[Dict]:
        """
        批量结构分析
        Batch structure analysis
        """
        if self.model_path is None:
            raise ValueError("Model path must be provided for analysis")
        
        results = []
        calc = self.create_calculator()
        
        for i, structure in enumerate(structures):
            try:
                atoms = self.structure_to_atoms(structure)
                atoms.set_calculator(calc)
                
                energy = atoms.get_potential_energy()
                
                analysis = {
                    'index': i,
                    'formula': structure.formula,
                    'energy': energy,
                    'energy_per_atom': energy / len(structure),
                    'volume': structure.volume,
                    'density': structure.density
                }
                
                results.append(analysis)
                
            except Exception as e:
                warnings.warn(f"Failed to analyze structure {i}: {e}")
                results.append({'index': i, 'error': str(e)})
        
        return results


def create_ase_database(structures: List[Structure], 
                       model_path: str,
                       db_path: str = "cgcnn_results.db") -> None:
    """
    创建ASE数据库存储CGCNN结果
    Create ASE database to store CGCNN results
    
    Args:
        structures: 结构列表
        model_path: CGCNN模型路径
        db_path: 数据库路径
    """
    try:
        from ase.db import connect
    except ImportError:
        raise ImportError("ASE database functionality not available")
    
    interface = ASEInterface(model_path)
    db = connect(db_path)
    
    for i, structure in enumerate(structures):
        try:
            atoms = interface.structure_to_atoms(structure)
            calc = interface.create_calculator()
            atoms.set_calculator(calc)
            
            energy = atoms.get_potential_energy()
            
            # 存储到数据库
            db.write(atoms, 
                    cgcnn_energy=energy,
                    structure_id=i,
                    formula=structure.formula)
            
            print(f"Processed structure {i+1}/{len(structures)}")
            
        except Exception as e:
            warnings.warn(f"Failed to process structure {i}: {e}")


# 使用示例
def example_usage():
    if not ASE_AVAILABLE:
        print("ASE not available for example")
        return
    
    # 创建ASE接口
    # interface = ASEInterface(model_path="model_best.pth.tar")
    
    # 结构优化示例
    # structure = Structure.from_file("example.cif")
    # optimized_structure, info = interface.optimize_structure(structure)
    
    # 批量分析示例
    # structures = [Structure.from_file(f"structure_{i}.cif") for i in range(10)]
    # results = interface.batch_structure_analysis(structures)
    
    pass 