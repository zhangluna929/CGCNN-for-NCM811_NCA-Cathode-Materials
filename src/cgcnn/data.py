"""
Data Loading and Processing for CGCNN

Handles CIF file parsing, graph construction, and batch processing
for crystal structure data used in CGCNN training and inference.

Author: lunazhang
Date: 2023
"""

from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class CIFData(Dataset):
    """
    CIF数据集类，用于加载和处理晶体结构数据
    Dataset for loading and processing crystal structure data from CIF files
    """
    
    def __init__(self, root_dir: str, max_num_nbr: int = 12, radius: float = 8, 
                 dmin: float = 0, step: float = 0.2, random_seed: int = 123):
        """
        初始化CIFData数据集
        
        Parameters
        ----------
        root_dir : str
            包含CIF文件和id_prop.csv的根目录
        max_num_nbr : int
            每个原子的最大邻居数
        radius : float
            搜索半径（埃）
        dmin : float
            最小距离（埃）
        step : float
            高斯展开的步长
        random_seed : int
            随机种子
        """
        self.root_dir = root_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.dmin = dmin
        self.step = step
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 加载原子初始化信息
        self.atom_init_file = os.path.join(root_dir, 'atom_init.json')
        assert os.path.exists(self.atom_init_file), f'atom_init.json not found at {self.atom_init_file}'
        
        with open(self.atom_init_file) as f:
            atom_init = json.load(f)
        self.atom_init = {int(key): value for key, value in atom_init.items()}
        self.atom_features_len = len(list(self.atom_init.values())[0])
        
        # 加载数据标签
        id_prop_file = os.path.join(root_dir, 'id_prop.csv')
        if os.path.exists(id_prop_file):
            with open(id_prop_file) as f:
                reader = f.readlines()
            self.id_prop_data = [line.strip().split(',') for line in reader[1:]]  # 跳过标题行
        else:
            # 如果没有标签文件，只处理CIF文件
            cif_files = [f for f in os.listdir(root_dir) if f.endswith('.cif')]
            self.id_prop_data = [[f.replace('.cif', ''), '0.0'] for f in cif_files]
            warnings.warn(f"id_prop.csv not found in {root_dir}, using dummy labels")
        
        # 高斯展开的参数
        self.gdf_filter = np.arange(dmin, radius + step, step)
        
    def __len__(self):
        return len(self.id_prop_data)
    
    def __getitem__(self, idx: int):
        """
        获取单个样本
        """
        cif_id = self.id_prop_data[idx][0]
        cif_file = os.path.join(self.root_dir, f'{cif_id}.cif')
        
        # 加载晶体结构
        crystal = Structure.from_file(cif_file)
        
        # 获取原子特征
        atom_fea = np.vstack([self.atom_init[atom.specie.number] for atom in crystal])
        
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(f'{cif_id} not enough neighbors ({len(nbr)})')
                # 用第一个邻居填充
                nbr += [nbr[0]] * (self.max_num_nbr - len(nbr))
            nbr = nbr[:self.max_num_nbr]
            
            nbr_fea_idx.append([int(x[2]) for x in nbr])
            nbr_fea.append([self._get_nbr_fea(x[1]) for x in nbr])
            
        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)
        
        # 获取标签
        target = float(self.id_prop_data[idx][1]) if len(self.id_prop_data[idx]) > 1 else 0.0
        
        # 处理分类标签（如果存在）
        if len(self.id_prop_data[idx]) > 2:
            cls_target = int(self.id_prop_data[idx][2])
        else:
            cls_target = 0
        
        return (torch.Tensor(atom_fea),
                torch.Tensor(nbr_fea),
                torch.LongTensor(nbr_fea_idx),
                torch.Tensor([target]),
                torch.LongTensor([cls_target]),
                cif_id)
    
    def _get_nbr_fea(self, distance: float) -> np.ndarray:
        """
        获取邻居特征（高斯展开）
        """
        return np.exp(-(distance - self.gdf_filter)**2 / (self.step**2))


def collate_pool(dataset_list: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    将数据集中的样本整理成批次
    Collate function to batch samples from dataset
    
    Parameters
    ----------
    dataset_list : List[Tuple]
        数据样本列表
        
    Returns
    -------
    Tuple of tensors for batched data
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target, batch_cls_target = [], [], []
    batch_cif_ids = []
    base_idx = 0
    
    for i, (atom_fea, nbr_fea, nbr_fea_idx, target, cls_target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # 晶体中的原子数
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cls_target.append(cls_target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
        
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
            torch.stack(batch_target, dim=0),
            torch.stack(batch_cls_target, dim=0),
            batch_cif_ids) 