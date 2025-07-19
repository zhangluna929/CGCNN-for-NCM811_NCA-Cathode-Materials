# CGCNN for High-Ni Cathode Materials
# 高镍正极材料的晶体图卷积神经网络研究

> Crystal Graph Convolutional Neural Networks for property prediction and defect engineering in high-Ni layered cathode materials (NCM811/NCA)  
> 基于晶体图卷积神经网络的高镍层状正极材料物性预测与缺陷工程研究

**Author**: lunazhang  
**Date**: 2023  

---

## Abstract / 摘要

This research implements Crystal Graph Convolutional Neural Networks (CGCNN) for materials property prediction in high-Ni layered cathode materials. The framework enables rapid screening of defect structures and composition optimization for NCM811 and NCA materials through machine learning-accelerated materials discovery.

本研究实现了晶体图卷积神经网络用于高镍层状正极材料的物性预测。该框架通过机器学习加速的材料发现方法，实现对NCM811和NCA材料缺陷结构的快速筛选和成分优化。

**Key Features / 主要特性**:
- Multi-property prediction (formation energy, band gap, elastic moduli, etc.)
- Bayesian optimization for active learning  
- Crystal structure representation as graphs
- Uncertainty quantification via MC-Dropout

## Methodology / 方法

### 1. Graph Neural Network Architecture / 图神经网络架构

The CGCNN algorithm represents crystal structures as directed multigraphs where atoms are nodes and bonds are edges. Each atomic node contains learnable feature vectors, and graph convolution operations aggregate neighbor information to predict materials properties.

CGCNN算法将晶体结构表示为有向多重图，其中原子为节点，化学键为边。每个原子节点包含可学习的特征向量，图卷积操作聚合邻居信息以预测材料性质。

### 2. Active Learning Strategy / 主动学习策略  

Bayesian optimization guides the exploration of chemical space through uncertainty-aware sampling and iterative model improvement with experimental feedback.

贝叶斯优化通过不确定性感知采样和实验反馈的迭代模型改进来指导化学空间的探索。

### 3. Performance Comparison / 性能对比

| Aspect / 方面 | DFT Calculations | CGCNN Approach |
|------|----------|--------|
| **Speed / 速度** | Hours to days / 数小时到数天 | Seconds / 秒级 |
| **Scale / 规模** | 10-100 structures / 10-100个结构 | 1000+ structures / 1000+结构 |
| **Uncertainty / 不确定性** | Not quantified / 无量化 | MC-Dropout / 蒙特卡洛Dropout |
| **优化策略 / Optimisation Strategy** | 人工经验 / Human Experience | 贝叶斯自动优化 / Bayesian Auto-Optimization |
---

## Implementation Details / 实现细节

### 1. Network Architecture / 网络架构
- **Input**: Crystal structure (CIF) → Atomic graph representation
- **Processing**: Multi-layer graph convolution + pooling + dense layers  
- **Output**: Property predictions with uncertainty estimates

网络将晶体结构转换为原子图表示，通过多层图卷积处理后输出物性预测和不确定性估计。

```
Crystal Structure → Graph Representation → Conv Layers → Pooling → Property Prediction
    CIF文件      →      图表示        →   卷积层    →  池化   →      物性预测
```

### 2. Supported Properties / 支持的物性

| Property / 物性 | Units / 单位 | Applications / 应用 |
|------|------|----------|
| Formation energy / 形成能 | eV/atom | Thermodynamic stability / 热力学稳定性 |
| Band gap / 带隙 | eV | Electronic properties / 电子性质 |
| Elastic moduli / 弹性模量 | log(GPa) | Mechanical properties / 力学性质 |
| Fermi level / 费米能级 | eV | Electronic structure / 电子结构 |
| Poisson ratio / 泊松比 | - | Mechanical response / 力学响应 |

---

## Usage / 使用方法

### Environment Setup / 环境配置
```bash
# Create conda environment / 创建conda环境
conda env create -f environment.yml
conda activate cgcnn

# Manual installation / 手动安装
conda create -n cgcnn python=3.9 pytorch scikit-learn pymatgen -c pytorch -c conda-forge
pip install scikit-optimize
```

### 1. Property Prediction / 物性预测

**Python API**:
```python
from predict import predict

# Predict formation energy / 预测形成能
results = predict(
    model_path='pre-trained/formation-energy-per-atom.pth.tar',
    cif_dir='vacancy_data',
    n_dropout=25  # Uncertainty estimation / 不确定性估计
)

# Display results / 显示结果
for cif_id, (mean, std) in results.items():
    print(f'{cif_id}: {mean:.3f} ± {std:.3f} eV/atom')
```

**Command Line**:
```bash
# Predict formation energy / 预测形成能
python predict.py pre-trained/formation-energy-per-atom.pth.tar vacancy_data --n-dropout 25

# Predict band gap / 预测带隙
python predict.py pre-trained/band-gap.pth.tar vacancy_data --n-dropout 25
```

### 2. Active Learning / 主动学习

```bash
# Start active learning loop / 启动主动学习循环
python active_learning.py
```

**Custom Workflow / 自定义流程**:
```python
# Modify search space in active_learning.py / 修改搜索空间
space = [
    Real(0.0, 1.0, name='li_vacancy_ratio'),    # Li vacancy ratio / 锂空位比例
    Real(0.6, 0.9, name='ni_content'),          # Ni content / 镍含量
    Real(0.0, 0.3, name='co_content'),          # Co content / 钴含量
]

# Custom structure generation / 自定义结构生成
def mutate_cif(base_cif, li_vac, ni_frac, step):
    # Implement your structure generation logic
    pass

# DFT calculation interface / DFT计算接口
def run_dft_simulation(cif_path):
    # Call VASP/Quantum ESPRESSO etc.
    pass
```

### 3. Model Training / 模型训练

```bash
# Train from scratch / 从头训练
python main.py vacancy_data --task regression --epochs 100

# Fine-tuning / 微调
python main.py vacancy_data --task regression --epochs 30 \
    --resume pre-trained/formation-energy-per-atom.pth.tar

# Multi-task learning / 多任务学习
python main.py vacancy_data --task multi --cls-weight 0.5 --epochs 50
```

---

## Pre-trained Models / 预训练模型

Nine pre-trained models based on Materials Project dataset:

基于Materials Project数据集的九个预训练模型：

| Model File / 模型文件 | Property / 物性 | MAE | Applications / 应用 |
|----------|----------|-----|----------|
| `formation-energy-per-atom.pth.tar` | Formation energy / 形成能 | 0.04-0.10 eV/atom | Stability screening / 稳定性筛选 |
| `band-gap.pth.tar` | Band gap / 带隙 | 0.32 eV | Electronic properties / 电子性质 |
| `bulk-moduli.pth.tar` | Bulk modulus / 体模量 | 0.04 log(GPa) | Mechanical properties / 力学性质 |
| `shear-moduli.pth.tar` | Shear modulus / 剪切模量 | - | Deformation resistance / 变形抗性 |
| `efermi.pth.tar` | Fermi level / 费米能级 | - | Electronic structure / 电子结构 |
| `poisson-ratio.pth.tar` | Poisson ratio / 泊松比 | - | Mechanical response / 力学响应 |

> See [`pre-trained/README.md`](pre-trained/README.md) for detailed information

---

## 数据格式与准备

### 项目目录结构 / Project Directory Structure
```
NCM811_NCA_Cgcnn_Project/
├── README.md                    # 项目说明文档 / Project documentation
├── LICENSE                      # MIT开源协议 / MIT license
├── .gitignore                   # Git忽略文件 / Git ignore file
├── requirements.txt             # pip依赖文件 / pip requirements
├── environment.yml              # Conda环境配置 / Conda environment
├── atom_init.json              # 原子初始化参数 / Atom initialization parameters
├── main.py                     # 训练主脚本 / Main training script
├── model.py                    # 向后兼容的模型文件 / Backward compatible model file
├── predict.py                  # 预测接口 / Prediction interface
├── active_learning.py          # 主动学习脚本 / Active learning script
├── utils.py                    # 工具函数 / Utility functions
├── cgcnn/                      # CGCNN核心模块 / CGCNN core module
│   ├── __init__.py             # 模块初始化 / Module initialization
│   ├── model.py                # 模型定义 / Model definitions
│   ├── data.py                 # 数据加载 / Data loading
│   ├── uncertainty.py          # 不确定性量化 / Uncertainty quantification
│   ├── features.py             # 特征工程 / Feature engineering
│   ├── interpretability.py     # 可解释性分析 / Interpretability analysis
│   ├── multimodal.py           # 多模态融合 / Multimodal fusion
│   ├── physics.py              # 物理约束 / Physics constraints
│   └── integrations/           # 外部工具集成 / External integrations
│       ├── __init__.py
│       ├── ase_interface.py    # ASE接口 / ASE interface
│       └── materials_project.py # Materials Project API
├── pre-trained/                # 预训练模型库 / Pre-trained models
│   ├── README.md               # 模型说明文档 / Model documentation
│   ├── formation-energy-per-atom.pth.tar
│   ├── band-gap.pth.tar
│   ├── bulk-moduli.pth.tar
│   ├── shear-moduli.pth.tar
│   ├── efermi.pth.tar
│   ├── poisson-ratio.pth.tar
│   ├── final-energy-per-atom.pth.tar
│   └── semi-metal-classification.pth.tar
└── vacancy_data/               # 示例数据集 / Example dataset
    ├── id_prop.csv             # 标签文件 / Label file
    ├── atom_init.json          # 原子参数 / Atom parameters
    ├── make_li_vac.py          # 缺陷生成脚本 / Defect generation script
    ├── build_clean_cif.py      # CIF清理脚本 / CIF cleaning script
    ├── NCM811_LiVac0.cif       # NCM811锂空位结构 / NCM811 Li-vacancy
    ├── NCA_LiVac0.cif          # NCA锂空位结构 / NCA Li-vacancy
    └── *.cif                   # 其他结构文件 / Additional structures
```

### 标签文件格式 / Label File Format
```csv
cif_id,formation_energy,defect_classification
NCM811_LiVac0,-3.812,1
NCA_LiVac0,-3.705,1
NCM811_pristine,-4.123,0
```

### 缺陷结构生成
```bash
# 自动生成锂空位结构
cd vacancy_data
python make_li_vac.py  # 删除第一个Li原子生成空位
```

---

## 应用案例

### 案例1：NCM811材料稳定性筛选
```python
# 批量预测形成能，筛选稳定结构
results = predict('pre-trained/formation-energy-per-atom.pth.tar', 'candidate_structures/')
stable_structures = {k: v for k, v in results.items() if v[0] < -3.5}  # 阈值筛选
```

### 案例2：优化镍含量与空位浓度
```python
# 在active_learning.py中定义2D搜索空间
space = [Real(0.0, 0.2, name='vacancy_ratio'), 
         Real(0.7, 0.9, name='ni_ratio')]
# 自动寻找最优组合
```

### 案例3：多物性协同优化
```python
# 同时考虑形成能和带隙
formation_pred = predict('pre-trained/formation-energy-per-atom.pth.tar', 'structures/')
bandgap_pred = predict('pre-trained/band-gap.pth.tar', 'structures/')
# 多目标优化逻辑
```

---

## 高级功能

### MC-Dropout不确定性估计
```python
# 启用不确定性量化
results = predict(model_path, cif_dir, n_dropout=50)
for cif_id, (mean, uncertainty) in results.items():
    if uncertainty > threshold:
        print(f"{cif_id} 需要DFT验证：不确定性 {uncertainty:.3f}")
```

### 自定义模型架构
```python
from model import CrystalGraphConvNet, CrystalGraphConvNetMulti

# 单任务模型
model = CrystalGraphConvNet(
    orig_atom_fea_len=92, nbr_fea_len=41,
    atom_fea_len=64, n_conv=3, h_fea_len=128
)

# 多任务模型
multi_model = CrystalGraphConvNetMulti(
    orig_atom_fea_len=92, nbr_fea_len=41
)
```

---

## 性能基准

### 计算效率对比
| 方法 | 单结构预测时间 | 批量处理能力 | GPU加速 |
|------|----------------|--------------|---------|
| DFT (VASP) | 2-24小时 | 并行受限 | 部分支持 |
| **CGCNN** | **0.1-1秒** | **>1000结构/分钟** | **完全支持** |

### 预测精度验证
- **形成能预测**：在Materials Project测试集上MAE < 0.1 eV/atom
- **带隙预测**：与DFT结果相关系数R² > 0.85
- **分类任务**：缺陷识别准确率 > 90%

---

## 贡献指南

欢迎提交Issue和PR！特别期待：

- [ ] 新增其他正极材料类型支持（如磷酸铁锂、尖晶石等）
- [ ] 集成更多DFT计算接口（Quantum ESPRESSO、CP2K等）
- [ ] 开发在线预测Web界面
- [ ] 添加更多主动学习策略（如基于信息增益的采样）

### 开发环境设置
```bash
git clone https://github.com/lunazhang/NCM811_NCA_Cgcnn_Project.git
cd NCM811_NCA_Cgcnn_Project
conda env create -f environment.yml
conda activate cgcnn_2023
```
---

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议，欢迎学术和工业界自由使用。


---

<div align="center">

**如果这个项目对您有帮助，请给我一个Star！**  
**If this project helps you, please give me a Star!**

*Computational Materials Science Research*
*计算材料科学研究*




