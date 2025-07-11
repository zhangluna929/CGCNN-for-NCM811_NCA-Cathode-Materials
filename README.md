# CGCNN-Ni: 高镍层状正极（NCM811 / NCA）图卷积预测与主动学习平台  
# CGCNN-Ni: Crystal Graph Convolutional Neural Network Toolkit for High-Ni Layered Cathodes (NCM811, NCA)

中英双语文档 / Bilingual Documentation (Chinese-English)

---

## 项目亮点 (Highlights)

中文 | English
--- | ---
秒级多属性预测：9 个预训练模型（形成能、带隙、弹性模量等） | Instant property inference with nine pre-trained models (formation energy, band gap, elastic moduli, etc.)
统一任务接口：支持回归、分类、多任务 | Unified interface for regression, classification and multi-task learning
主动学习脚本：Bayesian Optimisation + MC-Dropout 不确定度 | Built-in active-learning loop with Bayesian optimisation and MC-Dropout uncertainty
易于微调：`predict.predict()` 可直接载入 `.pth.tar` 继续训练 | Easy fine-tuning via importable `predict.predict()`

---

## 目录结构 (Directory Layout)

```text
cgcnn_project/
├── cgcnn/                # Core library
│   ├── data.py           # CIFData & collate_pool
│   ├── model.py          # CrystalGraphConvNet / multi-task variant
│   └── __init__.py
├── main.py               # Training entry
├── predict.py            # Importable inference utility
├── active_learning.py    # Bayesian optimisation & active-learning demo
├── pre-trained/          # Pre-trained checkpoints
│   └── README.md         # Model catalogue (bilingual)
├── vacancy_data/         # Li-vacancy example structures
└── ...
```

---

## 环境安装 (Installation)

```bash
conda create -n cgcnn-ni python=3.9 pytorch scikit-learn pymatgen -c pytorch -c conda-forge
conda activate cgcnn-ni
# optional packages for active learning
pip install scikit-optimize   # or botorch, deap
```

PyTorch ≥1.10 推荐使用 GPU；若无 GPU，CPU 亦可运行。  
GPU recommended (PyTorch ≥1.10), CPU supported.

---

## 快速开始 (Quick Start)

### 1. 使用预训练模型推断 (Inference with pre-trained weights)

```python
from predict import predict

results = predict(
    'pre-trained/formation-energy-per-atom.pth.tar',
    cif_dir='vacancy_data',
    n_dropout=25           # MC-Dropout passes for uncertainty
)
for cid, (mu, sigma) in results.items():
    print(f'{cid:24s}  {mu:8.3f} ± {sigma:.3f} eV/atom')
```

命令行调用 (CLI):
```bash
python predict.py pre-trained/band-gap.pth.tar vacancy_data --n-dropout 25
```

### 2. 训练与微调 (Training / Fine-tuning)

```bash
# Fine-tune for formation energy, 30 epochs
python main.py data/sample-regression --task regression \
       --epochs 30 --resume pre-trained/formation-energy-per-atom.pth.tar

# Multi-task: formation energy + defect classification
python main.py vacancy_data --task multi --cls-weight 0.5 --epochs 50
```

### 3. 主动学习循环 (Active-learning loop)

```bash
python active_learning.py   # 100 BO steps, DFT placeholder every 20 steps
```

流程：生成候选 CIF → `predict()` 评分 ± σ → BO 决策 → 定期选样本跑 DFT → 将真值追加数据集并自动微调。

> 将 `mutate_cif()` 与 `run_dft_simulation()` 替换为自己的结构生成及计算流程。

---

## 预训练权重 (Pre-trained checkpoints)

详细列表见 `pre-trained/README.md`。  
文件较大，使用 Git LFS 管理；若 clone 后未下载，请执行 `git lfs pull`。

| 文件 | 物性 / Property | 典型 MAE* |
|------|----------------|-----------|
| formation-energy-per-atom.pth.tar | Formation energy | 0.04–0.10 eV/atom |
| band-gap.pth.tar                 | Band gap         | 0.32 eV |
| bulk-moduli.pth.tar              | Bulk modulus (log) | 0.04 |
| …                                | …                | … |

\*Error on Materials Project hold-out; for reference only.

---

## 数据格式 (Dataset format)

`id_prop.csv`
```csv
cif_id,formation_energy,defect_label
NCM811_LiVac0,-3.812,1
```
* 回归标签可单列；分类标签可省略或设为空。  
* 同名 `.cif` 与 `atom_init.json` 需位于同一目录。

---

## 引用 (Citation)

如果本项目对您有帮助，请引用 / If this toolkit helps your work, please cite:

* T. Xie & J. C. Grossman, *Phys. Rev. Lett.* 120, 145301 (2018).
* A. Jain et al., *APL Materials* 1, 011002 (2013).

---

## 许可证 (License)

MIT License – free for academic and industrial use with proper attribution.



