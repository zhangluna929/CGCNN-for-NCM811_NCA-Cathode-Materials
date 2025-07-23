# 高镍正极材料性质预测与缺陷工程 —— CGCNN 工具集  
# CGCNN Toolkit for High-Ni Cathode Property Prediction and Defect Engineering

---

## 1. 项目标题 (Project Title)
高镍层状正极晶体图卷积神经网络框架  
Crystal Graph Convolutional Neural Network Framework for Ni-Rich Layered Cathodes

## 2. 项目简介 (Project Description)
- **中文**：本项目基于 Crystal Graph Convolutional Neural Networks (CGCNN)，针对 NCM811 与 NCA 等高镍层状氧化物建立结构-性质预测模型，旨在在秒级时间尺度内给出近 DFT 精度的形成能、带隙及弹性模量等关键物性，并通过主动学习与不确定度评估实现缺陷结构的高效筛选。
- **English**: This repository provides a CGCNN-based workflow for Ni-rich layered cathodes (e.g., NCM811, NCA). It delivers near-DFT accuracy for formation energy, band gap, elastic moduli, etc., within seconds and integrates active learning and uncertainty quantification for efficient defect exploration.

## 3. 功能与亮点 (Features & Highlights)
| 功能 | Feature | 说明 / Notes |
|------|---------|--------------|
| 多物性预测 | Multi-property prediction | Formation energy, band gap, elastic moduli, Fermi level |
| 不确定度分解 | Uncertainty decomposition | Epistemic & Aleatoric components |
| 主动学习循环 | Active learning loop | Bayesian optimization with MC-Dropout ranking |
| 可解释性分析 | Interpretability | Atom-/bond-level contribution maps |
| 多尺度卷积 & 注意力 | Multi-scale GCN & Attention | Captures long-range interactions |
| ASE 接口 | ASE interface | Acts as ML potential for structure relaxation |

## 4. 技术栈 (Tech Stack)
- Python ≥ 3.9, PyTorch ≥ 1.10  
- PyMatGen, ASE  
- scikit-learn, scikit-optimize  
- 依赖版本详见 `configs/environment.yml`

## 5. 安装与配置 (Installation & Setup)
```bash
# Conda
conda env create -f configs/environment.yml
conda activate cgcnn

# 或 Manual / Manual setup
conda create -n cgcnn python=3.9 pytorch -c pytorch -c conda-forge
pip install -r configs/requirements.txt
```

## 6. 使用方法 (Usage)
### 6.1 CLI
```bash
# Formation energy with dropout-based uncertainty
activate cgcnn
python src/predict.py pre-trained/formation-energy-per-atom.pth.tar vacancy_data --n-dropout 25
```
### 6.2 Python API
```python
from src.predict import predict
results = predict('pre-trained/formation-energy-per-atom.pth.tar', 'vacancy_data', n_dropout=25)
for cid, (mu, sigma) in results.items():
    print(f"{cid}: {mu:.3f} ± {sigma:.3f} eV/atom")
```
### 6.3 Active Learning
```bash
python src/active_learning.py  # default search space inside script
```
### 6.4 Training from Scratch
```bash
python src/main.py vacancy_data --task regression --epochs 100
```

## 7. 数据与实验 (Data & Experiments)
- **数据 (Data)**: `data/` 目录包含经过清洗的 CIF、标签 (`id_prop.csv`) 及分析结果。
- **基准 (Benchmarks)**:
  | 任务 / Task | 指标 / Metric | 值 / Value |
  |-------------|--------------|------------|
  | Formation energy | MAE | < 0.10 eV/atom |
  | Band gap | MAE | ≈ 0.32 eV |
  | Defect classification | Accuracy | > 90 % |
- 复现脚本见 `scripts/evaluation/`，详细结果见 `data/analysis/`。

## 8. 贡献 (Contributing)
1. Fork 本仓库并创建分支。  
2. 遵循 PEP8，提交前确保 `pytest` 通过。  
3. 通过 Pull Request 描述变更动机及测试结果。

## 9. 许可证 (License)
本项目采用 MIT 许可证，详情见 `LICENSE`。  
MIT License – see `LICENSE` for details.

## 10. 作者 (Author)
LunaZhang

## 11. 未来工作 (Future Work)
- 支持更多 Ni-rich 体系 (NCMx90, NCA-F)  
- 引入自监督预训练以降低标注需求  
- 云端推理与可视化仪表盘  
- CI / CD & Benchmark 自动化

## 12. 项目结构 (Project Structure)
```text
├── configs/              # 环境与依赖
├── data/                 # 数据集与分析
├── docs/README.md        # 当前文档
├── models/               # 训练检查点
├── pre-trained/          # 公布的模型权重
├── scripts/              # 数据处理 & 评估脚本
├── src/                  # 主要源码
│   ├── cgcnn/            # CGCNN 核心实现
│   ├── main.py           # 训练入口
│   ├── predict.py        # 推理脚本
│   └── utils.py          # 工具函数
└── tests/                # 单元测试
```




