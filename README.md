# 高镍层状氧化物智能性质预测与缺陷工程系统 —— 基于晶体图卷积神经网络的多尺度计算平台
# Intelligent Property Prediction and Defect Engineering System for Ni-Rich Layered Oxides —— Multi-Scale Computational Platform Based on Crystal Graph Convolutional Neural Networks

---

## 1. 项目标题 (Project Title)
高镍层状氧化物晶体图卷积神经网络智能预测与缺陷筛选系统  
Intelligent Crystal Graph Convolutional Neural Network System for Ni-Rich Layered Oxide Property Prediction and Defect Screening

## 2. 项目简介 (Project Description)
本项目构建了一套基于 Crystal Graph Convolutional Neural Networks (CGCNN) 的智能计算平台，专门针对 NCM811、NCA 等高镍层状氧化物材料的结构-性质关系进行深度建模。系统集成了多尺度图卷积、注意力机制、不确定度分解等先进算法，能够在毫秒级时间尺度内实现近 DFT 精度的形成能、带隙、弹性模量、费米能级等关键物性的预测。通过贝叶斯优化驱动的主动学习循环，结合 Epistemic 与 Aleatoric 不确定度的精确分解，系统能够智能识别化学空间中的高价值区域，实现缺陷结构的高效筛选与优化。该平台不仅提供了原子级别的可解释性分析，还支持与 ASE 等计算化学软件的深度集成，可作为机器学习势能参与结构优化与分子动力学模拟，为高能量密度电池材料的理性设计提供强有力的计算支撑。

This project establishes an intelligent computational platform based on Crystal Graph Convolutional Neural Networks (CGCNN), specifically designed for deep modeling of structure-property relationships in Ni-rich layered oxides such as NCM811 and NCA. The system integrates advanced algorithms including multi-scale graph convolution, attention mechanisms, and uncertainty decomposition, achieving near-DFT accuracy predictions for critical properties including formation energy, band gap, elastic moduli, and Fermi level within millisecond time scales. Through Bayesian optimization-driven active learning cycles, combined with precise decomposition of epistemic and aleatoric uncertainties, the system intelligently identifies high-value regions in chemical space, enabling efficient defect structure screening and optimization. The platform not only provides atom-level interpretability analysis but also supports deep integration with computational chemistry software such as ASE, serving as a machine learning potential for structure optimization and molecular dynamics simulations, providing powerful computational support for rational design of high-energy-density battery materials.

## 3. 功能与亮点 (Features & Highlights)
| Feature | Description |
|---------|-------------|
| Multi-property prediction | Formation energy, band gap, elastic moduli, Fermi level |
| Uncertainty decomposition | Epistemic & Aleatoric components |
| Active learning loop | Bayesian optimization with MC-Dropout ranking |
| Interpretability | Atom-/bond-level contribution maps |
| Multi-scale GCN & Attention | Captures long-range interactions |
| ASE interface | Acts as ML potential for structure relaxation |

## 4. 技术栈 (Tech Stack)
- Python ≥ 3.9, PyTorch ≥ 1.10  
- PyMatGen, ASE  
- scikit-learn, scikit-optimize  
- Dependencies: `configs/environment.yml`

## 5. 安装与配置 (Installation & Setup)
```bash
# Conda
conda env create -f configs/environment.yml
conda activate cgcnn

# Manual setup
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
- **Data**: `data/` directory contains cleaned CIF files, labels (`id_prop.csv`) and analysis results.
- **Benchmarks**:
  | Task | Metric | Value |
  |------|--------|-------|
  | Formation energy | MAE | < 0.10 eV/atom |
  | Band gap | MAE | ≈ 0.32 eV |
  | Defect classification | Accuracy | > 90 % |
- Reproduction scripts: `scripts/evaluation/`, detailed results: `data/analysis/`.

## 8. 贡献 (Contributing)
1. Fork repository and create branch.  
2. Follow PEP8, ensure `pytest` passes before commit.  
3. Describe changes and test results via Pull Request.

## 9. 许可证 (License)
MIT License – see `LICENSE` for details.

## 10. 作者 (Author)
LunaZhang

## 11. 未来工作 (Future Work)
- Support for more Ni-rich systems (NCMx90, NCA-F)  
- Self-supervised pre-training to reduce annotation requirements  
- Cloud inference and visualization dashboard  
- CI/CD & Benchmark automation

## 12. 项目结构 (Project Structure)
```text
├── configs/              # Environment and dependencies
├── data/                 # Datasets and analysis
├── docs/README.md        # Documentation
├── models/               # Training checkpoints
├── pre-trained/          # Published model weights
├── scripts/              # Data processing & evaluation scripts
├── src/                  # Main source code
│   ├── cgcnn/            # CGCNN core implementation
│   ├── main.py           # Training entry
│   ├── predict.py        # Inference script
│   └── utils.py          # Utility functions
└── tests/                # Unit tests
```




