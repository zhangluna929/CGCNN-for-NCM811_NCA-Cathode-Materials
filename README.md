# Crystal Graph Convolutional Neural Networks (CGCNN) 多任务版本 | Multi-Task Version

> **ZH**：基于经典 CGCNN 的回归 / 分类 / 多任务学习框架，支持材料形成能预测与缺陷类型分类，并附带 Li vacancy 数据集与脚本。  
> **EN**: A multi-task extension of the classic CGCNN for simultaneous regression (e.g. formation energy) and classification (e.g. defect type). Includes a Li-vacancy dataset and helper scripts.

---

## 目录 | Table of Contents

1. [项目简介 | Project Overview](#项目简介--project-overview)  
2. [文件结构 | Directory Structure](#文件结构--directory-structure)  
3. [环境依赖 | Requirements](#环境依赖--requirements)  
4. [数据格式 | Data Format](#数据格式--data-format)  
5. [核心脚本 | Key Scripts](#核心脚本--key-scripts)  
6. [快速开始 | Quick Start](#快速开始--quick-start)  
7. [Vacancy 数据集与脚本 | Vacancy Dataset & Script](#vacancy-数据集与脚本--vacancy-dataset--script)  
8. [预训练模型 | Pre-trained Models](#预训练模型--pre-trained-models)  
9. [许可证 | License](#许可证--license)

---

## 项目简介 | Project Overview

**ZH**：
- 支持 `--task regression`、`classification`、`multi` 三种任务模式。
- `CrystalGraphConvNetMulti`（`cgcnn/model.py`）输出 `(E_form, defect_logits)`。
- `cgcnn/data.py` 改进为同时返回回归 + 分类标签。
- `main.py` 中通过 `--cls-weight` 控制分类损失权重。

**EN**:
- Three task modes are available: `regression`, `classification`, and `multi`.
- `CrystalGraphConvNetMulti` (`cgcnn/model.py`) outputs `(E_form, defect_logits)` for multi-task training.
- `cgcnn/data.py` returns both regression & classification labels.
- Use `--cls-weight` in `main.py` to balance classification loss.

---

---

## 环境依赖 | Requirements

| Package | Version |
| ------- | ------- |
| Python  | ≥3.8 |
| PyTorch | ≥1.10 |
| scikit-learn | latest |
| pymatgen | latest |

```bash
# Conda installation (example)
conda create -n cgcnn python=3.9 pytorch torchvision scikit-learn pymatgen -c pytorch -c conda-forge
conda activate cgcnn
```

---

## 数据格式 | Data Format

`id_prop.csv` (**3 columns** | 三列)：

| cif_id | formation_energy | defect_label |
| ------ | :--------------: | :-----------: |
| NCA_LiVac0 | 3.800 | 1 |

- **formation_energy**: float, used for regression.  
- **defect_label**: int (0 / 1), used for classification.

Additional files:  
1. `atom_init.json` – element embeddings.  
2. `*.cif` – crystal structures with filenames matching `cif_id`.

---

## 核心脚本 | Key Scripts

| Script | ZH 说明 | EN Description |
| ------ | ------- | -------------- |
| `main.py` | 训练模型 (`--task` 选择任务, `--cls-weight` 分类损失权重) | Train models (`--task`, `--cls-weight`) |
| `predict.py` | 加载权重并预测 | Load weights and run inference |
| `vacancy_data/make_li_vac.py` | 在正极材料中随机去除 Li 制造空位并输出 CIF | Randomly removes Li to generate vacancy structures |

---

## 快速开始 | Quick Start

### 单任务训练 | Single-Task Training

```bash
# Regression (formation energy)
python main.py data/sample-regression --task regression --epochs 30

# Classification (metal / semiconductor)
python main.py data/sample-classification --task classification --epochs 30
```

### 多任务训练 | Multi-Task Training

```bash
python main.py vacancy_data --task multi --cls-weight 0.5 --epochs 50 \
  --batch-size 128 --lr 0.001
```

Outputs:  
* `checkpoint.pth.tar` – last epoch  
* `model_best.pth.tar` – best validation score  
* `test_results.csv` – ID, target, prediction

### 预测 | Inference

```bash
python predict.py model_best.pth.tar vacancy_data
```

`test_results.csv` will contain formation energy predictions or classification probabilities.

---

## Vacancy 数据集与脚本 | Vacancy Dataset & Script

| File | ZH 描述 | EN Description |
| ---- | ------- | -------------- |
| `NCA_LiVac0.cif` | NCA 原始结构 | Pristine NCA structure |
| `NCA_LiVac0_aug*.cif` | 三种 Li 空位增广 | Augmented Li-vacancy structures |
| `make_li_vac.py` | 生成 Li 空位 | Generate Li-vacancy structures |

Usage / 用法：

```bash
python vacancy_data/make_li_vac.py NCA_LiVac0.cif --num_aug 3 --seed 42
```

---

## 预训练模型 | Pre-trained Models

Official single-task weights are provided in `pre-trained/`. You can fine-tune them in multi-task mode by adding a classification head.

---

## 许可证 | License

MIT License. See `LICENSE` for details.



