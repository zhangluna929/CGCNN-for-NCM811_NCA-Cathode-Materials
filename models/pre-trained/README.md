# Pre-trained CGCNN Models  
# 预训练 CGCNN 模型

> Rapid property prediction & active-learning booster for **high-Ni layered cathodes** (NCM811, NCA)
> 
> 针对 **高镍层状正极**（NCM811、NCA）的快速物性预测及主动学习加速器

---

## 1. What’s inside? / 内容概览

English | 中文
--- | ---
Formation energy, band-gap, elastic moduli, Fermi level & more – nine models trained on Materials Project (MP) are provided.<br/>They give quick, DFT-level estimates for any CIF you feed in. | 本目录提供 9 个基于 Materials Project 训练的模型，可对任意 CIF 结构秒级输出形成能、带隙、弹性模量、费米能级等近似 DFT 结果。
All files end with `.pth.tar`; load them with `predict.predict()` or via the CLI (`python predict.py …`). | 所有模型均为 `.pth.tar`，可通过 `predict.predict()` 函数或命令行 (`python predict.py …`) 调用。


### Supported properties / 支持的物性

File | Property (EN) | 属性 (CN) | Units
---- | ------------- | --------- | -----
`formation-energy-per-atom` | Formation energy | 形成能（每原子） | eV/atom
`final-energy-per-atom`    | Total energy     | 总能量（每原子） | eV/atom
`band-gap`                 | Band gap         | 带隙            | eV
`efermi`                   | Fermi energy     | 费米能级         | eV
`bulk-moduli`              | Bulk modulus     | 体模量(log)     | log(GPa)
`shear-moduli`             | Shear modulus    | 剪切模量(log)   | log(GPa)
`poisson-ratio`            | Poisson ratio    | 泊松比           | —
`semi-metal-classification`| Semi-metal?      | 半金属分类       | 0/1

## 2. Why they matter for NCM811 / NCA

1. **Stability check** – Formation energy helps rule out Li-vacancy or dopant configurations that are obviously unstable.
2. **Electronic conductivity** – Band gap & Fermi level hint at electronic transport capability after cycling.
3. **Mechanical integrity** – Elastic moduli & Poisson ratio correlate with crack initiation during volume change.
4. **Rapid screening** – Evaluate thousands of vacancy/supercell variants in minutes before expensive DFT.

简体中文摘要：
1. **热力学筛选**：形成能快速甄别不稳定缺陷/掺杂构型；
2. **电导评估**：带隙与费米能级预判电子导电性；
3. **结构可靠性**：弹性模量&泊松比提示循环开裂风险；
4. **计算加速**：在送入 VASP 之前，用秒级预测从上千方案中挑选 Top-N。

## 3. Quick start / 快速上手

```python
from predict import predict

res = predict(
    'pre-trained/formation-energy-per-atom.pth.tar',
    cif_dir='vacancy_data',   # contains NCM811_LiVac0.cif etc.
    n_dropout=25              # uncertainty estimate
)
for cif_id, (mu, sigma) in res.items():
    print(f'{cif_id:20s}  {mu:.3f} ± {sigma:.3f} eV/atom')
```

Command-line / 命令行：
```bash
python predict.py pre-trained/band-gap.pth.tar vacancy_data --n-dropout 25
```

## 4. Integrate with active learning / 主动学习集成

1. Edit your search space & mutation rules in `active_learning.py`.
2. Set `MODEL_CKPT` to the property you want to optimise (e.g. formation energy).
3. Run `python active_learning.py` – Bayesian Optimisation will generate candidates, call CGCNN for fast scoring, and (optionally) trigger DFT refinement every *K* steps.

> Tip: replace the placeholder `run_dft_simulation()` with your own VASP / QE submission script, and append the converged value to `id_prop.csv` for **on-the-fly fine-tuning**.

## 5. Caveats / 使用须知

* Trained on MP → works best for ICSD-like, fully relaxed structures. Highly distorted or hypothetical phases may incur larger error.
* Mean MAE ~0.04–0.10 eV/atom for formation energy on MP hold-out; other properties see the original paper (Xie & Grossman 2018) for metrics.
* Always validate key candidates with high-fidelity DFT or experiment before publication.

> 模型训练于 Materials Project 数据，适用于接近实验结构的层状氧化物；强畸变或非氧化物体系误差可能增大。最终结果请以 DFT/实验复核为准。

## 6. Citation / 引用

If these models assist your work, please cite:

* **Model**: T. Xie & J. C. Grossman, *Phys. Rev. Lett.* **120**, 145301 (2018).
* **Data**: A. Jain *et al.*, *APL Materials* **1**, 011002 (2013).

若使用本仓库的预训练模型，请同时引用以上数据与模型文献。

---

*Author / 作者*: lunazhang  
*Date / 日期*: 2023

