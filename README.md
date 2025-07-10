# Li‐vacancy CGCNN Demo (NCM811 & NCA)


```bash
conda env create -f environment.yml
conda activate cgcnn_2023

run_demo.bat



数据: NCM811 / NCA CIF |
缺位生成: `make_li_vac.py` + pymatgen 删除首个 Li |
补丁: pymatgen  `species.elements[0].Z` |

---

## 结果

```
Epoch: [0]  Loss 0.15  MAE 0.12
...
Epoch: [28] Loss 0.03  MAE 0.03
Best MAE on test set = 0.06
```

> 样本数仅作演示，真实科研需扩充至 ≥10³ 结构并做 k‑fold CV。

---

##  扩展

- **多材料**：LLZO、LGPS、β‑Li₃PS₄ 等固态电解质  
- **多缺陷**：多 Li vacancy / 杂质掺杂  
- **目标切换**：迁移能垒、体相电压、相对稳定性  
- **Explainability**：Grad‑CAM on graph edges

---

## 引用

- T. Xie & J. C. Grossman, “Crystal Graph Convolutional Neural Networks…”, *Phys. Rev. Lett.* **120**, 145301 (2018).  
- S. Ong *et al.*, *pymatgen: Python Materials Genomics*, *Comp. Mater. Sci.* **68** 314–319 (2013).

---

© 2025  LunaZhang — MIT License