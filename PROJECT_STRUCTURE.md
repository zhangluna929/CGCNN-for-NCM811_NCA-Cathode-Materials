# NCM811/NCA CGCNN 项目结构

本项目已按照标准的机器学习项目结构进行重新组织，以提高代码的可维护性和可读性。

## 目录结构

```
NCM811_NCA_Cgcnn_Project-main/
├── src/                          # 源代码目录
│   ├── cgcnn/                    # CGCNN核心模块
│   │   ├── integrations/         # 第三方集成接口
│   │   ├── model.py             # 基础CGCNN模型
│   │   ├── enhanced_model.py    # 增强版CGCNN模型
│   │   ├── data.py              # 数据处理模块
│   │   ├── features.py          # 特征工程
│   │   ├── uncertainty.py       # 不确定性量化
│   │   ├── advanced_uncertainty.py # 高级不确定性分析
│   │   ├── smart_active_learning.py # 智能主动学习
│   │   ├── generalization_framework.py # 泛化框架
│   │   ├── data_balance_handler.py # 数据平衡处理
│   │   ├── robust_training.py   # 鲁棒训练
│   │   ├── comprehensive_interpretability.py # 综合可解释性
│   │   ├── multi_scale_framework.py # 多尺度框架
│   │   ├── high_ni_analysis.py  # 高镍材料分析
│   │   ├── precision_analysis.py # 精度分析
│   │   ├── physics.py           # 物理约束
│   │   ├── multimodal.py        # 多模态学习
│   │   └── interpretability.py  # 可解释性分析
│   ├── main.py                  # 主训练脚本
│   ├── predict.py               # 预测脚本
│   ├── active_learning.py       # 主动学习脚本
│   └── utils.py                 # 工具函数
│
├── data/                         # 数据目录
│   ├── raw/                     # 原始数据
│   │   └── atom_init.json       # 原子初始化参数
│   ├── processed/               # 处理后的数据
│   │   ├── id_prop.csv          # 基础属性数据
│   │   └── id_prop_complete.csv # 完整属性数据
│   ├── cif/                     # CIF结构文件
│   │   ├── NCM811/              # NCM811材料结构
│   │   ├── NCA/                 # NCA材料结构
│   │   └── candidates/          # 候选材料结构
│   ├── dft/                     # DFT计算结果
│   │   └── dft_labels_full.csv  # 完整DFT标签
│   ├── analysis/                # 分析结果
│   │   ├── training_history.json # 训练历史
│   │   ├── model_benchmarks.json # 模型基准测试
│   │   └── uncertainty_analysis.csv # 不确定性分析
│   ├── performance/             # 性能评估
│   │   ├── performance_metrics.csv # 性能指标
│   │   ├── experimental_validation.csv # 实验验证
│   │   └── synthesis_optimization.csv # 合成优化
│   └── materials_db/            # 材料数据库
│       └── materials_database.json # 材料数据库
│
├── models/                      # 模型目录
│   ├── pre-trained/             # 预训练模型
│   │   ├── formation-energy-per-atom.pth.tar
│   │   ├── band-gap.pth.tar
│   │   ├── bulk-moduli.pth.tar
│   │   ├── shear-moduli.pth.tar
│   │   ├── poisson-ratio.pth.tar
│   │   ├── efermi.pth.tar
│   │   ├── final-energy-per-atom.pth.tar
│   │   └── semi-metal-classification.pth.tar
│   └── checkpoints/             # 训练检查点
│
├── scripts/                     # 脚本目录
│   ├── data_processing/         # 数据处理脚本
│   │   ├── make_li_vac.py       # 锂空位生成
│   │   └── build_clean_cif.py   # CIF文件清理
│   ├── training/                # 训练脚本
│   └── evaluation/              # 评估脚本
│
├── tests/                       # 测试目录
│   └── test_imports.py          # 导入测试
│
├── docs/                        # 文档目录
│   ├── README.md                # 项目说明
│   ├── PROJECT_SUMMARY.md       # 项目总结
│   └── LICENSE                  # 许可证
│
├── examples/                    # 示例目录
├── log/                         # 日志目录
├── requirements.txt             # Python依赖
├── environment.yml              # Conda环境配置
└── .gitignore                   # Git忽略文件
```

## 主要改进

1. **模块化组织**: 将所有源代码整理到`src/`目录下，提高代码组织性
2. **数据分类**: 按数据类型和处理阶段分类存储，便于管理和访问
3. **模型管理**: 预训练模型和检查点分开存储，支持版本管理
4. **脚本分离**: 按功能分类存储脚本文件，提高可维护性
5. **文档集中**: 所有文档集中在`docs/`目录，便于查阅
6. **配置文件**: 环境配置文件放在根目录，便于环境搭建

## 使用说明

- 训练模型: `python src/main.py`
- 预测: `python src/predict.py`
- 主动学习: `python src/active_learning.py`
- 运行测试: `python tests/test_imports.py`

这种结构使项目更加专业化，便于团队协作和代码维护。 