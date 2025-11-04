# MIT-BIH心电图心律失常分类项目

基于深度学习的MIT-BIH心律失常数据库心电图自动分类系统。

## 项目简介

本项目实现了对MIT-BIH心律失常数据库的完整处理流程，包括：
- 数据加载和预处理
- 深度学习模型构建（CNN和CNN-LSTM）
- 模型训练和评估
- 结果可视化

## 项目结构

```
.
├── data_loader.py          # 数据加载模块
├── preprocess.py            # 数据预处理模块
├── model.py                # 深度学习模型定义
├── train.py                # 模型训练脚本
├── visualize.py            # 数据可视化模块
├── example_usage.py        # 使用示例
├── requirements.txt        # 依赖包列表
├── info.md                 # 详细说明文档（心电图基础知识）
├── README.md              # 本文件
└── heart_data/            # MIT-BIH数据集文件夹
    ├── *.dat              # 信号数据文件
    ├── *.hea              # 头文件
    ├── *.atr              # 注释文件
    └── RECORDS            # 记录列表
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行示例

查看数据加载和可视化示例：
```bash
python example_usage.py
```

### 3. 训练模型

开始训练深度学习模型：
```bash
python train.py
```

训练完成后会生成：
- `best_model.pth`: 最佳模型权重
- `training_curves.png`: 训练过程曲线
- `confusion_matrix.png`: 混淆矩阵

## 主要功能

### 数据加载 (`data_loader.py`)
- 自动读取MIT-BIH数据集的WFDB格式文件
- 支持批量加载多条记录
- 提取R波位置和心跳类型标注

### 数据预处理 (`preprocess.py`)
- 小波去噪和带通滤波
- 基于R波位置的心拍分割
- 数据标准化

### 模型架构 (`model.py`)
- **ECGCNN**: 纯CNN架构，适合快速训练
- **ECGCNNLSTM**: CNN-LSTM混合架构，捕获时序依赖

### 训练脚本 (`train.py`)
- 完整的训练流程
- 自动数据集划分（训练/验证/测试）
- 模型评估和指标计算
- 训练过程可视化

### 可视化 (`visualize.py`)
- ECG信号可视化
- R波标记
- 心拍示例展示
- 类别分布图
- 训练曲线

## 配置参数

在`train.py`的`main()`函数中可以修改以下参数：

```python
config = {
    'data_dir': 'heart_data',           # 数据目录
    'max_records': None,                 # 最大加载记录数（None=全部）
    'target_classes': ['N', 'L', 'R', 'A', 'V'],  # 目标分类类别
    'beat_length': 360,                  # 心拍长度（采样点）
    'batch_size': 64,                    # 批次大小
    'num_epochs': 50,                    # 训练轮数
    'learning_rate': 0.001,               # 学习率
    'model_type': 'cnn',                 # 模型类型：'cnn' 或 'cnn_lstm'
    'test_size': 0.2,                    # 测试集比例
    'val_size': 0.1,                     # 验证集比例
}
```

## 心跳类型说明

| 符号 | 名称 | 说明 |
|------|------|------|
| N | 正常心跳 | 正常窦性心律 |
| L | 左束支传导阻滞 | 左束支传导异常 |
| R | 右束支传导阻滞 | 右束支传导异常 |
| A | 房性早搏 | 心房提前收缩 |
| V | 室性早搏 | 心室提前收缩 |

更多类型说明请查看 `info.md`。

## 使用示例

### 加载单条记录

```python
from data_loader import MITBIHLoader

loader = MITBIHLoader("heart_data")
signal, record_info, ann_info = loader.load_record("100")
```

### 预处理数据

```python
from preprocess import ECGBeatPreprocessor

preprocessor = ECGBeatPreprocessor(fs=360, beat_length=360)
beats, valid_indices = preprocessor.process_record(
    signal, ann_info['sample'], channel=0, denoise=True
)
```

### 可视化

```python
from visualize import ECGVisualizer

visualizer = ECGVisualizer(fs=360)
visualizer.plot_signal(signal[:, 0], title="ECG信号", duration=10.0)
visualizer.plot_signal_with_rpeaks(signal[:, 0], ann_info['sample'], duration=10.0)
```

## 详细文档

完整的心电图基础知识、数据集说明、预处理方法和模型架构说明请查看 **`info.md`**。

## 系统要求

- Python 3.8+
- PyTorch 2.0+
- 建议使用GPU加速训练（可选，CPU也可运行）

## 注意事项

1. 首次运行需要确保`heart_data`文件夹包含完整的MIT-BIH数据集
2. 训练时间取决于数据量和硬件配置，建议使用GPU
3. 如果遇到内存不足，可以减少`batch_size`或`max_records`
4. 数据不平衡问题可能需要使用类别权重或采样策略

## 许可证

本项目仅供学习和研究使用。

## 参考资料

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- WFDB Python Package: https://github.com/MIT-LCP/wfdb-python

## 作者

机器学习工程师

---

如有问题或建议，欢迎提出Issue！

