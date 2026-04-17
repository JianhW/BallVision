# BallShow ReID: 基于TransReID的篮球持球人身份重识别

<p align="center">
  <strong>TransReID + 篮球场景定制优化 | mAP 91.8% | Rank-1 94.6%</strong>
</p>

---

## 🏀 项目简介

本项目面向 **BallShow「球秀」赛题**，基于 [TransReID (ICCV 2021)](https://github.com/damo-cv/TransReID) 实现篮球比赛中持球人的跨镜头身份重识别。

与通用行人ReID不同，篮球场景面临三大特有挑战：

| 挑战 | 描述 |
|:---:|:---|
| **相似球衣区分** | 队友穿着完全相同的球衣，仅靠颜色无法区分，需依赖号码、体态、护具等细粒度特征 |
| **严重遮挡与姿态多变** | 持球人常处于多人包夹状态，运球、上篮等动作导致姿态剧烈变化 |
| **环境干扰** | 涵盖室内木地板（强反光）、室外塑胶场、夜间灯光等多种环境，且运动模糊严重 |

## 📊 实验结果

| 评估指标 | Baseline (TransReID) | **改进后 (Ours)** | 赛题达标要求 |
|:---:|:---:|:---:|:---:|
| **mAP** | 91.0% | **91.8%** | ≥ 91.5% |
| **Rank-1** | 93.6% | **94.6%** | ≥ 94% |
| **Rank-5** | — | **97.9%** | — |
| **Rank-10** | — | **99.2%** | — |

## 🔧 核心改进

在保留 TransReID 原有 **ViT-Base + JPM + SIE** 架构的基础上，以最小化修改原则进行针对性优化：

### 1. 篮球场景数据增强 V2 (`datasets/basketball_aug.py`)

针对相似球衣和遮挡问题，设计了三种互斥增强策略：
- **分区遮挡**：保护号码区域（上半身），仅对下半身和外围施加遮挡，模拟包夹场景
- **HSV颜色扰动**：对球衣区域做轻微色相/饱和度扰动，强迫模型学习颜色之外的判别特征
- **增强姿态变化**：翻转、旋转、透视变换模拟运球/上篮等动作姿态

### 2. 损失权重动态配置 (`loss/make_loss.py`)

支持通过配置文件灵活调整 ID Loss 与 Triplet Loss 的权重平衡：
```yaml
ID_LOSS_WEIGHT: 0.6       # 提升ID损失权重
TRIPLET_LOSS_WEIGHT: 1.4  # 降低三元组损失权重
```

### 3. 训练策略优化 (`solver/scheduler_factory.py`)

- **优化器**：SGD + Cosine Annealing（余弦退火学习率调度）
- **Warmup**：前 5 epoch 线性预热
- **训练轮数**：200 epoch，每 20 epoch 评估，每 10 epoch 保存检查点
- **断点续训**：支持通过 `SOLVER.START_EPOCH` 指定起始轮次

## 🚀 快速开始

### 环境依赖

```bash
conda create -n ballreid python=3.8
conda activate ballreid
pip install torch torchvision timm yacs termcolor
pip install -r requirements.txt
```

### 数据准备

将赛题提供的数据集解压至 `data/BallShow/`：

```
data/BallShow/
    ├── bounding_box_train/    # 训练集
    ├── bounding_box_test/     # 测试集Gallery
    └── query/                 # 测试集Query
```

### 预训练权重

下载 ViT-Base ImageNet 预训练权重：
[jx_vit_base_p16_224-80ecf9dd.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)

在配置文件中修改 `PRETRAIN_PATH` 为实际路径。

### 训练

```bash
python train.py --config_file configs/BallShow/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
```

### 测试

```bash
python test.py --config_file configs/BallShow/vit_transreid_stride.yml \
    MODEL.DEVICE_ID "('0')" \
    TEST.WEIGHT 'logs/BallShow_vit_transreid_stride/transformer_best.pth'
```

### 可视化测试

随机抽取 Query 进行检索并生成 HTML 报告：

```bash
python visual_test_improve.py --config_file configs/BallShow/vit_transreid_stride.yml \
    --num_rounds 50 --top_k 5 \
    MODEL.DEVICE_ID "('0')" \
    TEST.WEIGHT 'logs/BallShow_vit_transreid_stride/transformer_best.pth'
```

## 📁 项目结构

```
TransReID/
├── configs/BallShow/
│   └── vit_transreid_stride.yml     # 主训练配置
├── datasets/
│   ├── basketball_aug.py            # 篮球场景数据增强 V2
│   └── make_dataloader.py           # 数据加载（集成增强Pipeline）
├── loss/
│   └── make_loss.py                 # 可配置损失函数
├── model/
│   ├── make_model.py                # ViT + JPM + SIE 模型
│   └── transformer.py               # Transformer模块
├── solver/
│   ├── build.py                     # 优化器构建
│   └── scheduler_factory.py         # 学习率调度器
├── processor/
│   └── processor.py                 # 训练/测试流程（含断点续训）
├── train.py                         # 训练入口
├── test.py                          # 测试入口
└── visual_test_improve.py           # 可视化检索测试
```

## ⚙️ 关键配置参数

```yaml
# configs/BallShow/vit_transreid_stride.yml
MODEL:
  STRIDE_SIZE: [12, 12]        # Patch stride（保留更多细节）
  JPM: True                    # Joints Pooling Module
  SIE_CAMERA: True             # Side Information Embedding
  SIE_COE: 3.0
  ID_LOSS_WEIGHT: 0.6
  TRIPLET_LOSS_WEIGHT: 1.4

INPUT:
  BASKETBALL_AUG_PROB: 0.35    # 篮球增强总概率
  OCCLUSION_PROB: 0.20         # 分区遮挡概率
  POSE_CHANGE_PROB: 0.15       # 姿态变化概率
  COLOR_JITTER_PROB: 0.15      # HSV颜色扰动概率

SOLVER:
  MAX_EPOCHS: 200
  OPTIMIZER_NAME: 'SGD'
  WARMUP_METHOD: 'cosine'
  WARMUP_EPOCHS: 5
  START_EPOCH: 0               # 修改此值可断点续训
```

## 📄 引用

本项目基于以下论文实现：

```bibtex
@InProceedings{He_2021_ICCV,
    author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
    title     = {TransReID: Transformer-Based Object Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15013-15022}
}
```

---

<p align="center">
  Built with <strong>TransReID</strong> · Optimized for <strong>BallShow</strong> Basketball Scene
</p>
