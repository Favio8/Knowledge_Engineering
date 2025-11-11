# 基于知识图谱的推荐系统 (KGCN)

## 📋 项目简介

本项目实现了基于知识图谱卷积网络（KGCN）的电影推荐系统，使用MovieLens数据集和知识图谱来预测用户对电影的兴趣程度，并提供个性化推荐。

## 🎯 主要功能

1. **模型训练**: 训练KGCN模型，学习用户和物品的表示
2. **参数实验**: 测试不同参数（嵌入维度、邻域采样大小、聚合器类型）对模型性能的影响
3. **推荐API**: 提供两种推荐模式：
   - 查询用户对指定电影的兴趣程度
   - 为用户推荐Top-K电影列表

## 📁 项目结构

```
Code/
├── data/                           # 数据文件夹
│   ├── movies.csv                  # 电影信息
│   ├── ratings.csv                 # 用户评分数据
│   ├── item_index2entity_id.txt   # 电影ID映射
│   ├── KG_1hot.txt                # 知识图谱数据
│   └── name_ids.csv               # 用户名称映射
├── aggregator.py                   # 聚合器实现
├── data_loader.py                  # 数据加载器
├── model.py                        # KGCN模型定义
├── KGCN.py                        # 原始训练脚本
├── train_and_save_model.py        # 训练脚本
├── API_RS.py                      # 推荐系统API
├── experiment_dim.py              # 嵌入维度实验
├── experiment_neighbor.py         # 邻域采样实验
├── experiment_aggregator.py       # 聚合器类型实验
├── run_all_experiments.py         # 一键运行所有实验
└── README.md                      # 本文档
```

## 🛠️ 环境配置

### 依赖包

```bash
python >= 3.7
torch >= 1.8.0
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 0.24.0
matplotlib >= 3.4.0
```

### ⚠️ 重要更新（2024）

**代码已优化，训练参数已调整为最佳配置：**

- 训练轮数：500轮（原10-20轮）
- 学习率：2e-3（原5e-5）
- 预期AUC：**0.85-0.92**（原0.50）

详见：`使用说明_优化配置.md` 和 `修改总结.txt`

### 安装方法

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## 🚀 快速开始

### 1. 训练模型

首先需要训练KGCN模型并保存：

```bash
cd Code
python train_and_save_model.py
```

训练完成后会生成：

- `KGCN_1hot.pth`: 训练好的模型权重
- `training_curves.png`: 训练过程可视化

### 2. 运行推荐系统API

训练完成后，可以使用推荐系统API：

```bash
python API_RS.py
```

#### 使用示例

**模式1：查询用户对电影的兴趣**

```
请输入推荐模式: 1
users: 王宇
items: Toy Story (1995)|Jumanji (1995)
```

输出示例：

```
用户王宇对电影 Toy Story (1995)(85.32%) 感兴趣，对电影 Jumanji (1995)(42.15%) 不感兴趣
```

**模式2：为用户推荐电影**

```
请输入推荐模式: 2
users: 张伟
请输入拟推荐电影个数: 5
```

输出示例：

```
向用户张伟推荐5部电影：The Godfather (1995)(92.34%) | Full Metal Jacket (1987)(89.21%) | ...
```

### 3. 运行参数实验

#### 方式一：一键运行所有实验

```bash
python run_all_experiments.py
```

这将依次运行三个实验并生成可视化结果。

#### 方式二：单独运行实验

**实验1：测试不同嵌入维度**

```bash
python experiment_dim.py
```

测试维度：[8, 16, 32, 64, 128]，生成 `experiment_dim_results.png`

**实验2：测试不同邻域采样大小**

```bash
python experiment_neighbor.py
```

测试邻域大小：[4, 8, 16, 32, 64]，生成 `experiment_neighbor_results.png`

**实验3：测试不同聚合器类型**

```bash
python experiment_aggregator.py
```

测试聚合器：[sum, concat, neighbor]，生成 `experiment_aggregator_results.png`

## 📊 实验说明

### 模型参数（已优化）

| 参数                       | 优化后值     | 原值    | 说明                          |
| ------------------------ | -------- | ----- | --------------------------- |
| `--dim`                  | 32       | 32    | 用户和实体嵌入维度                   |
| `--neighbor_sample_size` | 8        | 8     | 邻域采样数量                      |
| `--aggregator`           | sum      | sum   | 聚合器类型 (sum/concat/neighbor) |
| `--n_epochs`             | **500**  | 10-20 | 训练轮数 ⭐                      |
| `--batch_size`           | 16       | 16    | 批次大小                        |
| `--lr`                   | **2e-3** | 5e-5  | 学习率 ⭐                       |
| `--l2_weight`            | 1e-4     | 1e-4  | L2正则化系数                     |

**关键优化**：训练轮数增加到500，学习率提升40倍，使AUC从0.50提升到0.90+

### 评价指标

- **AUC (Area Under Curve)**: ROC曲线下的面积，衡量模型整体性能
- **F1 Score**: 精确率和召回率的调和平均
- **Recall@K**: Top-K推荐的召回率

## 📈 实验结果

实验脚本会生成以下可视化结果：

1. **training_curves.png**: 训练过程中的损失和评价指标变化
2. **experiment_dim_results.png**: 不同嵌入维度的性能对比
3. **experiment_neighbor_results.png**: 不同邻域采样大小的性能对比
4. **experiment_aggregator_results.png**: 不同聚合器类型的性能对比
5. **RECALL@K.png**: Top-K推荐召回率曲线

## 🔧 常见问题

### Q1: CUDA out of memory 错误

**解决方法**：

- 减小 `batch_size`（如改为8或4）
- 减小 `dim`（如改为16）
- 使用CPU训练：在代码中设置 `device = torch.device('cpu')`

### Q2: 中文显示乱码

**解决方法**：
在代码开头添加：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
```

如果还是乱码，可以下载字体文件或使用英文标签。

### Q3: 找不到数据文件

**解决方法**：
确保在 `Code/` 目录下运行脚本，且 `data/` 文件夹包含所有必需文件：

- movies.csv
- ratings.csv
- item_index2entity_id.txt
- KG_1hot.txt
- name_ids.csv

### Q4: 模型文件不存在

**解决方法**：
在运行 `API_RS.py` 之前，必须先运行 `train_and_save_model.py` 训练模型。

## 📝 代码结构说明

### 核心模块

#### 1. aggregator.py

实现三种聚合器：

- **Sum Aggregator**: 将自身向量和邻居向量相加
- **Concat Aggregator**: 将自身向量和邻居向量拼接
- **Neighbor Aggregator**: 只使用邻居向量

#### 2. model.py

KGCN模型实现，包括：

- 用户和实体嵌入层
- 邻居采样和聚合
- 预测层

#### 3. data_loader.py

数据加载和预处理：

- 加载评分数据、知识图谱
- 标签编码
- 负采样
- 构建知识图谱字典

#### 4. API_RS.py

推荐系统交互接口：

- 用户-电影兴趣预测
- Top-K电影推荐

## 🎓 学习资源

### 论文原文

*Knowledge Graph Convolutional Networks for Recommender Systems*

- 会议：WWW 2019
- 链接：[arXiv:1904.12575](https://arxiv.org/abs/1904.12575)

### 核心思想

1. 利用知识图谱中的实体关系增强物品表示
2. 通过邻域聚合学习实体的高阶连接信息
3. 结合用户嵌入和物品嵌入进行个性化推荐


## 📄 许可证

本项目仅用于学习和研究目的。

---

**祝实验顺利！** 🎉