# 医疗知识图谱构建项目

## 项目简介

本项目是一个基于Neo4j图数据库的**医疗知识图谱构建系统**，旨在将医疗领域的结构化数据转换为可查询、可推理的知识图谱。该系统能够有效地组织和关联医疗实体（疾病、药物、症状、检查、科室等），为医疗信息检索、智能问答、临床决策支持等应用提供基础。

## 项目特色

- 📊 **丰富的数据维度**：涵盖疾病、药物、症状、检查、科室、食物、生产厂商等7类核心医疗实体
- 🔗 **完整的关系网络**：构建了10多种不同类型的实体关系，形成完整的医疗知识网络
- 🚀 **自动化构建**：提供一键式知识图谱构建和数据导出功能
- 💾 **数据规模**：基于150条疾病数据构建，包含数千个医疗实体和关系
- 🔍 **便捷查询**：支持Neo4j的Cypher查询语言进行复杂数据检索

## 数据结构

### 实体类型 (7类)

| 实体类型 | 描述 | 示例 |
|---------|------|------|
| **Disease** (疾病) | 疾病信息，包含描述、病因、预防等详细信息 | 肺泡蛋白质沉积症、百日咳 |
| **Drug** (药物) | 治疗药物 | 乳酸左氧氟沙星片、阿奇霉素片 |
| **Symptom** (症状) | 疾病症状表现 | 紫绀、胸痛、呼吸困难 |
| **Check** (检查) | 诊断检查项目 | 胸部CT检查、血常规 |
| **Department** (科室) | 医院科室 | 内科、呼吸内科、儿科 |
| **Food** (食物) | 饮食相关 | 推荐食谱、宜吃/忌吃食物 |
| **Producer** (生产厂商) | 药品生产厂商 | 药品生产企业信息 |

### 关系类型 (10类)

| 关系类型 | 起始实体 | 目标实体 | 描述 |
|---------|---------|---------|------|
| `recommand_eat` | Disease | Food | 疾病推荐食谱 |
| `no_eat` | Disease | Food | 疾病忌吃食物 |
| `do_eat` | Disease | Food | 疾病宜吃食物 |
| `belongs_to` | Department | Department | 科室归属关系 |
| `common_drug` | Disease | Drug | 疾病常用药品 |
| `drugs_of` | Producer | Drug | 厂商生产药品 |
| `recommand_drug` | Disease | Drug | 疾病好评药品 |
| `need_check` | Disease | Check | 疾病诊断检查 |
| `has_symptom` | Disease | Symptom | 疾病症状关系 |
| `acompany_with` | Disease | Disease | 疾病并发症关系 |

## 项目结构

```
01-medicalGraph/
├── data/
│   └── medical.json              # 原始医疗数据（150条疾病记录）
├── dict/                         # 实体字典文件
│   ├── check.txt                 # 检查项目列表
│   ├── department.txt            # 科室列表
│   ├── disease.txt               # 疾病列表
│   ├── drug.txt                  # 药物列表
│   ├── food.txt                  # 食物列表
│   ├── producer.txt              # 生产厂商列表
│   ├── symptom.txt               # 症状列表
│   └── Query/                    # 从Neo4j数据库查询导出的实体列表
│       ├── check.txt
│       ├── department.txt
│       ├── disease.txt
│       ├── drug.txt
│       ├── food.txt
│       ├── producer.txt
│       └── symptoms.txt
├── build_medicalgraph.py         # 知识图谱构建脚本
├── get_dict.py                   # 数据字典导出脚本
└── README.md                     # 项目说明文档
```

## 技术栈

- **图数据库**: Neo4j 4.x
- **编程语言**: Python 3.x
- **核心依赖**: 
  - `py2neo`: Neo4j Python客户端
  - `json`: JSON数据处理

## 安装和使用

### 环境要求

1. **Neo4j数据库**: 版本 4.x 或更高
2. **Python**: 版本 3.6 或更高
3. **依赖库**:
   ```bash
   pip install py2neo
   ```

### 数据库配置

1. 启动Neo4j数据库服务
2. 创建数据库用户和密码（默认配置：用户名`neo4j`，密码`12345678`）
3. 确保数据库运行在默认端口 `7687`

### 使用步骤

#### 1. 构建知识图谱

```bash
python build_medicalgraph.py
```

这个脚本会：
- 读取 `data/medical.json` 中的医疗数据
- 创建7类实体节点
- 建立10类实体关系
- 将数据导入Neo4j数据库

#### 2. 导出实体字典

```bash
python get_dict.py
```

这个脚本会：
- 从Neo4j数据库查询各类实体
- 导出实体列表到 `dict/Query/` 目录

### 核心功能说明

#### MedicalGraph类主要方法：

1. **`read_nodes()`**: 解析medical.json数据，提取实体和关系
2. **`create_graphnodes()`**: 创建图数据库节点
3. **`create_graphrels()`**: 创建图数据库关系
4. **`create_relationship()`**: 创建特定类型的实体关系
5. **`export_data()`**: 导出实体数据到文本文件

## 数据示例

### 疾病节点示例
```json
{
  "name": "大叶性肺炎",
  "desc": "由肺炎双球菌等细菌感染引起的呈大叶性分布的急性肺实质炎症",
  "cause": "多种细菌均可引起，绝大多数为肺炎链球菌",
  "prevent": "注意预防上呼吸道感染，加强耐寒锻炼",
  "cure_department": ["内科", "呼吸内科"],
  "cure_way": ["青霉素等抗生素药物治疗"],
  "cure_lasttime": "7-10天",
  "cured_prob": "90%以上"
}
```

### 关系示例
```cypher
# 疾病-症状关系
(大叶性肺炎)-[:has_symptom]->(胸痛)
(大叶性肺炎)-[:has_symptom]->(发烧)

# 疾病-药物关系  
(大叶性肺炎)-[:common_drug]->(乳酸左氧氟沙星片)
(大叶性肺炎)-[:recommand_drug]->(阿奇霉素片)

# 疾病-科室关系
(大叶性肺炎)-[:belongs_to]->(呼吸内科)
```

## 查询示例

### 基础查询

```cypher
// 查询所有疾病
MATCH (d:Disease) RETURN d.name LIMIT 10

// 查询特定疾病的症状
MATCH (d:Disease {name: "大叶性肺炎"})-[:has_symptom]->(s:Symptom) 
RETURN s.name

// 查询疾病的治疗药物
MATCH (d:Disease {name: "大叶性肺炎"})-[:common_drug]->(drug:Drug) 
RETURN drug.name
```

### 复杂查询

```cypher
// 查询具有相同症状的疾病
MATCH (d1:Disease)-[:has_symptom]->(s:Symptom)<-[:has_symptom]-(d2:Disease)
WHERE d1.name = "大叶性肺炎" AND d1 <> d2
RETURN d2.name, s.name

// 查询疾病的完整治疗信息
MATCH (d:Disease {name: "大叶性肺炎"})
OPTIONAL MATCH (d)-[:belongs_to]->(dept:Department)
OPTIONAL MATCH (d)-[:common_drug]->(drug:Drug)
OPTIONAL MATCH (d)-[:need_check]->(check:Check)
RETURN d.name, d.desc, collect(dept.name), collect(drug.name), collect(check.name)
```

## 应用场景

1. **医疗问答系统**: 基于知识图谱的医疗智能问答
2. **疾病诊断辅助**: 根据症状推荐可能的疾病和检查
3. **药物推荐**: 基于疾病推荐相关药物
4. **医疗知识检索**: 快速查找疾病相关信息
5. **临床决策支持**: 为医生提供相关医疗知识参考

## 数据统计

- **疾病数量**: 150条
- **药物数量**: 约3,800种
- **症状数量**: 约6,000种  
- **检查项目**: 数百种
- **食物条目**: 约4,900种
- **科室数量**: 数十个
- **生产厂商**: 数百家

## 扩展建议

1. **数据扩充**: 增加更多疾病和医疗实体数据
2. **关系丰富**: 添加更多类型的医疗实体关系
3. **接口开发**: 开发REST API接口供其他应用调用
4. **可视化**: 集成图形化界面展示知识图谱
5. **智能推理**: 增加基于规则的推理功能

## 注意事项

⚠️ **免责声明**: 本项目仅用于学习和研究目的，不能作为医疗诊断依据。任何医疗决策都应咨询专业医生。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目采用开源许可证，详情请查看项目中的许可证文件。

---
