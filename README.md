# TKRL_Reproduction

New: Add instruction for data preprocessing

# INTRODUCTION

Type-embodied Knowledge Representation Learning (TKRL)

Representation Learning of Knowledge Graphs with Hierarchical Types (IJCAI'16)

Written by Ruobing Xie


# COMPILE 

Just type make in the folder ./


# DATA

FB15k is published by the author of the paper "Translating Embeddings for Modeling Multi-relational Data (2013)." 
<a href="https://everest.hds.utc.fr/doku.php?id=en:transe">[download]</a>
You can also get FB15k from here: <a href="http://pan.baidu.com/s/1eSvyY46">[download]</a>

Entity types and relation-specific type constraint information for FB15k. <a href="http://pan.baidu.com/s/1c1ChN7i">[download]</a> 

FB15k+ is a new dataset based on FB15k, including new relations and triples. <a href="http://pan.baidu.com/s/1mitkRRq">[download]</a>

For data preprocessing, please refer to Data_preprocessing_for_TKRL.pdf.

### 一、 属于 FB15k（以及 FB15k+）的原始三元组数据

这部分文件记录了知识图谱中最基础的结构——三元组 `(Head, Relation, Tail)`。

*   **`train.txt`, `valid.txt`, `test.txt`**：这三个文件是 FB15k（或经过作者扩充的 FB15k+）的核心数据集，分别用于模型的训练、验证和测试。
    *   **内容举例（来自 `train.txt`）**：
        ```text
        /m/027rn	/m/06cx9	/location/country/form_of_government
        /m/017dcd	/m/06v8s0	/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor
        ```
        每一行是一个三元组，由制表符（`\t`）分隔。例如第一行表示：`/m/027rn`（多米尼加共和国） 和 `/m/06cx9`（共和国） 这两个实体之间存在 `/location/country/form_of_government`（国家政体） 的关系。

### 二、 实体和关系的 ID 映射表

模型训练通常无法直接处理字符串，因此需要将所有的字符串实体和关系映射为连续的整数 ID。

*   **`entity2id.txt`**：将 Freebase 中的实体 MID（Machine ID）映射为一个整数。
    *   **内容举例**：
        ```text
        /m/06rf7	0
        /m/0c94fn	1
        ```
        表示实体 `/m/06rf7` 在 C++ 代码的数组中对应的索引为 `0`。
*   **`relation2id.txt`**：将 Freebase 中的关系字符串映射为一个整数。

### 三、 实体的层次类别信息 (Hierarchical Entity Types)

这是 TKRL 论文的核心，它打破了传统模型中实体只有一个向量表示的限制。论文将类别分为两层：底层的具体 Type 和顶层的宽泛 Domain。

*   **`type2id.txt`**：底层具体类型（Type）的 ID 映射表。
    *   **内容举例**：
        ```text
        people/appointed_role	0
        people/appointer	1
        ```
        表示 `people/appointed_role`（被任命的角色） 这个具体类型对应的 ID 为 `0`。
*   **`domain2id.txt`**：顶层宽泛领域（Domain）的 ID 映射表。
    *   **内容举例**：
        ```text
        people	0
        location	1
        ```
        可以看到，Domain 比 Type 要宽泛得多，比如 `people` 领域下会包含 `people/appointed_role` 等多个具体的 Type。

*   **`entity2type.txt` / `typeEntity.txt`**：这俩文件互为倒排索引，记录了每个实体所属的类型。
    *   **`typeEntity.txt` 内容举例**：
        ```text
        0	10	7008	1065	10165 ...
        1	32	207	3830	12131 ...
        ```
        以第一行为例：第一个数字 `0` 代表 Type ID 为 0（即 `people/appointed_role`），后面的数字 `10`表示该类别下总共有 10 个实体，紧接着的 `7008 1065 10165...` 则是这 10 个实体对应的 Entity ID。这就是为什么代码里会有 `type_entity_list[temp_tail_type][jj]` 这样的操作，用于在软类别约束（STC）负采样时快速找到同类别的实体。

### 四、 特定关系下的类别约束 (Relation-specific Type Constraint)

在某一个特定的关系下，头实体和尾实体应该属于什么类型，这是受到严格（或软性）约束的。这对应于论文中公式 4 的 $C_{rh}$，用于构建实体的投影矩阵和指导 STC 负采样。

*   **`relationType.txt`**：记录某个特定关系下，允许的 Head Type 和 Tail Type。
    *   **内容举例**：
        ```text
        /people/appointed_role/appointment./people/appointment/appointed_by	people/appointed_role	people/appointer
        ```
        这一行表示：对于关系 `/people/.../appointed_by`，它的头实体（Head）必须属于 `people/appointed_role`（被任命的角色）类型，它的尾实体（Tail）必须属于 `people/appointer`（任命者）类型。
*   **`relationDomain.txt`**：与上面类似，记录特定关系下允许的 Head Domain 和 Tail Domain。
    *   **内容举例**：
        ```text
        /people/appointed_role/appointment./people/appointment/appointed_by	people	people
        ```
        表示上述同一个关系，其头实体和尾实体在 Domain 层面都必须属于 `people`（人）这个大类。

在 `Train_transT.cpp` 中，代码正是读取了 `relationType.txt` 和 `relationDomain.txt`，直接知道了每个关系对应的 `Head/Tail` 的 `Type/Domain` ID，从而省去了遍历计算，直接索引到了对应的投影矩阵（即 `domain_mat[tempHeadDomain]` 和 `type_mat[tempHeadType]`）。

# Method

在 TKRL 的论文中，作者提出了如何利用实体的层次类别（Hierarchical Types）来增强知识图谱的表示学习。你提到的公式 3、4、5、6、7 正是该论文核心思想的数学表达。

在开源代码的具体实现中，为了兼顾计算效率，作者对数学公式进行了一些工程上的等价化简和适配。代码中实体的层次类别被简化为两层：**第一层是具体的 Type（如 `book/author`），第二层是更宽泛的 Domain（如 `book`）**。

下面我将结合 `Train_transT.cpp` 代码（分别对应 RHE+STC 和 WHE+STC），为你详细拆解这 5 个公式是如何被翻译成 C++ 代码的。

---

### 1. 公式 3 与 公式 4：关系特定的类别编码（Type Encoder）
**公式 3（一般形式）**：$M_e = \alpha_1 M_{c_1} + \alpha_2 M_{c_2} + \dots + \alpha_n M_{c_n}$
**公式 4（特定关系下的约束）**：$M_{rh} = \frac{\sum_{i=1}^n \alpha_i M_{c_i}}{\sum_{i=1}^n \alpha_i}$，其中若 $c_i \in C_{rh}$ 则 $\alpha_i = 1$，否则为 $0$。

**代码实现细节**：
论文里提到一个实体可能有多个 Type，但在某个特定的关系 $r$ 下，只有符合该关系约束 $C_{rh}$ 的 Type 才会被激活（即 $\alpha_i=1$）。
在代码中，**作者对这一步进行了极大的工程化简**：数据集已经提前预处理好了每种关系 $r$ 对应的唯一的合法 Domain 和 Type（存储在 `relationDomain.txt` 和 `relationType.txt` 中）。
因此，公式 4 中的求和直接坍缩成了唯一的那个合法 Type。

代码在 `prepare()` 函数中直接把这个约束写死了，把每个关系 $r$ 的合法 Type 和 Domain 存入数组：
```cpp
// 读取 relationType.txt 和 relationDomain.txt
head_type_vec[tempRel] = type2id[buf]; // 头实体在该关系下的合法 Type
tail_type_vec[tempRel] = type2id[buf];
head_domain_vec[tempRel] = domain2id[buf]; // 头实体在该关系下的合法 Domain
```
在计算实体向量时，代码**直接通过关系 $r$ 获取唯一的投影矩阵索引**，而不再去做复杂的 $\alpha_i$ 遍历和加权：
```cpp
// calc_entity_vec 函数中
int tempHeadType = head_type_vec[rel];     // 直接拿到公式4算出的合法约束 Type
int tempHeadDomain = head_domain_vec[rel]; // 直接拿到合法约束 Domain
```

---

### 2. 公式 5：递归层次编码器（RHE - Recursive Hierarchy Encoder）
**公式 5**：$M_c = \prod_{i=1}^m M_{c^{(i)}} = M_{c^{(1)}} M_{c^{(2)}} \dots M_{c^{(m)}}$

**代码实现细节**：
该公式对应 `RHE+STC` 目录下的代码。公式表示实体向量在投影时，需要顺着层级结构**递归地进行矩阵乘法**。因为代码中有两层（Domain 是顶层，Type 是底层），所以 $M_c e_h = M_{type} \times M_{domain} \times e_h$。

在 `RHE+STC/Train_transT.cpp` 的 `calc_entity_vec` 函数中，由于多个矩阵相乘的时间复杂度极高 $O(n^3)$，代码采用了结合律 $M_{type} \times (M_{domain} \times e_h)$，将复杂度降为 $O(n^2)$：

```cpp
// 第一步：先用最宽泛的 domain_mat 乘以实体向量，得到中间向量 mid_head_vec
for(int i=0; i<n; i++) {
    mid_head_vec[tid][flag][i] = 0;
    for(int ii=0; ii<n; ii++) {
        // M_domain * e_h
        mid_head_vec[tid][flag][i] += domain_mat[tempHeadDomain][i][ii] * entity_vec[head][ii]; 
    }
}
// 第二步：再用最具体的 type_mat 乘以中间向量，得到最终在关系空间下的 head_final_vec
for(int i=0; i<n; i++) {
    head_final_vec[tid][i] = 0;
    for(int ii=0; ii<n; ii++) {
        // M_type * mid_head_vec
        head_final_vec[tid][i] += type_mat[tempHeadType][i][ii] * mid_head_vec[tid][flag][ii];
    }
}
```
这两步完美的复现了公式 5 中层级矩阵相乘的过程。

---

### 3. 公式 6 与 公式 7：加权层次编码器（WHE - Weighted Hierarchy Encoder）
**公式 6**：$M_c = \sum_{i=1}^m \beta_i M_{c^{(i)}} = \beta_1 M_{c^{(1)}} + \dots + \beta_m M_{c^{(m)}}$
**公式 7**：$\beta_i : \beta_{i+1} = (1-\eta) : \eta$，且 $\sum_{i=1}^m \beta_i = 1$

**代码实现细节**：
这两个公式对应 `WHE+STC` 目录下的代码。WHE 认为不同的层次不应该是连乘关系，而应该是不同矩阵独立映射后，进行**加权求和**。其中越精确的底层（Type）权重越大，越宽泛的高层（Domain）权重越小。

在代码顶部，作者定义了公式 7 中的超参数：
```cpp
double type_weight = 0.9;  // 对应底层 Type 的权重 \beta_1，也即论文中的 1 - \eta
```
因为只有两层，所以 Domain 的权重自然就是 `1 - type_weight` 即 `0.1`（对应 $\beta_2$ ，也即 $\eta$），完美符合 $(1-\eta) : \eta = 0.9 : 0.1$。

在 `WHE+STC/Train_transT.cpp` 的 `calc_entity_vec` 函数中，可以看到加权求和的完全实现：
```cpp
for(int i=0; i<n; i++) {
    double type_score = 0, domain_score = 0;
    
    // 1. 先独立计算底层 Type 矩阵的投影 (M_{type} * e_h)
    for(int ii=0; ii<n; ii++) {
        type_score += type_mat[tempHeadType][i][ii] * entity_vec[head][ii];
    }
    
    // 2. 再独立计算高层 Domain 矩阵的投影 (M_{domain} * e_h)
    for(int ii=0; ii<n; ii++) {
        domain_score += domain_mat[tempHeadDomain][i][ii] * entity_vec[head][ii];
    }
    
    // 3. 执行公式 6 和 7：用预设好的 \beta_1 (type_weight) 和 \beta_2 (1-type_weight) 进行加权求和
    head_final_vec[tid][i] = type_score * type_weight + domain_score * (1-type_weight);
}
```

### 总结
TKRL 的开源代码对原论文做了非常高效的工程实现：
1. 用预先抽取的最强关联关系（`relationType.txt`）直接平替了公式 3、4 中的复杂概率求和。
2. RHE 利用矩阵乘法分配律，把连乘转化为了两次 $O(n^2)$ 的向量-矩阵相乘。
3. WHE 直接写死了 $\beta_1=0.9$ 和 $\beta_2=0.1$ 的衰减比例（$\eta=0.1$），用标量加权融合了两个不同层级映射出来的空间向量。


# Step
cd TKRL_Reproduction/RHE+STC

cd TKRL_Reproduction/WHE+STC

make

./Train_transT -size 50 -margin 1 -method 0

./Train_transT -size 50 -margin 1 -method 0 -resume

./Test_transT unif

# CITE

If the codes or datasets help you, please cite the following paper:

Ruobing Xie, Zhiyuan Liu, Maosong Sun. Representation Learning of Knowledge Graphs with Hierarchical Types. The 25th International Joint Conference on Artificial Intelligence (IJCAI'16).
