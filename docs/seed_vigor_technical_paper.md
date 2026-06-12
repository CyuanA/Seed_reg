# 面向种子发芽实验的视觉表型监测、活力评价与数据闭环系统

作者：待填写  
单位：待填写  
项目名称：SeedGerm-Vigor  
代码仓库：https://github.com/CyuanA/Seed_reg

## 摘要

种子发芽实验是种子质量检测、品种筛选、胁迫处理评价和农业教学实验中的基础环节。传统实验流程通常依赖人工定时拍照、人工计数和手工制表，面对多时间点、多批次样本时容易出现统计效率低、主观判断差异和结果整理成本高等问题。针对上述问题，本文设计并实现了一套面向农业实验的低成本种子活力视觉表型监测与自动评价系统 SeedGerm-Vigor。系统以普通 RGB 培养皿图像和 USB 摄像头为输入，基于 YOLO 目标检测模型识别已发芽与未发芽种子，并进一步完成单时间点发芽率统计、时序发芽率曲线构建、T50 估计、平均发芽时间、发芽速度指数、整齐度评分和综合活力评分计算。系统同时支持多批次实验对比、摄像头定时采集、人工校正与 YOLO 格式再训练数据导出，形成“图像采集—模型识别—人工校正—指标计算—报告输出—数据回流”的实验自动化闭环。

在公开发芽图像数据集上的测试结果表明，接入的目标检测模型在测试集上取得 Precision 0.944、Recall 0.922、mAP50 0.967 和 mAP50-95 0.892；基于检测结果计算的发芽率平均绝对误差为 4.87%，T50 平均绝对误差为 1.61 h。结果说明，该系统能够在普通图像采集条件下提供可用的种子发芽实验辅助统计能力。本文同时讨论了当前系统在跨物种泛化、复杂拍摄环境、胚根长度测量和活力评分标准化方面的局限，并给出后续改进方向。

**关键词：** 种子发芽；视觉表型；目标检测；T50；活力评价；人工校正；数据闭环

## Abstract

Seed germination tests are fundamental in seed quality assessment, variety screening, stress-treatment evaluation, and agricultural teaching experiments. Conventional workflows rely heavily on manual imaging, visual counting, spreadsheet-based statistics, and manual report preparation. These procedures become inefficient and error-prone when experiments involve multiple time points, seed lots, or treatment groups. To address this problem, this paper presents SeedGerm-Vigor, a low-cost visual phenotyping and automatic evaluation system for seed vigor analysis in agricultural experiments. The system uses ordinary RGB petri-dish images and USB camera input, applies a YOLO-based detector to identify germinated and non-germinated seeds, and automatically computes germination rate, time-series germination curves, T50, mean germination time, germination speed index, uniformity score, and a system-defined vigor score. In addition, the system supports timed camera acquisition, multi-batch comparison, manual correction of detection results, YOLO-format retraining data export, and automatic experiment report generation.

On a public germination image dataset, the deployed detector achieved a precision of 0.944, recall of 0.922, mAP50 of 0.967, and mAP50-95 of 0.892 on the test split. The germination-rate mean absolute error was 4.87%, and the T50 mean absolute error was 1.61 h. These results indicate that the proposed system can provide practical image-based assistance for germination experiment statistics under ordinary RGB imaging conditions. The paper also discusses limitations in cross-species generalization, imaging-condition robustness, radicle length measurement, and the standardization of vigor scoring.

**Keywords:** seed germination; visual phenotyping; object detection; T50; seed vigor; manual correction; data loop

## 1 引言

种子发芽能力是农业生产、种子质量检测和实验室筛选中的关键表型之一。对于常规发芽实验，研究人员通常需要在多个时间点观察培养皿图像，记录每个时间点的发芽数量，并进一步计算最终发芽率、T50、平均发芽时间等指标。该流程的主要问题不在于单次计数是否困难，而在于当实验扩展到多个品种、多个处理组和多个时间点后，人工计数和统计整理会迅速变成低效且容易出错的重复劳动。

现有农业图像分析方法已经能够使用数字图像处理或深度学习模型完成种子目标识别。公开数据集和相关研究也表明，基于图像的发芽检测在玉米、黑麦、珍珠粟等培养皿图像场景中具有可行性 [1,2]。然而，单纯输出“已发芽/未发芽”的检测框并不能直接满足农业实验工作流。实验人员真正需要的是：在低成本设备条件下采集图像，自动统计多时间点数据，计算农业实验指标，比较不同批次或处理组，并生成可归档的实验报告。与此同时，模型在实际使用中不可避免存在误检、漏检和类别误判，若系统不能支持人工校正与数据回流，则很难形成持续优化能力。

本文提出 SeedGerm-Vigor 系统，目标不是构建一个孤立的发芽二分类模型，而是构建一个面向种子发芽实验流程的低成本视觉表型监测与自动评价系统。系统围绕农业实验中的完整链条展开：图像采集、种子检测、时序统计、活力评价、多批次对比、人工校正和报告输出。与只做单张图像识别的系统相比，本文工作的重点是把视觉检测结果转换为实验可用的表型指标和结论输出。

本文主要贡献如下：

1. 设计了面向培养皿图像的低成本视觉表型监测系统，支持图像上传、本机摄像头拍照和 USB 摄像头定时采集。
2. 基于 YOLO 目标检测模型实现已发芽与未发芽种子的自动识别，并提供高精度、平衡和高召回三种推理策略。
3. 构建了从检测结果到农业实验指标的自动计算流程，包括发芽率、T50、平均发芽时间、发芽速度指数、整齐度评分和综合活力评分。
4. 实现了多批次实验对比和实验报告导出，支持 CSV、Markdown 和 Word 格式结果归档。
5. 实现了人工校正与再训练数据导出模块，使用户能够修正误检、漏检和类别错误，并导出 YOLO 格式标签用于后续模型微调。

## 2 相关工作与技术背景

### 2.1 种子发芽实验与活力评价

发芽率和发芽速度是种子质量评价中的常用观测指标。国际种子检验协会（International Seed Testing Association, ISTA）发布的种子检验规则为种子检测实验提供了重要的标准化依据 [3]。在实验研究中，最终发芽率可反映样本在给定条件下的总体发芽能力；T50 用于描述达到 50% 发芽率所需时间，能够反映发芽速度；平均发芽时间和发芽速度指数则可进一步描述发芽过程的快慢和集中程度。

需要明确的是，本文系统中的“综合活力评分”是基于最终发芽率、T50 和整齐度构造的系统辅助评价指标，目的在于支持批次排序和实验展示，并不等同于官方种子检验标准中的认证结论。该边界必须明确，否则系统容易被误解为替代专业种子检验流程。

### 2.2 发芽图像数据集与视觉表型分析

Mendeley Data 发布的 Germination Detection Dataset 提供了面向发芽检测任务的培养皿图像与标注资源 [1]。相关研究提出了基于图像分析的谷物和豆类发芽预测系统，为利用计算机视觉进行发芽状态识别提供了数据基础和方法参考 [2]。此外，SeedGerm 等工具也说明了低成本图像分析软件在种子发芽实验中的应用价值 [4]。

这些工作证明了图像分析在发芽实验中的可行性，但从系统落地角度看，仅完成检测或预测仍不够。农业实验现场更需要的是围绕实验流程的工具链：采集、识别、统计、校正、对比和报告。本项目正是在这一系统层面进行整合。

### 2.3 目标检测模型

YOLO 系列方法将目标定位和类别识别统一为端到端检测任务，具备较高的推理效率 [5]。Ultralytics YOLO 框架提供了相对成熟的训练、验证和推理接口，适合在工程系统中快速接入目标检测能力 [6]。本文系统采用基于 YOLO 框架训练得到的种子检测模型，将培养皿图像中的种子分为已发芽和未发芽两类，并在此基础上进行统计分析。

## 3 系统总体设计

### 3.1 设计目标

SeedGerm-Vigor 的设计目标包括四个层面。

第一，降低采集成本。系统使用普通 RGB 图像和 USB 摄像头，不依赖高光谱、显微成像或专用表型平台。

第二，减少人工统计负担。系统自动检测培养皿中的种子目标，并区分已发芽与未发芽种子。

第三，服务实验指标计算。系统不仅显示检测框，还自动生成发芽率曲线并计算 T50、平均发芽时间、发芽速度指数和活力评分。

第四，形成数据闭环。系统允许用户修正检测结果，并导出再训练标签，使后续模型微调具备数据来源。

### 3.2 系统架构

系统流程如下：

```text
图像上传 / 本机摄像头 / USB摄像头定时采集
        ↓
图像预处理与时间戳记录
        ↓
YOLO种子目标检测
        ↓
已发芽 / 未发芽状态识别
        ↓
人工校正与检测结果修正
        ↓
单时间点发芽率统计
        ↓
时序发芽率曲线构建
        ↓
T50 / 平均发芽时间 / 发芽速度 / 活力评分计算
        ↓
多批次对比与报告导出
        ↓
YOLO格式再训练数据导出
```

系统采用 Streamlit 作为交互界面，PyTorch/Ultralytics YOLO 作为目标检测推理框架，PIL/OpenCV 处理图像输入和摄像头采集，Pandas 进行统计表构建，python-docx 完成 Word 报告生成。

### 3.3 功能模块

系统主要包括六个模块。

**图像输入模块。** 支持单图上传、时序图像上传、本机摄像头拍照和 USB 摄像头定时采集。定时采集模块允许用户设置实验编号、摄像头编号、采集间隔和采集张数，采集后自动保存图像并更新统计曲线。

**种子检测模块。** 系统调用 YOLO 检测模型输出目标框、类别和置信度。类别包括 germinated 和 non_germinated。系统提供高精度、平衡和高召回三种推理策略，并通过置信度阈值和 NMS IoU 调节误检与漏检之间的权衡。

**时序分析模块。** 对多时间点图像按时间排序，计算每个时间点的总种子数、已发芽数、未发芽数和发芽率，并绘制发芽率曲线。

**活力评价模块。** 基于时序统计结果计算最终发芽率、T50、平均发芽时间、发芽速度指数、整齐度评分和综合活力评分。

**多批次对比模块。** 系统按文件名或目录结构自动识别不同批次，分别计算各批次指标，并根据综合活力评分和最终发芽率排序，输出最优批次。

**人工校正与数据闭环模块。** 用户可在表格中删除误检框、修改类别、调整框坐标或新增漏检框。校正后系统重新计算发芽率，并导出图像、YOLO 标签、校正表和元数据，用于后续模型微调。

## 4 关键方法

### 4.1 发芽率计算

设第 \(t\) 个时间点检测到的总种子数为 \(N_t\)，已发芽种子数为 \(G_t\)，则该时间点发芽率为：

\[
R_t = \frac{G_t}{N_t} \times 100\%
\]

当 \(N_t=0\) 时，系统将发芽率记为 0，以避免除零错误。该处理适合软件鲁棒性，但在真实实验中若出现 \(N_t=0\)，应检查图像质量或模型检测失败原因。

### 4.2 T50 插值估计

T50 表示发芽率首次达到 50% 所需时间。系统按照时间升序遍历发芽率曲线。若相邻时间点 \((t_{i-1},R_{i-1})\) 和 \((t_i,R_i)\) 满足 \(R_{i-1}<50\leq R_i\)，则采用线性插值：

\[
T50=t_{i-1}+\frac{50-R_{i-1}}{R_i-R_{i-1}}(t_i-t_{i-1})
\]

如果整个时序中的最高发芽率未达到 50%，系统将 T50 标记为“未达到”。

### 4.3 平均发芽时间与发芽速度指数

设相邻时间点之间新增发芽数为 \(\Delta G_i\)，对应时间为 \(t_i\)。平均发芽时间定义为：

\[
MGT=\frac{\sum_i \Delta G_i t_i}{\sum_i \Delta G_i}
\]

发芽速度指数定义为：

\[
GSI=\sum_i \frac{\Delta G_i}{t_i}
\]

当 \(t_i=0\) 时，系统在实现中用 0.5 h 作为替代值以避免除零问题。该替代值是工程实现中的数值保护，不应被解释为实验标准。

### 4.4 综合活力评分

系统定义综合活力评分：

\[
Score = 0.5F + 0.3S + 0.2U
\]

其中 \(F\) 为最终发芽率，\(S\) 为由 T50 转换得到的速度评分，\(U\) 为发芽整齐度评分。实现中速度评分为：

\[
S = \max(0,\min(100,100-2T50))
\]

整齐度评分根据发生新增发芽的时间跨度计算：

\[
U=\max(0,100-3\Delta T)
\]

其中 \(\Delta T\) 是有新增发芽事件的最大时间与最小时间之差。该评分用于系统内部批次排序和实验展示，不宣称替代正式种子活力标准。

### 4.5 人工校正与再训练数据导出

人工校正模块将检测框表示为表格字段：

```text
include, x, y, w, h, label, confidence
```

用户可通过 include 删除误检框，通过 label 修改类别，通过坐标字段修正边界框，也可新增漏检框。导出 YOLO 标签时，系统将边界框从左上角坐标 \((x,y,w,h)\) 转换为 YOLO 所需的归一化中心点格式：

\[
x_c=\frac{x+w/2}{W},\quad y_c=\frac{y+h/2}{H},\quad b_w=\frac{w}{W},\quad b_h=\frac{h}{H}
\]

其中 \(W,H\) 分别为图像宽度和高度。导出包包含原始图像、标签文件、校正表和元数据文件，可用于后续训练集扩充。

## 5 实现

系统采用单页 Streamlit 应用实现，核心目录结构如下：

```text
seed_germination_demo/
├── app.py
├── models/seed_detector.pt
├── results/
│   ├── detection_metrics.csv
│   ├── germination_metrics.csv
│   └── t50_summary.csv
├── src/
│   ├── detector.py
│   ├── yolo_detector.py
│   ├── reporting.py
│   └── vigor.py
└── scripts/
```

其中 `src/yolo_detector.py` 负责模型推理，`src/vigor.py` 负责 T50、平均发芽时间、发芽速度指数、整齐度和活力评分计算，`app.py` 负责交互界面、摄像头采集、人工校正、批次对比和报告导出。

定时采集模块使用 OpenCV 读取 USB 摄像头画面，并将每次采集的图像保存至 `captures/` 目录。每帧图像保存时记录时间戳、检测数量、已发芽数量、未发芽数量和发芽率。由于 Streamlit 的执行模型是脚本式刷新，当前实现采用顺序阻塞式采集，适合演示和低频实验监测；如果要部署为长时间无人值守监测系统，应将采集任务迁移为后台调度进程或边缘端服务。

人工校正模块采用 `st.data_editor` 呈现检测结果表格。该方式实现简单、稳定，但还不是图形化拖拽标注工具。对于比赛演示和小规模校正足够；若要面向真实标注生产，应进一步加入画布式交互标注组件。

## 6 实验设计与结果

### 6.1 数据与任务

系统训练与测试使用公开发芽检测数据集。项目当前结果文件显示，检测模型在测试集上以图像尺寸 640 进行评估；发芽率误差评估使用 1164 张测试图像；T50 评估包含 12 组时序样本，其中 10 组能够计算有效 T50。本文只使用项目 `results/` 目录中的真实结果，不虚构额外实验。

### 6.2 检测性能

目标检测评估结果如表 1 所示。

**表 1 目标检测性能**

| 指标 | 数值 |
|---|---:|
| Precision | 0.9438 |
| Recall | 0.9220 |
| mAP50 | 0.9670 |
| mAP50-95 | 0.8923 |
| 测试图像尺寸 | 640 |

检测模型取得较高 mAP50，说明在测试集分布内能够较好定位培养皿图像中的种子目标。Recall 低于 Precision，说明模型仍存在一定漏检；在发芽实验统计中，漏检会影响总数和发芽率，因此系统提供人工校正功能是必要的。

### 6.3 发芽率误差

发芽率统计误差如表 2 所示。

**表 2 发芽率与计数误差**

| 指标 | 数值 |
|---|---:|
| 测试图像数量 | 1164 |
| 已发芽数量 MAE | 0.4897 |
| 已发芽数量 RMSE | 0.8890 |
| 发芽率 MAE | 0.0487 |
| 发芽率 RMSE | 0.0867 |
| 总种子数 MAE | 0.2345 |

发芽率 MAE 为 0.0487，即约 4.87 个百分点。对于辅助统计系统而言，该误差具有实用意义；但若用于正式种子检验或高精度科学实验，仍需结合人工复核和更多场景验证。

### 6.4 T50 误差

T50 评估结果如表 3 所示。

**表 3 T50 评估结果**

| 指标 | 数值 |
|---|---:|
| 时序样本数 | 12 |
| 有效 T50 样本数 | 10 |
| T50 MAE | 1.6116 h |
| T50 RMSE | 2.1832 h |

T50 MAE 为 1.61 h，说明系统在当前测试条件下能够给出较稳定的发芽速度估计。需要注意，T50 的误差不仅受单图检测误差影响，还受时间采样间隔影响。若采样间隔较大，即使单图检测准确，T50 插值仍可能产生误差。

## 7 应用场景分析

### 7.1 种子质量检测

系统可用于普通实验室中的种子批次初筛。实验人员将同一批次种子放入培养皿，按固定间隔采集图像，系统自动生成发芽率曲线和活力评分。对于需要快速比较多个批次的场景，系统可以降低人工计数和制表成本。

### 7.2 品种筛选与处理组对比

在品种筛选、盐胁迫、药剂处理或储藏时间影响实验中，研究人员通常需要比较多个处理组。系统的多批次模块可以按批次自动分组，并输出最终发芽率、T50、平均发芽时间和综合评分，从而辅助实验结论形成。

### 7.3 农业教学实验

农业教学实验通常强调流程可见性和结果可解释性。系统能够展示从图像到检测框、从检测框到发芽率曲线、从曲线到 T50 和活力评分的完整过程，适合作为教学演示工具。

### 7.4 数据闭环与模型持续优化

在实际部署中，模型不可避免会遇到训练集之外的种子品种、光照条件和背景环境。人工校正模块允许用户将错误结果修正并导出为再训练样本，后续可用于模型微调。这一机制使系统从一次性检测工具扩展为可持续改进的实验平台。

## 8 局限性

当前系统仍存在明显局限。

第一，跨场景泛化尚未充分验证。现有结果来自公开数据集测试划分，不能自动推出系统在所有种子品种、培养基背景、相机视角和光照条件下都同样有效。

第二，系统当前只区分已发芽和未发芽，没有测量胚根长度。对于更严格的种子活力评价，胚根长度、幼苗长度和生长均匀性是重要指标，后续需要结合实例分割或骨架提取方法。

第三，综合活力评分是系统定义的辅助指标，不是官方认证标准。论文和答辩中必须把它描述为批次排序和实验辅助评价指标，不能声称替代专业种子检验。

第四，当前定时采集实现适合低频实验监测和比赛演示，但不是完整工业级后台服务。长时间无人值守部署需要独立采集进程、断点续传、异常恢复和数据备份机制。

第五，人工校正当前采用表格编辑方式，不是图形化拖拽框。虽然足以完成数据闭环演示，但用户体验仍可提升。

## 9 结论与展望

本文设计并实现了一套面向农业发芽实验的低成本视觉表型监测与自动评价系统 SeedGerm-Vigor。系统基于普通 RGB 图像和 USB 摄像头，结合 YOLO 目标检测、时序发芽率分析、T50 估计、多批次对比、人工校正和报告导出，实现了从图像采集到实验结论输出的自动化流程。公开数据集测试结果显示，接入检测模型取得 Precision 0.944、Recall 0.922、mAP50 0.967 和 mAP50-95 0.892，发芽率 MAE 为 4.87%，T50 MAE 为 1.61 h。

后续工作应重点从三个方向推进。第一，扩充不同种子类型和拍摄条件下的数据，验证跨物种和跨设备泛化能力。第二，加入胚根长度和幼苗形态测量，使系统从“发芽状态统计”扩展到“生长质量评价”。第三，将摄像头定时采集模块改造为后台服务或边缘端部署方案，提高长时间监测稳定性。若能完成这些改进，系统将更接近可落地的农业实验视觉表型平台。

## 参考文献

[1] N. Genze and D. Grimm. Data for: Accurate Machine Learning-Based Germination Detection, Prediction and Quality Assessment of Various Seed Cultivars. Mendeley Data, V3, 2023. DOI: 10.17632/4wkt6thgp6.3. https://data.mendeley.com/datasets/4wkt6thgp6/3  
[2] N. Genze, R. Bharti, M. Grieb, S. J. Schultheiss, and D. G. Grimm. “Accurate machine learning-based germination detection, prediction and quality assessment of three grain crops.” *Plant Methods*, vol. 16, article 157, 2020. DOI: 10.1186/s13007-020-00699-x. https://plantmethods.biomedcentral.com/articles/10.1186/s13007-020-00699-x  
[3] International Seed Testing Association. International Rules for Seed Testing. https://www.seedtest.org/en/international-rules-for-seed-testing-_content---1--1083.html  
[4] J. Colmer, C. M. O’Neill, R. Wells, A. Bostrom, D. Reynolds, D. Websdale, G. Shiralagi, W. Lu, Q. Lou, T. Le Cornu, J. Ball, J. Renema, G. Flores Andaluz, R. Benjamins, S. Penfield, and J. Zhou. “SeedGerm: a cost-effective phenotyping platform for automated seed imaging and machine-learning based phenotypic analysis of crop seed germination.” *New Phytologist*, vol. 228, no. 2, pp. 778–793, 2020. DOI: 10.1111/nph.16736. https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.16736  
[5] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. “You Only Look Once: Unified, Real-Time Object Detection.” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016. https://arxiv.org/abs/1506.02640  
[6] Ultralytics. YOLO documentation. https://docs.ultralytics.com/

## 附录 A：当前可复现实验指标来源

本文实验指标来自项目目录：

```text
seed_germination_demo/results/detection_metrics.csv
seed_germination_demo/results/germination_metrics.csv
seed_germination_demo/results/t50_summary.csv
```

其中检测指标记录 Precision、Recall、mAP50 和 mAP50-95；发芽率误差记录测试图像数量、计数 MAE/RMSE 和发芽率 MAE/RMSE；T50 指标记录时序样本数量、有效 T50 样本数量、T50 MAE 和 T50 RMSE。

## 附录 B：答辩推荐表述

本项目不应被表述为“一个 YOLO 发芽检测系统”，而应表述为：

> 面向种子发芽实验的低成本视觉表型监测、活力评价与数据闭环系统。系统通过普通摄像头和 RGB 图像完成种子状态检测，进一步构建时序发芽率曲线并计算 T50 等活力指标，同时支持多批次对比、人工校正和再训练数据导出，实现农业实验从图像采集到结论输出的自动化闭环。
