# 公共数据集接入计划

## 主数据集

Mendeley Germination Detection Dataset:

https://data.mendeley.com/datasets/4wkt6thgp6/3

该数据集包含玉米、黑麦、珍珠粟培养皿时序图像，并提供 germinated / non-germinated bounding-box 标注。

## 接入步骤

1. 下载数据到 `data/public_dataset/`。
2. 将原始标注转换为 YOLO 或 COCO 格式。
3. 训练目标检测模型，类别为：

```text
0: non_germinated
1: germinated
```

4. 将训练权重放入 `models/seed_detector.pt`。
5. 在系统中替换当前 OpenCV 演示检测器。

## 实验指标

- 检测 mAP
- 发芽/未发芽分类准确率
- 单图发芽率 MAE
- 批次最终发芽率 MAE
- T50 误差

## 注意事项

内置示例只用于演示系统交互。参赛报告中的所有模型性能和质检效果必须来自公共数据集验证结果。
