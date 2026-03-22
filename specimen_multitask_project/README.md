
# Specimen-level 双分支多任务识别工程

这是一个完整可运行的 PyTorch + timm 工程，面向 **specimen-level** 的多任务学习：

- 主任务：`class_13_id` 的 13 类分类
- 辅助任务：`weathering_grade_specimen` 风化等级分类
- 输入模态：
  - macro 图像 bag
  - micro 图像 bag
  - specimen-level 表格特征
- 输出单位：**specimen**
- 支持缺失模态：
  - macro-only
  - micro-only
  - macro + micro
- 融合方式：
  - baseline：双分支 mean pooling + tabular concat fusion
  - advanced：attention/gated MIL + gated fusion + modality dropout

代码默认遵循你当前数据口径：以 `specimens.csv` 为主表、`images.csv` 通过 `specimen_id` 关联、split 必须在 specimen 级、macro/micro 只是同属一个 specimen 而不做一一配对。与上传 PDF 中 specimen 级组织与 specimen 级 split 的口径一致。fileciteturn0file0

---

## 目录结构

```text
specimen_multitask_project/
  configs/
    baseline_resnet18.yaml
    advanced_mil.yaml
  datasets/
    __init__.py
    metadata.py
    tabular.py
    specimen_dataset.py
  losses/
    __init__.py
    focal.py
    build.py
  models/
    __init__.py
    backbones.py
    pooling.py
    fusion.py
    tabular_encoder.py
    baseline.py
    advanced.py
    build.py
  utils/
    __init__.py
    common.py
    config.py
    seed.py
    logger.py
    checkpoint.py
    image_utils.py
    metrics.py
    split_utils.py
    evaluation.py
  split.py
  train.py
  eval.py
  infer.py
  requirements.txt
  README.md
```

---

## 核心设计

### 1. 数据组织

- 以 `specimens.csv` 为 specimen 主表。
- 每个 specimen 下把 `images.csv` 中的图像按 `scale_type` 分成：
  - `macro_items`
  - `micro_items`
- 支持 `macro` / `micro` 不成对，只要同属一个 specimen。
- `macro_annotations.csv` / `micro_annotations.csv` 可选；存在则自动聚合成表格特征。
- 路径解析不强依赖 Windows 绝对路径：
  - 优先使用 config 中 `root_dir`
  - 再通过 `img_path` 去掉旧前缀后重建相对路径
  - 如果失败，会给出清晰候选路径报错

### 2. split 策略

- **永远按 specimen 划分，不按 image 划分**
- 默认 `train:val:test = 7:2:1`
- 尽量按 `class_13_id` 分层
- 小样本稀有类时的默认策略：
  - 1 个 specimen 的类：直接放 train
  - 2 个 specimen 的类：1 train + 1 val/test
  - 3 个 specimen 的类：2 train + 1 val/test
  - 更大的类：按比例切分，并尽量保证 val/test 不为空
- 会导出：
  - `metadata/specimen_split.csv`
  - `metadata/split_summary.json`

### 3. 表格特征设计

默认采用“安全不泄漏”的策略：

**允许**
- specimen 级元数据（可配置）
- macro/micro 数量统计
- blur / brightness / size 统计（mean/std/min/max）
- annotation 聚合统计（count / ratio）
- 其他采集质量与结构字段

**默认排除**
- `class_13_id`
- `class_13_name`
- `raw_label`
- `raw_label_std`
- `raw_label_id`
- `folder_class_name`
- `weathering_grade_specimen`
- `notes`
- `split`

预处理流程：
- 数值特征：缺失值填补 + 标准化
- specimen 级安全类别特征：one-hot
- annotation 聚合特征：count / ratio 数值化
- 最终再进入 `models/tabular_encoder.py` 中的 MLP 编码器

### 4. Baseline 版本

- macro branch：`timm` backbone（默认 `resnet18`）
- micro branch：`timm` backbone（默认 `resnet18`）
- 两个 backbone **不共享参数**
- 训练期：
  - `num_macro_samples = 8`
  - `num_micro_samples = 8`
  - 不足自动重复采样
- 验证/测试期：
  - 默认 `all` 聚合，可配置 `max_k`
- 聚合方式：
  - backbone -> bag 内 `mean pooling`
  - macro + micro + tabular concat
  - 共享 MLP 融合
  - main head + aux head

### 5. Advanced 版本

- 默认 backbone：`convnext_tiny`
- bag pooling：默认 `gated attention MIL`
- 可切换：
  - `attention`
  - `gated_attention`
  - `transformer`
- 融合方式可切换：
  - `concat`
  - `gated`
- 支持：
  - variable-length bag
  - padding + mask
  - modality dropout
  - missing branch mask
  - attention weight 导出
  - fusion weight 导出

### 6. 损失与评估

主任务：
- `ce`
- `weighted_ce`
- `focal`

辅助任务：
- `CrossEntropyLoss`

总损失：
```text
total_loss = main_loss + lambda_aux * aux_loss
```

默认指标：
- macro F1（主指标）
- per-class F1
- confusion matrix
- specimen-level accuracy
- top-3 accuracy
- aux accuracy / macro F1

模型选择：
- 按验证集 `main macro F1` 保存 `best.pth`
- 同时保存 `last.pth`

---

## 你运行前最先要改的地方

先改 `configs/*.yaml` 里的这几个字段：

1. `data.root_dir`
2. `data.path_resolution.path_strip_prefixes`
3. 如果你的 CSV 列名和当前默认不一致，改 `data.columns`
4. 如果你不想用某些表格字段，改 `data.tabular.*`

对你当前这批 CSV，默认映射已经对齐到：
- `specimen_id`
- `class_13_id`
- `weathering_grade_specimen`
- `scale_type`
- `img_path`
- `file_name`
- `keep_flag`
- `blur_score`
- `brightness_score`

---

## 安装

```bash
pip install -r requirements.txt
```

如果你是离线环境、且 `timm` 预训练权重拉不下来，把 config 里的：

```yaml
model:
  pretrained: false
```

---

## 运行步骤

### 第一步：先生成固定 split

```bash
python split.py --config configs/baseline_resnet18.yaml
```

这一步会在对应输出目录下生成：
- `metadata/specimen_split.csv`
- `metadata/split_summary.json`

### 第二步：训练 baseline

```bash
python train.py --config configs/baseline_resnet18.yaml
```

### 第三步：训练 advanced

```bash
python train.py --config configs/advanced_mil.yaml
```

### 第四步：评估 best checkpoint

```bash
python eval.py --checkpoint /path/to/run/checkpoints/best.pth --split test
```

### 第五步：推理

使用训练时同样的 metadata：

```bash
python infer.py --checkpoint /path/to/run/checkpoints/best.pth
```

使用新的 metadata 文件做推理：

```bash
python infer.py       --checkpoint /path/to/run/checkpoints/best.pth       --specimens-csv /your/new/specimens.csv       --images-csv /your/new/images.csv       --macro-annotations-csv /your/new/macro_annotations.csv       --micro-annotations-csv /your/new/micro_annotations.csv       --root-dir /your/new/image_root
```

指定 specimen 子集：

```bash
python infer.py       --checkpoint /path/to/run/checkpoints/best.pth       --specimen-ids sp_R001_001,sp_R002_001
```

---

## 训练输出

每次训练都会生成一个 run 目录，例如：

```text
/mnt/data/specimen_multitask_runs/baseline_resnet18_multitask_YYYYmmdd_HHMMSS/
```

里面通常包含：

```text
config.yaml
train.log
train_log.csv
tensorboard/
artifacts/
  label_encoders.json
  tabular_preprocessor.json
checkpoints/
  best.pth
  last.pth
metadata/
  specimen_split.csv
  split_summary.json
best_val/
  val_predictions.csv
  val_main_confusion_matrix.png
  ...
final_eval/
  val/
  test/
infer/
  infer_predictions.csv
  infer_attention_weights.csv   # advanced 且开启 attention 时
  infer_fusion_weights.csv      # gated fusion 时
```

---

## 预测 CSV 导出字段

`eval.py` / `infer.py` 导出的 specimen-level 预测 CSV 至少包含：

- `specimen_id`
- `split`
- `true_class_13_id`
- `pred_class_13_id`
- `top1_prob`
- `top3_classes`
- `top3_probs`
- `true_weathering_grade_specimen`
- `pred_weathering_grade_specimen`
- `macro_image_count`
- `micro_image_count`
- `has_macro`
- `has_micro`

advanced 模型如果启用了 attention/gated fusion，会额外导出：
- `attention_weights.csv`
- `fusion_weights.csv`

---

## 常见报错排查

### 1. 路径解析失败

典型报错：
```text
Unable to resolve image path ...
```

处理方法：
- 检查 `data.root_dir`
- 检查 `data.path_resolution.path_strip_prefixes`
- 检查 `images.csv` 里的 `img_path` 是否还指向旧 Windows 路径
- 如果你已经有相对路径列，把 `data.columns.relative_path` 指到那一列

### 2. 缺列报错

典型报错：
```text
Missing required column ...
```

处理方法：
- 改 `configs/*.yaml -> data.columns`
- 不要直接改代码里的列名

### 3. 某个 specimen 两个图像分支都没有

典型报错：
```text
has neither macro nor micro images ...
```

处理方法：
- 检查 `keep_flag` 过滤是否把该 specimen 图全过滤掉了
- 检查路径是否全失效
- 必要时把：
  ```yaml
  data:
    path_resolution:
      skip_missing_images: true
  ```
  打开，并人工检查 metadata

### 4. 预训练权重下载失败

处理方法：
- 把 `model.pretrained` 改成 `false`
- 或者提前把 timm 权重下载好

### 5. Windows 下 DataLoader 卡住

处理方法：
- 先把 config 里的：
  ```yaml
  num_workers: 0
  ```
  保持 0 跑通
- 跑通后再逐步调大

### 6. 稀有类导致 val/test 中类别不完整

这是正常现象，因为你现在 specimen 数很少且很多类是 1~3 个 specimen。
当前 split 逻辑会优先保证：
- 训练集尽量覆盖类别
- val/test 尽量保留样本
- 避免 image-level 泄漏

如果后续 specimen 增多，再重新划分即可。

---

## 推荐使用顺序

1. 先跑 `baseline_resnet18.yaml`
2. 确认数据路径、split、预测导出都正常
3. 再跑 `advanced_mil.yaml`
4. 对比：
   - `macro F1`
   - `per-class F1`
   - `attention/fusion 权重`
   - 缺失模态鲁棒性

---

## 备注

这个工程已经按照你当前 CSV 结构做了默认适配，但真正开始训练前，你仍然应该：
- 先确认图片根目录
- 先确认 split 导出是否符合预期
- 再开始正式训练
