# 数据说明

本仓库不包含原始 ECoG 记录。训练和回放示例面向 BCI Competition IV
相关的 MATLAB 数据格式；请从数据提供方或竞赛官方网站自行获取数据，并遵守其许可、
引用和隐私要求：

- BCI Competition IV: <https://www.bbci.de/competition/iv/>

不要将被试数据直接提交到本仓库，也不要在未确认授权的情况下重新分发原始记录或
由其导出的可识别信息。

## 预期 MATLAB 变量

单被试 `.mat` 文件应至少包含：

- `train_data`：二维数组，形状为 `(time, channels)`。
- `train_dg`：二维数组，形状为 `(time, 5)`，对应五个手指的位置标签。

可选变量：

- `test_data`：二维数组，形状为 `(time, channels)`，用于无标签流式回放。

不同被试可能具有不同通道数。当前单被试训练脚本直接使用文件中的通道宽度；训练和
推理必须保持相同的特征配置、窗口长度和通道宽度。

## 本地目录建议

将数据放在仓库根目录下的 `data/` 中，例如：

```text
data/
└── sub1_comp.mat
```

`data/` 和所有 `*.mat` 文件均已被 `.gitignore` 排除。README 中的命令使用
`./data/sub1_comp.mat` 作为示例路径。

## 生成内容

训练脚本默认将模型、指标和图表写入 `artifacts/`。该目录及 `*.joblib` 文件不会
提交到 GitHub；如需共享模型，请先确认数据许可允许分发衍生产物，再使用单独的受控
发布渠道。
