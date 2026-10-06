# BCI Realtime Decoding

基于 ECoG 的五指位置解码项目，覆盖单被试离线训练、本地流式回放、Apache Beam /
Google Cloud Dataflow 推理，以及 Pub/Sub + Streamlit 实时监控。

公开仓库仅包含源码和部署配置，不包含原始脑电数据、训练模型、实验日志或云端凭据。
数据格式和许可注意事项见 [docs/DATA.md](docs/DATA.md)，模块关系见
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 功能

- Ridge、Random Forest、MLP 和 Hoeffding 回归树训练
- 时域与生理频带滑窗特征
- 离线回放与 Beam worker 共用的 `BCIDecodePipeline`
- DirectRunner 本地验证及 Dataflow 流式部署
- Pub/Sub 输入发布、输出订阅与 Streamlit 曲线监控

## 项目结构

```text
.
├── src/                         # 训练、特征、解码和命令行入口
│   └── beam/                    # Beam/Dataflow 流水线
├── app.py                       # Streamlit Pub/Sub 监控
├── docs/                        # 数据与架构说明
├── deploy/submit_dataflow.sh    # 直接提交 Dataflow 作业
├── scripts/                     # Flex Template 构建与运行
├── requirements.txt             # 离线训练和本地回放
├── requirements-streaming-tree.txt
├── requirements-dataflow.txt
├── requirements-ui.txt
├── Dockerfile.dataflow
├── cloudbuild.dataflow.yaml
└── dataflow_flex_template.json
```

## 环境

- Python 3.10 或更高版本（推荐 Python 3.11）
- Windows PowerShell 可运行离线流程
- Dataflow/Flex Template 脚本需要 Bash、Google Cloud SDK 和已配置的 GCP 项目

创建本地环境：

```powershell
git clone https://github.com/yw4626/bci-realtime-decoding.git
cd bci-realtime-decoding
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 数据准备

从合法来源获取 BCICIV MATLAB 数据，并放到本地 `data/` 目录：

```text
data/
└── sub1_comp.mat
```

训练文件应包含 `train_data`（时间点 × 通道）和 `train_dg`（时间点 × 5 个手指）。
`data/`、`*.mat` 和所有训练产物均已被 Git 忽略。

## 离线训练

Ridge 简单分支：

```powershell
python -m src.train_ridge --mat_path ".\data\sub1_comp.mat"
```

Random Forest 复杂分支：

```powershell
python -m src.train_random_forest --mat_path ".\data\sub1_comp.mat"
```

MLP 复杂分支：

```powershell
python -m src.train_nn --mat_path ".\data\sub1_comp.mat"
```

默认输出到 `artifacts/`：

- Ridge：`model_simple.joblib`
- Random Forest：`model_complex.joblib`
- MLP：`model_mlp.joblib`
- 对应的指标 CSV、训练配置 JSON 和图表

训练脚本按时间顺序划分训练段和验证段。使用 `--help` 查看窗口、采样和模型参数。

### Hoeffding 流式树

```powershell
pip install -r requirements-streaming-tree.txt
python -m src.train_hoeffding --mat_path ".\data\sub1_comp.mat"
python -m src.run_realtime_sim_hoeffding_only --mat_path ".\data\sub1_comp.mat"
```

## 本地解码回放

先训练 Ridge 和 Random Forest，再运行共享解码 pipeline：

```powershell
python -m src.run_pipeline_sim `
  --mat_path ".\data\sub1_comp.mat" `
  --model_dir ".\artifacts" `
  --mode blended `
  --latency
```

`run_pipeline_sim` 默认读取 `model_simple.joblib` 和 `model_complex.joblib`。若只需验证
某个模型，可使用：

```powershell
python -m src.run_realtime_sim_ridge_only --mat_path ".\data\sub1_comp.mat"
python -m src.run_realtime_sim_rf_only --mat_path ".\data\sub1_comp.mat"
python -m src.run_realtime_sim_nn_only --mat_path ".\data\sub1_comp.mat"
```

## Beam DirectRunner

安装 Dataflow 依赖：

```powershell
pip install -r requirements-dataflow.txt
```

输入为 JSONL，每行一个时间点：

```json
{"session_id":"demo","channels":[0.1,0.2,0.3],"sample_seq":1}
```

`channels` 的长度必须与模型训练时的通道宽度一致。使用 Ridge + MLP 运行本地 Beam：

```powershell
python -m src.beam_dataflow `
  --runner DirectRunner `
  --input_path ".\samples.jsonl" `
  --output_path ".\_beam_output" `
  --deadletter_path ".\_beam_deadletter" `
  --model_simple ".\artifacts\model_simple.joblib" `
  --model_complex ".\artifacts\model_mlp.joblib" `
  --mode blended
```

## Google Cloud Dataflow

需要预先创建 GCS bucket、Pub/Sub 输入订阅、输出 Topic（可选 dead-letter Topic）及
Artifact Registry 仓库，并把训练模型上传到 GCS。

构建 Flex Template：

```bash
export GOOGLE_CLOUD_PROJECT=your-project
export REGION=us-central1
export AR_REPOSITORY=your-artifact-repository
export GCS_BUCKET=your-bucket
bash scripts/build_flex_template.sh
```

运行 Flex Template：

```bash
export GOOGLE_CLOUD_PROJECT=your-project
export REGION=us-central1
export GCS_BUCKET=your-bucket
export INPUT_SUBSCRIPTION=projects/your-project/subscriptions/bci-samples
export OUTPUT_TOPIC=projects/your-project/topics/bci-decoded
export MODEL_SIMPLE=gs://your-bucket/artifacts/model_simple.joblib
export MODEL_COMPLEX=gs://your-bucket/artifacts/model_mlp.joblib
bash scripts/run_flex_template.sh
```

也可使用 `deploy/submit_dataflow.sh` 直接提交 `DataflowRunner` 作业。脚本只从环境变量
读取项目、Topic、bucket 和模型位置，不应把真实凭据写入仓库。

## Pub/Sub 工具与监控

发布本地 ECoG 行：

```powershell
python -m src.publish_ecog_stream `
  --project "your-project" `
  --topic "bci-ecog-in" `
  --mat_path ".\data\sub1_comp.mat" `
  --segment train `
  --realtime
```

命令行订阅解码结果：

```powershell
python -m src.subscribe_decoded `
  --project "your-project" `
  --subscription "bci-decoded-pull"
```

启动 Streamlit 监控：

```powershell
pip install -r requirements-ui.txt
$env:GOOGLE_CLOUD_PROJECT = "your-project"
$env:OUTPUT_SUBSCRIPTION = "bci-decoded-pull"
gcloud auth application-default login
streamlit run app.py
```

也可以在 Streamlit 侧边栏手动填写项目 ID 和订阅名。

## 安全与复现

- 不要提交 `.mat`、`artifacts/`、`*.joblib`、`.env` 或服务账号 JSON。
- 训练和推理必须使用相同通道宽度、窗口参数和特征开关。
- Dataflow 全链路会产生 GCP 费用，请在完成实验后停止作业并清理资源。
- 本项目仅用于研究和工程实验，不用于医疗诊断。

## License

源码采用 [MIT License](LICENSE)。数据集及其衍生产物仍受各自数据提供方条款约束。
