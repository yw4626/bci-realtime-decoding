# BCI Realtime Decoding

A five-finger position decoding project based on ECoG signals. It supports
single-subject offline training, local streaming replay, Apache Beam / Google
Cloud Dataflow inference, and real-time monitoring with Pub/Sub and Streamlit.

This public repository contains source code and deployment configuration only.
It does not include raw neural recordings, trained models, experiment logs, or
cloud credentials. See [docs/DATA.md](docs/DATA.md) for the expected data
format and licensing notes, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the system design.

## Features

- Ridge, Random Forest, MLP, and Hoeffding regression training
- Sliding-window time-domain and physiological frequency-band features
- A shared `BCIDecodePipeline` for offline replay and Beam workers
- Local validation with DirectRunner and streaming deployment on Dataflow
- Pub/Sub input publishing, output subscription, and Streamlit monitoring

## Project Structure

```text
.
├── src/                         # Training, features, decoding, and CLI entry points
│   └── beam/                    # Beam/Dataflow pipeline
├── app.py                       # Streamlit Pub/Sub monitor
├── docs/                        # Data and architecture documentation
├── deploy/submit_dataflow.sh    # Direct Dataflow submission
├── scripts/                     # Flex Template build and run scripts
├── requirements.txt             # Offline training and local replay
├── requirements-streaming-tree.txt
├── requirements-dataflow.txt
├── requirements-ui.txt
├── Dockerfile.dataflow
├── cloudbuild.dataflow.yaml
└── dataflow_flex_template.json
```

## Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- Windows PowerShell for the offline workflow shown below
- Bash, Google Cloud SDK, and a configured GCP project for Dataflow/Flex Template scripts

Create a local environment:

```powershell
git clone https://github.com/yw4626/bci-realtime-decoding.git
cd bci-realtime-decoding
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation

Obtain the BCICIV MATLAB data from an authorized source and place it in the
local `data/` directory:

```text
data/
└── sub1_comp.mat
```

The training file must contain `train_data` with shape `(time, channels)` and
`train_dg` with shape `(time, 5 fingers)`. The `data/` directory, all `*.mat`
files, and generated training artifacts are ignored by Git.

## Offline Training

Train the Ridge simple branch:

```powershell
python -m src.train_ridge --mat_path ".\data\sub1_comp.mat"
```

Train the Random Forest complex branch:

```powershell
python -m src.train_random_forest --mat_path ".\data\sub1_comp.mat"
```

Train the MLP complex branch:

```powershell
python -m src.train_nn --mat_path ".\data\sub1_comp.mat"
```

The default outputs are written to `artifacts/`:

- Ridge: `model_simple.joblib`
- Random Forest: `model_complex.joblib`
- MLP: `model_mlp.joblib`
- Corresponding metrics CSV files, training-profile JSON files, and plots

The training scripts split training and validation segments chronologically.
Use `--help` to inspect window, sampling, and model options.

### Hoeffding Streaming Trees

```powershell
pip install -r requirements-streaming-tree.txt
python -m src.train_hoeffding --mat_path ".\data\sub1_comp.mat"
python -m src.run_realtime_sim_hoeffding_only --mat_path ".\data\sub1_comp.mat"
```

## Local Decoding Replay

Train the Ridge and Random Forest models first, then run the shared decoding
pipeline:

```powershell
python -m src.run_pipeline_sim `
  --mat_path ".\data\sub1_comp.mat" `
  --model_dir ".\artifacts" `
  --mode blended `
  --latency
```

`run_pipeline_sim` reads `model_simple.joblib` and `model_complex.joblib` by
default. To validate an individual model, use one of the following commands:

```powershell
python -m src.run_realtime_sim_ridge_only --mat_path ".\data\sub1_comp.mat"
python -m src.run_realtime_sim_rf_only --mat_path ".\data\sub1_comp.mat"
python -m src.run_realtime_sim_nn_only --mat_path ".\data\sub1_comp.mat"
```

## Beam DirectRunner

Install the Dataflow dependencies:

```powershell
pip install -r requirements-dataflow.txt
```

The input is JSONL with one time point per line:

```json
{"session_id":"demo","channels":[0.1,0.2,0.3],"sample_seq":1}
```

The length of `channels` must match the channel width used during training.
Run Beam locally with Ridge and MLP:

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

Before deployment, create a GCS bucket, a Pub/Sub input subscription, an
output topic, an optional dead-letter topic, and an Artifact Registry
repository. Upload the trained models to GCS.

Build the Flex Template:

```bash
export GOOGLE_CLOUD_PROJECT=your-project
export REGION=us-central1
export AR_REPOSITORY=your-artifact-repository
export GCS_BUCKET=your-bucket
bash scripts/build_flex_template.sh
```

Run the Flex Template:

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

You can also use `deploy/submit_dataflow.sh` to submit a `DataflowRunner` job
directly. The scripts read project, topic, bucket, and model locations from
environment variables. Never store real credentials in the repository.

## Pub/Sub Tools and Monitoring

Publish local ECoG rows:

```powershell
python -m src.publish_ecog_stream `
  --project "your-project" `
  --topic "bci-ecog-in" `
  --mat_path ".\data\sub1_comp.mat" `
  --segment train `
  --realtime
```

Subscribe to decoded results from the command line:

```powershell
python -m src.subscribe_decoded `
  --project "your-project" `
  --subscription "bci-decoded-pull"
```

Start the Streamlit monitor:

```powershell
pip install -r requirements-ui.txt
$env:GOOGLE_CLOUD_PROJECT = "your-project"
$env:OUTPUT_SUBSCRIPTION = "bci-decoded-pull"
gcloud auth application-default login
streamlit run app.py
```

You can also enter the project ID and subscription name manually in the
Streamlit sidebar.

## Security and Reproducibility

- Do not commit `.mat` files, `artifacts/`, `*.joblib`, `.env`, or service
  account JSON files.
- Training and inference must use the same channel width, window parameters,
  and feature flags.
- The Dataflow workflow incurs GCP charges. Stop jobs and clean up resources
  after experiments.
- This project is intended for research and engineering experiments only. It
  is not a medical diagnostic system.

## License

The source code is released under the [MIT License](LICENSE). The dataset and
derived artifacts remain subject to the data provider's terms.
