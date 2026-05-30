# Industrial AI Process Logic

Zero One Hack Track 1 solution for learning and benchmarking semiconductor process logic.

The repository is built around the official Industrial AI starter pack from `Lumos-Data/zero_one_hack_01`. It includes the public training data, a grammar-aware baseline, a deterministic anomaly detector using the official validator, optional PyTorch transformer training, local development metrics, and Leonardo Slurm scripts.

## What Is Included

- Official copied data in `data/raw/training_data/`
- `industrial_ai.prepare`: vocabulary and corpus stats
- `industrial_ai.generate_extra`: synthetic sequence generation through the official generator
- `industrial_ai.make_devset`: local dev eval creation
- `industrial_ai.infer`: submission CSV generation
- `industrial_ai.metrics`: local dev scoring
- `industrial_ai.train`: optional step-token transformer training
- Leonardo scripts in `scripts/`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The baseline pipeline itself uses only Python stdlib. `torch` is needed only for neural training.

## Local Smoke Run

```bash
python -m industrial_ai.prepare
python -m industrial_ai.make_devset --valid-per-family 5 --anomaly-valid-per-family 5 --anomaly-invalid-per-family 5
python -m industrial_ai.infer
python -m industrial_ai.metrics
```

This creates:

- `data/dev/eval_input_valid.csv`
- `data/dev/eval_input_anomaly.csv`
- `submissions/nextstep.csv`
- `submissions/completion.csv`
- `submissions/anomaly.csv`

## Official Eval Inference

When organizers provide the official eval files, copy them into `data/eval/` and run:

```bash
python -m industrial_ai.infer \
  --valid-input data/eval/eval_input_valid.csv \
  --anomaly-input data/eval/eval_input_anomaly.csv \
  --out-dir submissions
```

Required Track 1 submission files:

- `submissions/nextstep.csv`
- `submissions/completion.csv`
- `submissions/anomaly.csv`

## Training

Generate more valid sequences:

```bash
python -m industrial_ai.generate_extra --family mosfet --count 10000 --seed 101 --output data/generated/MOSFET_extra.csv
python -m industrial_ai.generate_extra --family igbt --count 10000 --seed 102 --output data/generated/IGBT_extra.csv
python -m industrial_ai.generate_extra --family ic --count 10000 --seed 103 --output data/generated/IC_extra.csv
```

Train a model:

```bash
python -m industrial_ai.train --model-size tiny --epochs 4 --batch-size 64 --device cuda
```

Supported model sizes: `tiny`, `small`, `medium`.

## Leonardo

Copy `.env.example` to `.env` locally and fill your Leonardo username/password outside git. Stage this repo on Leonardo, then use:

```bash
sbatch scripts/leonardo_generate.sh
sbatch --export=MODEL_SIZE=tiny,EPOCHS=6 scripts/leonardo_train.sh
sbatch --export=VALID_INPUT=data/eval/eval_input_valid.csv,ANOMALY_INPUT=data/eval/eval_input_anomaly.csv scripts/leonardo_infer.sh
```

Known hackathon defaults:

- partition: `boost_usr_prod`
- reservation: `s_tra_ncc`
- login host: `login01-ext.leonardo.cineca.it`

## Event Submission Checklist

Submit through the Tally form by Sunday 10:00:

- Team name
- Public GitHub repository URL
- Slides PDF, max 10 slides
- Demo video or link, max 2 minutes

Repo requirements:

- public
- MIT licensed
- `README.md`
- `REPORT.md`
- dependency manifest
- no secrets
- clean-checkout runnable

Track-specific repo deliverables:

- `nextstep.csv`
- `completion.csv`
- `anomaly.csv`
- training logs/checkpoints/loss curves or links
- metrics report with per-family breakdown where possible
- baseline vs trained examples

## Data Attribution

Official challenge data and grammar are copied from:

https://github.com/Lumos-Data/zero_one_hack_01/tree/main/tracks/industrial-infineon/training_data

