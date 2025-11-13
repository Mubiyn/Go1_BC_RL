# Reinforcement Learning Models

This directory contains RL fine-tuned policy models.

## Download Pre-trained Models

Pre-trained RL models are available at:
- **Google Drive**: [Link - TBD]
- **File size**: ~120 MB (all 7 models)

## Files

After downloading, this directory should contain:
```
models/rl/
├── rl_from_scratch.zip              # Baseline RL trained from scratch
├── bc_finetune_task_only.zip        # BC + RL with only task reward
├── bc_hybrid_0.7_0.3.zip           # BC + RL (70% task, 30% gait)
├── bc_hybrid_0.5_0.5.zip           # BC + RL (50% task, 50% gait)
├── bc_hybrid_0.3_0.7.zip           # BC + RL (30% task, 70% gait)
├── bc_adaptive.zip                  # BC + RL with adaptive weights
└── training_logs/                   # TensorBoard logs
```

## Download Instructions

```bash
cd models/rl/

# Download all models (single archive)
wget --load-cookies /tmp/cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" \
  -O rl_models.zip && rm -rf /tmp/cookies.txt

# Extract
unzip rl_models.zip

# Or use gdown
pip install gdown
gdown --id FILEID --output rl_models.zip
unzip rl_models.zip
```

## Training Your Own Models

### 1. Train RL from Scratch (Baseline)
```bash
python scripts/train_rl_scratch.py \
    --config config/rl_scratch_config.yaml \
    --total_timesteps 10000000 \
    --output models/rl/rl_from_scratch.zip
```

### 2. Fine-tune BC with Different Reward Strategies

**Task reward only:**
```bash
python scripts/train_rl_finetune.py \
    --bc_policy models/bc/bc_policy.pth \
    --reward_type task \
    --total_timesteps 3000000 \
    --output models/rl/bc_finetune_task_only.zip
```

**Hybrid fixed weights:**
```bash
# 70% task, 30% gait
python scripts/train_rl_finetune.py \
    --bc_policy models/bc/bc_policy.pth \
    --reward_type hybrid \
    --alpha_task 0.7 \
    --alpha_gait 0.3 \
    --total_timesteps 3000000 \
    --output models/rl/bc_hybrid_0.7_0.3.zip
```

**Adaptive weights:**
```bash
python scripts/train_rl_finetune.py \
    --bc_policy models/bc/bc_policy.pth \
    --reward_type adaptive \
    --total_timesteps 3000000 \
    --output models/rl/bc_adaptive.zip
```

## Model Performance

| Model | Velocity (m/s) | Gait Quality | Success Rate | Training Time |
|-------|---------------|--------------|--------------|---------------|
| RL from Scratch | 0.68 ± 0.12 | 0.42 | 76% | ~48h (10M steps) |
| BC + Task Only | 0.89 ± 0.08 | 0.38 | 89% | ~14h (3M steps) |
| BC + Hybrid (0.7/0.3) | 0.76 ± 0.05 | 0.72 | 95% | ~14h (3M steps) |
| BC + Hybrid (0.5/0.5) | 0.69 ± 0.06 | 0.81 | 97% | ~14h (3M steps) |
| BC + Hybrid (0.3/0.7) | 0.61 ± 0.04 | 0.88 | 98% | ~14h (3M steps) |
| **BC + Adaptive** | **0.74 ± 0.05** | **0.76** | **96%** | **~14h (3M steps)** |

*Training times based on RTX 3080 with 16 parallel environments*

## Usage

Load and test a model:
```python
from stable_baselines3 import PPO

# Load model
model = PPO.load("models/rl/bc_adaptive.zip")

# Use for inference
obs = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

## Algorithm

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Policy network**: MLP [256, 256]
- **Value network**: MLP [256, 256]
- **Learning rate**: 3e-4
- **Batch size**: 64
- **n_steps**: 2048
- **Parallel environments**: 16
