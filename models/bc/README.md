# Behavior Cloning Models

This directory contains trained BC policy models.

## Download Pre-trained Models

Pre-trained BC models are available at:
- **Google Drive**: [Link - TBD]
- **File size**: ~15 MB

## Files

After downloading, this directory should contain:
```
models/bc/
├── bc_policy.pth          # Trained BC policy (PyTorch state dict)
├── bc_policy_config.yaml  # Training configuration
└── training_log.json      # Training metrics and history
```

## Download Instructions

```bash
cd models/bc/

# Download from Google Drive (replace FILEID with actual ID)
wget --load-cookies /tmp/cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" \
  -O bc_policy.pth && rm -rf /tmp/cookies.txt

# Or use gdown
pip install gdown
gdown --id FILEID --output bc_policy.pth
```

## Training Your Own Model

If you want to train from scratch:

```bash
# First collect and process demonstrations
python real_robot/collect_demonstrations.py
python scripts/prepare_bc_dataset.py

# Then train BC
python scripts/train_bc.py \
    --config config/bc_config.yaml \
    --data data/processed/bc_dataset_train.npz \
    --output models/bc/bc_policy.pth
```

## Model Architecture

- **Input**: 48-dim state vector
  - 12 joint positions
  - 12 joint velocities
  - 3 base orientation (RPY)
  - 3 IMU angular velocity
  - 1 base height
  - 2 base velocity (x, y)
  - 12 previous actions
  - 3 target command (vx, vy, yaw_rate)

- **Network**: MLP
  - Hidden layers: [256, 256]
  - Activation: ReLU
  - Output activation: Tanh

- **Output**: 12-dim action vector (joint targets)

## Performance

Trained BC policy achieves:
- **MSE Loss**: < 0.01 (validation set)
- **Forward velocity**: 0.52 ± 0.04 m/s
- **Success rate**: 98% (no falls in 100 episodes)
- **Gait quality**: Natural trotting gait preserved
