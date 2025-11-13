# Processed BC Dataset

This directory contains processed datasets ready for BC training.

## Generate from Raw Demonstrations

Process raw demonstrations into BC training data:

```bash
python scripts/prepare_bc_dataset.py \
    --input_dir data/demonstrations \
    --output_dir data/processed \
    --val_split 0.15
```

This will create:
- `bc_dataset_train.npz`: Training set (85% of data)
- `bc_dataset_val.npz`: Validation set (15% of data)

## Download Pre-processed Data

Pre-processed BC dataset available at:
- **Google Drive**: [Link - TBD]
- **File size**: ~450 MB

```bash
cd data/processed/
# Download using gdown
gdown --id FILEID --output bc_dataset.zip
unzip bc_dataset.zip
```

## Data Format

Each `.npz` file contains:
```python
{
    'observations': array of state vectors (N, 48)
    'actions': array of action vectors (N, 12)
}
```

### Observation Vector (48-dim)
- [0:12] Joint positions
- [12:24] Joint velocities
- [24:27] Base orientation (roll, pitch, yaw)
- [27:30] IMU angular velocity
- [30] Base height
- [31:33] Base velocity (x, y)
- [33:45] Previous action
- [45:48] Target command (vx, vy, yaw_rate)

### Action Vector (12-dim)
- Target joint angles for position control
- Represents desired joint positions for next timestep

## Dataset Statistics

**Training Set:**
- Samples: ~32,640
- Episodes: Continuous demonstration segments
- Sources: 9 demonstration files
- Data augmentation: None (but mirroring could be added)

**Validation Set:**
- Samples: ~5,760
- Used for monitoring overfitting during training

**Total:** 38,400 state-action pairs from 40 minutes of demonstrations

## Loading Data

```python
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Load training data
data = np.load('data/processed/bc_dataset_train.npz')
obs = data['observations']
actions = data['actions']

print(f"Training samples: {len(obs)}")
print(f"Observation shape: {obs.shape}")
print(f"Action shape: {actions.shape}")

# Create DataLoader
class BCDataset(Dataset):
    def __init__(self, data_file):
        data = np.load(data_file)
        self.obs = data['observations']
        self.actions = data['actions']
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]

train_dataset = BCDataset('data/processed/bc_dataset_train.npz')
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
```

## Data Normalization

Observations and actions are **not normalized** in the saved files. Normalization is applied during training if needed:

```python
# Compute statistics
obs_mean = np.mean(obs, axis=0)
obs_std = np.std(obs, axis=0)

# Normalize
obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
```

Statistics can be saved and reused during inference.
