# Demonstration Data

This directory contains raw demonstration data collected from the Unitree Go1 robot.

## Download Demonstration Data

Pre-collected demonstration data is available at:
- **Google Drive**: [Link - TBD]
- **File size**: ~500 MB

## Data Collection

If you have access to a Unitree Go1 robot, collect your own demonstrations:

```bash
python real_robot/collect_demonstrations.py \
    --duration 60 \
    --velocity 0.5 \
    --gait trot \
    --output data/demonstrations/trot_05ms_demo1.npz
```

### Collection Protocol

We collected the following demonstrations:
- **Trot gaits**: 0.3, 0.5, 0.7 m/s (5 minutes each)
- **Walk gaits**: 0.2, 0.4 m/s (3 minutes each)
- **Turns**: Left/right at various speeds (2 minutes each)
- **Total**: ~40 minutes of data

## Data Format

Each `.npz` file contains:
```python
{
    'timestamps': array of timestamps (N,)
    'joint_positions': array of 12 joint angles (N, 12)
    'joint_velocities': array of 12 joint velocities (N, 12)
    'joint_torques': array of 12 joint torques (N, 12)
    'base_position': array of base position [x, y, z] (N, 3)
    'base_orientation': array of base RPY [roll, pitch, yaw] (N, 3)
    'base_velocity': array of base velocity [vx, vy, vz] (N, 3)
    'base_angular_velocity': array of angular velocity (N, 3)
    'foot_contacts': array of foot contact states (N, 4)
    'imu_data': array of IMU readings (N, 9)
}
```

## Loading Data

```python
import numpy as np

# Load demonstration
data = np.load('data/demonstrations/trot_05ms_demo1.npz')

# Access fields
joint_pos = data['joint_positions']
base_vel = data['base_velocity']

print(f"Duration: {len(data['timestamps'])} samples")
print(f"Control frequency: {1.0 / np.mean(np.diff(data['timestamps'])):.1f} Hz")
```

## Quality Control

Demonstrations were filtered to ensure:
- No foot slipping
- Stable, natural gaits
- Consistent velocity
- No collisions or falls
- Flat terrain only

Bad demonstrations were discarded before BC training.

## File Structure

```
data/demonstrations/
├── README.md
├── trot_03ms_demo1.npz
├── trot_03ms_demo2.npz
├── trot_05ms_demo1.npz
├── trot_05ms_demo2.npz
├── trot_07ms_demo1.npz
├── walk_02ms_demo1.npz
├── walk_04ms_demo1.npz
├── turns_left_demo1.npz
└── turns_right_demo1.npz
```

Total samples: ~38,400 at 16 Hz = 40 minutes
