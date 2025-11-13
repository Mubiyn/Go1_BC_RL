# Task 2: Imitation Learning and RL Fine-tuning for Unitree Go1 Quadruped

## Project Timeline: 4-6 Weeks (4 Team Members)
**Start Date**: After Task 1 completion

---

##  Project Overview

**Goal**: Develop an efficient two-stage learning pipeline for teaching the Unitree Go1 quadruped robot new locomotion skills:
1. **Stage 1: Behavior Cloning (BC)** - Learn natural gaits from demonstrations
2. **Stage 2: Reinforcement Learning (PPO)** - Fine-tune for specific tasks while preserving gait quality

**Key Research Question**: How do we design reward functions that optimize task performance (e.g., speed) while maintaining natural, energy-efficient gaits learned from demonstrations?

**Expected Outcome**: Demonstrate that hybrid BC+RL approach is superior to pure RL from scratch, achieving better task performance without "breaking" natural locomotion patterns.

**Why This Matters**: 
- Training quadruped locomotion from scratch with RL is slow and often produces unnatural gaits
- BC provides a strong starting point with natural movement
- The challenge is fine-tuning without destroying what BC learned
- This is a SOTA approach used by leading robotics labs

---

##  Success Metrics

### Quantitative Metrics
- **Task Performance**: Forward velocity (m/s) for speed task, or turning radius for maneuver task
- **Gait Naturalness**: Foot contact pattern similarity to BC demonstrations (% match)
- **Energy Efficiency**: Total power consumption (sum of joint torques²)
- **Stability**: Number of falls during testing, body orientation variance
- **Learning Efficiency**: Time to reach performance threshold vs. RL from scratch

### Qualitative Metrics
- Visual smoothness of gait (human evaluation)
- Similarity to real dog/Go1 natural walking
- Recovery behavior after disturbances
- Confidence in movement (no hesitation)

### Comparison Baselines
You'll compare **5 approaches**:
1. **BC Only**: Pure imitation, no RL fine-tuning
2. **RL from Scratch**: Pure PPO, no demonstrations
3. **BC + RL (Pure Task Reward)**: Fine-tune BC with only task reward (e.g., speed)
4. **BC + RL (Hybrid Reward - Fixed Weights)**: Your approach with fixed α
5. **BC + RL (Hybrid Reward - Adaptive Weights)**: Your approach with decaying α

---

##  Team Structure & Responsibilities

### **Member 1: Robot Interface & Data Collection Lead**
- Unitree Go1 SDK integration and control
- Demonstration data collection from real robot
- Real robot testing and deployment
- Safety protocols and hardware management

### **Member 2: Simulation & Environment Lead**
- PyBullet simulation setup with Go1 model
- Environment design (terrain, tasks, physics)
- Sim-to-real parameter matching
- Domain randomization implementation

### **Member 3: Behavior Cloning Specialist**
- BC dataset preparation and augmentation
- BC model architecture and training
- BC policy evaluation and analysis
- Demonstration quality assessment

### **Member 4: RL & Reward Engineering Lead**
- PPO fine-tuning implementation
- Reward function design and experimentation
- Gait preservation metrics implementation
- Training monitoring and hyperparameter tuning

**Cross-functional**: Everyone participates in real robot testing and experiments.

---

##  Week-by-Week Breakdown

---

## **WEEK 1: Setup, SDK Integration, and Initial Data Collection**

### **Day 1-2: Environment Setup and SDK Integration (ALL MEMBERS)**

#### **Member 1 Tasks** (Lead)
1. **Unitree Go1 SDK Setup**
   ```bash
   # Clone Unitree SDK
   git clone https://github.com/unitreerobotics/unitree_legged_sdk.git
   cd unitree_legged_sdk
   mkdir build && cd build
   cmake ..
   make
   ```

2. **Test basic robot control**
   ```python
   # test_go1_control.py
   import sys
   sys.path.append('/path/to/unitree_legged_sdk/lib/python')
   from unitree_legged_sdk import *
   
   # Initialize robot
   robot = Robot('Go1')
   robot.connect()
   
   # Test standing up
   robot.stand()
   time.sleep(2)
   
   # Test basic walking command
   robot.move(velocity_x=0.2, velocity_y=0.0, yaw_rate=0.0)
   time.sleep(5)
   
   # Stop and sit
   robot.stop()
   robot.sit()
   ```

3. **Set up data logging system**
   ```python
   # utils/data_logger.py
   import numpy as np
   import json
   from datetime import datetime
   
   class RobotDataLogger:
       def __init__(self, save_dir):
           self.save_dir = save_dir
           self.data = {
               'timestamps': [],
               'joint_positions': [],
               'joint_velocities': [],
               'joint_torques': [],
               'imu_data': [],
               'foot_contacts': [],
               'base_position': [],
               'base_orientation': []
           }
       
       def log_state(self, robot_state):
           """Log current robot state"""
           self.data['timestamps'].append(datetime.now().timestamp())
           self.data['joint_positions'].append(robot_state.q.tolist())
           self.data['joint_velocities'].append(robot_state.dq.tolist())
           self.data['joint_torques'].append(robot_state.tau.tolist())
           # ... log other data
       
       def save(self, filename):
           """Save logged data to file"""
           filepath = os.path.join(self.save_dir, filename)
           np.savez_compressed(filepath, **{k: np.array(v) for k, v in self.data.items()})
           print(f"Saved {len(self.data['timestamps'])} samples to {filepath}")
   ```

4. **Safety setup**
   - Emergency stop procedure (document and test)
   - Safe testing area (padded, clear space)
   - Battery monitoring system
   - Kill switch location and testing

#### **Member 2 Tasks** (Lead)
1. **Install PyBullet and load Go1 model**
   ```bash
   pip install pybullet
   pip install gym
   pip install stable-baselines3[extra]
   ```

2. **Test Go1 simulation**
   ```python
   # test_go1_sim.py
   import pybullet as p
   import pybullet_data
   import time
   
   # Start simulation
   physicsClient = p.connect(p.GUI)
   p.setAdditionalSearchPath(pybullet_data.getDataPath())
   p.setGravity(0, 0, -9.81)
   
   # Load ground and robot
   planeId = p.loadURDF("plane.urdf")
   robotId = p.loadURDF("go1/go1.urdf", [0, 0, 0.5])
   
   # Get joint info
   num_joints = p.getNumJoints(robotId)
   print(f"Go1 has {num_joints} joints")
   for i in range(num_joints):
       info = p.getJointInfo(robotId, i)
       print(f"Joint {i}: {info[1].decode('utf-8')}, Type: {info[2]}")
   
   # Run simulation
   for _ in range(1000):
       p.stepSimulation()
       time.sleep(1./240.)
   ```

3. **Create Go1 gym environment**
   ```python
   # envs/go1_env.py
   import gym
   import numpy as np
   import pybullet as p
   
   class Go1Env(gym.Env):
       def __init__(self, render=False):
           super().__init__()
           
           # Connect to PyBullet
           self.render_mode = render
           if render:
               self.client = p.connect(p.GUI)
           else:
               self.client = p.connect(p.DIRECT)
           
           p.setGravity(0, 0, -9.81)
           p.setTimeStep(1./240.)
           
           # Load robot
           self.plane = p.loadURDF("plane.urdf")
           self.robot = p.loadURDF("go1/go1.urdf", [0, 0, 0.4])
           
           # Define observation space (48-dim)
           # 12 joints × 2 (pos, vel) + 6 IMU (orientation + ang_vel) + 12 (previous action)
           self.observation_space = gym.spaces.Box(
               low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32
           )
           
           # Define action space (12 joint targets)
           self.action_space = gym.spaces.Box(
               low=-1, high=1, shape=(12,), dtype=np.float32
           )
           
           # Joint limits and properties
           self.joint_indices = self._get_motor_joint_indices()
           self.default_joint_angles = self._get_default_stance()
       
       def _get_motor_joint_indices(self):
           """Get indices of actuated joints"""
           motor_joints = []
           for i in range(p.getNumJoints(self.robot)):
               info = p.getJointInfo(self.robot, i)
               if info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                   motor_joints.append(i)
           return motor_joints
       
       def _get_default_stance(self):
           """Default standing position"""
           return np.array([
               0.0, 0.8, -1.6,  # Front right: hip, thigh, calf
               0.0, 0.8, -1.6,  # Front left
               0.0, 0.8, -1.6,  # Rear right
               0.0, 0.8, -1.6   # Rear left
           ])
       
       def reset(self):
           """Reset robot to initial state"""
           # Reset robot position
           p.resetBasePositionAndOrientation(
               self.robot, [0, 0, 0.3], [0, 0, 0, 1]
           )
           
           # Reset joints to default stance
           for i, angle in zip(self.joint_indices, self.default_joint_angles):
               p.resetJointState(self.robot, i, angle, 0)
           
           self.previous_action = np.zeros(12)
           
           return self._get_observation()
       
       def step(self, action):
           """Execute action and return next state"""
           # Scale actions to joint ranges
           target_angles = self.default_joint_angles + action * 0.5
           
           # Apply actions (position control)
           for i, target in zip(self.joint_indices, target_angles):
               p.setJointMotorControl2(
                   self.robot, i,
                   p.POSITION_CONTROL,
                   targetPosition=target,
                   force=25  # Max torque
               )
           
           # Step simulation
           p.stepSimulation()
           
           # Get observation and reward
           obs = self._get_observation()
           reward = self._compute_reward()
           done = self._is_terminal()
           
           self.previous_action = action
           
           return obs, reward, done, {}
       
       def _get_observation(self):
           """Construct observation vector"""
           # Joint states
           joint_states = p.getJointStates(self.robot, self.joint_indices)
           joint_pos = np.array([state[0] for state in joint_states])
           joint_vel = np.array([state[1] for state in joint_states])
           
           # Base orientation and angular velocity
           base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
           base_orn_euler = p.getEulerFromQuaternion(base_orn)
           base_vel, base_ang_vel = p.getBaseVelocity(self.robot)
           
           # Construct observation
           obs = np.concatenate([
               joint_pos,                    # 12
               joint_vel,                    # 12
               base_orn_euler,               # 3
               base_ang_vel,                 # 3
               [base_pos[2]],               # 1 (height)
               base_vel[:2],                # 2 (x, y velocity)
               self.previous_action         # 12
           ])
           
           return obs.astype(np.float32)
       
       def _compute_reward(self):
           """Compute reward (placeholder)"""
           return 0.0
       
       def _is_terminal(self):
           """Check if episode should end"""
           base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
           
           # Terminate if robot falls
           if base_pos[2] < 0.15:  # Height too low
               return True
           
           # Terminate if robot tips over
           orn_euler = p.getEulerFromQuaternion(base_orn)
           if abs(orn_euler[0]) > 0.5 or abs(orn_euler[1]) > 0.5:
               return True
           
           return False
   ```

#### **Member 3 Tasks**
1. **Install ML libraries**
   ```bash
   pip install torch torchvision
   pip install tensorboard
   pip install matplotlib seaborn
   pip install pandas scikit-learn
   ```

2. **Create BC dataset structure**
   ```bash
   mkdir -p data/demonstrations
   mkdir -p data/processed
   mkdir -p models/bc
   mkdir -p models/rl
   ```

3. **Study BC theory**
   - Read about supervised learning for robotics
   - Understand state-action mapping
   - Review data augmentation techniques
   - Study demonstration quality assessment

4. **Create BC training template**
   ```python
   # bc/train_bc.py
   import torch
   import torch.nn as nn
   from torch.utils.data import Dataset, DataLoader
   
   class BCDataset(Dataset):
       """Dataset for behavior cloning"""
       def __init__(self, data_files):
           self.states = []
           self.actions = []
           
           # Load all demonstration files
           for file in data_files:
               data = np.load(file)
               self.states.append(data['observations'])
               self.actions.append(data['actions'])
           
           self.states = np.concatenate(self.states, axis=0)
           self.actions = np.concatenate(self.actions, axis=0)
           
           print(f"Loaded {len(self.states)} state-action pairs")
       
       def __len__(self):
           return len(self.states)
       
       def __getitem__(self, idx):
           return (
               torch.FloatTensor(self.states[idx]),
               torch.FloatTensor(self.actions[idx])
           )
   
   class BCPolicy(nn.Module):
       """Neural network policy for BC"""
       def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
           super().__init__()
           
           layers = []
           input_dim = state_dim
           
           for hidden_dim in hidden_dims:
               layers.append(nn.Linear(input_dim, hidden_dim))
               layers.append(nn.ReLU())
               input_dim = hidden_dim
           
           layers.append(nn.Linear(input_dim, action_dim))
           layers.append(nn.Tanh())  # Output in [-1, 1]
           
           self.network = nn.Sequential(*layers)
       
       def forward(self, state):
           return self.network(state)
   ```

#### **Member 4 Tasks**
1. **Study PPO and reward design**
   - Review PPO algorithm (Stable-Baselines3 docs)
   - Read papers on reward shaping for locomotion
   - Research gait quality metrics
   - Study curriculum learning

2. **Create reward function framework**
   ```python
   # rl/reward_functions.py
   import numpy as np
   
   class RewardCalculator:
       """Compute rewards for RL training"""
       
       def __init__(self, config):
           self.config = config
       
       def compute_task_reward(self, state, action, next_state):
           """Task-specific reward (e.g., forward velocity)"""
           # Extract forward velocity from state
           forward_vel = next_state['base_velocity'][0]
           
           # Reward for moving forward
           velocity_reward = forward_vel * 10.0
           
           return velocity_reward
       
       def compute_gait_preservation_reward(self, state, action, bc_action):
           """Reward for staying close to BC policy"""
           # Action similarity to BC
           action_diff = np.linalg.norm(action - bc_action)
           action_similarity = np.exp(-action_diff * 2.0)
           
           return action_similarity * 5.0
       
       def compute_energy_penalty(self, action, joint_velocities):
           """Penalty for high energy consumption"""
           torques = action  # Assuming action represents torques
           power = np.sum(np.abs(torques * joint_velocities))
           return -power * 0.01
       
       def compute_stability_reward(self, state):
           """Reward for maintaining stable orientation"""
           roll, pitch, yaw = state['base_orientation']
           orientation_penalty = -(abs(roll) + abs(pitch)) * 2.0
           
           height = state['base_position'][2]
           height_penalty = -abs(height - 0.28) * 10.0  # Maintain nominal height
           
           return orientation_penalty + height_penalty
       
       def compute_total_reward(self, state, action, next_state, bc_action, weights):
           """Combine all reward components"""
           task_reward = self.compute_task_reward(state, action, next_state)
           gait_reward = self.compute_gait_preservation_reward(state, action, bc_action)
           energy_penalty = self.compute_energy_penalty(action, state['joint_velocities'])
           stability_reward = self.compute_stability_reward(next_state)
           
           total = (weights['task'] * task_reward +
                   weights['gait'] * gait_reward +
                   weights['energy'] * energy_penalty +
                   weights['stability'] * stability_reward)
           
           return total, {
               'task': task_reward,
               'gait': gait_reward,
               'energy': energy_penalty,
               'stability': stability_reward
           }
   ```

3. **Set up experiment tracking**
   ```bash
   mkdir -p experiments/logs
   mkdir -p experiments/videos
   mkdir -p experiments/checkpoints
   ```

**Deliverable Week 1**: 
- Unitree Go1 SDK working and tested
- PyBullet simulation running with Go1 model
- Data logging system ready
- Code repository structure established

---

### **Day 3-5: Demonstration Data Collection (Member 1 Lead, ALL Participate)**

#### **Member 1 Tasks** (Lead)
1. **Plan demonstration collection strategy**
   - **Trot gait**: 0.3, 0.5, 0.7 m/s forward speeds
   - **Walk gait**: 0.2, 0.4 m/s forward speeds  
   - **Turning**: Left and right turns at various radii
   - **Transitions**: Stand → Walk → Trot → Stand
   
2. **Collect demonstrations using Go1's built-in gaits**
   ```python
   # collect_demonstrations.py
   from data_logger import RobotDataLogger
   import time
   
   # Initialize robot and logger
   robot = Robot('Go1')
   robot.connect()
   logger = RobotDataLogger('data/demonstrations')
   
   # Demo 1: Trot at 0.5 m/s
   print("Collecting trot at 0.5 m/s...")
   robot.stand()
   time.sleep(2)
   
   # Start logging
   logger.start()
   
   # Command robot
   for _ in range(500):  # ~30 seconds at 16Hz
       robot.move(velocity_x=0.5, velocity_y=0.0, yaw_rate=0.0)
       state = robot.get_state()
       logger.log_state(state)
       time.sleep(1./16.)  # 16Hz control rate
   
   robot.stop()
   logger.save('trot_05ms_demo1.npz')
   
   # Repeat for other gaits and speeds...
   ```

3. **Quality control during collection**
   - Check for foot slipping (discard if excessive)
   - Ensure smooth, stable gaits
   - Verify data integrity (no missing values)
   - Maintain consistent environment (flat ground)

#### **All Members Tasks**
- **Member 1**: Operate robot and SDK
- **Member 2**: Monitor data quality, spot irregularities
- **Member 3**: Record video of demonstrations
- **Member 4**: Take notes on gait characteristics

**Data Collection Goals**:
- **Minimum**: 20 minutes total (19,200 samples at 16Hz)
- **Target**: 40 minutes total (38,400 samples)
- **Variety**: Multiple speeds, directions, and gaits

4. **Process raw data into BC dataset**
   ```python
   # bc/prepare_dataset.py
   import numpy as np
   
   def process_demonstration_file(raw_file):
       """Convert raw robot data to state-action pairs"""
       data = np.load(raw_file)
       
       # Extract features for observation
       observations = []
       actions = []
       
       for t in range(len(data['timestamps']) - 1):
           # Current state
           obs = np.concatenate([
               data['joint_positions'][t],      # 12
               data['joint_velocities'][t],     # 12
               data['base_orientation'][t],     # 3 (roll, pitch, yaw)
               data['imu_angular_velocity'][t], # 3
               [data['base_position'][t][2]],  # 1 (height)
               data['base_velocity'][t][:2],   # 2 (x, y vel)
               actions[-1] if len(actions) > 0 else np.zeros(12)  # previous action
           ])
           
           # Next action (next joint positions as targets)
           # This assumes position control
           action = data['joint_positions'][t + 1]
           
           observations.append(obs)
           actions.append(action)
       
       return np.array(observations), np.array(actions)
   
   # Process all demonstrations
   all_obs = []
   all_actions = []
   
   demo_files = glob.glob('data/demonstrations/*.npz')
   for file in demo_files:
       obs, actions = process_demonstration_file(file)
       all_obs.append(obs)
       all_actions.append(actions)
       print(f"Processed {file}: {len(obs)} samples")
   
   # Combine and save
   all_obs = np.concatenate(all_obs, axis=0)
   all_actions = np.concatenate(all_actions, axis=0)
   
   # Split train/val
   from sklearn.model_selection import train_test_split
   train_obs, val_obs, train_act, val_act = train_test_split(
       all_obs, all_actions, test_size=0.15, random_state=42
   )
   
   np.savez('data/processed/bc_dataset_train.npz',
            observations=train_obs, actions=train_act)
   np.savez('data/processed/bc_dataset_val.npz',
            observations=val_obs, actions=val_act)
   
   print(f"Train: {len(train_obs)} samples")
   print(f"Val: {len(val_obs)} samples")
   ```

**Deliverable Days 3-5**:
- 20-40 minutes of demonstration data collected
- Processed BC dataset (train/val split)
- Video recordings of demonstrations
- Data quality report

---

## **WEEK 2: Behavior Cloning Training and Baseline RL**

### **Day 6-8: BC Model Training (Member 3 Lead)**

#### **Member 3 Tasks** (Primary)
1. **Implement BC training loop**
   ```python
   # bc/train_bc.py (continued)
   
   def train_bc_policy():
       # Load data
       train_dataset = BCDataset(['data/processed/bc_dataset_train.npz'])
       val_dataset = BCDataset(['data/processed/bc_dataset_val.npz'])
       
       train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
       val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
       
       # Create model
       policy = BCPolicy(state_dim=48, action_dim=12, hidden_dims=[256, 256])
       policy = policy.cuda()
       
       # Optimizer and loss
       optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
       criterion = nn.MSELoss()
       
       # Training loop
       best_val_loss = float('inf')
       
       for epoch in range(100):
           # Training
           policy.train()
           train_losses = []
           
           for states, actions in train_loader:
               states, actions = states.cuda(), actions.cuda()
               
               # Forward pass
               predicted_actions = policy(states)
               loss = criterion(predicted_actions, actions)
               
               # Backward pass
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               
               train_losses.append(loss.item())
           
           # Validation
           policy.eval()
           val_losses = []
           
           with torch.no_grad():
               for states, actions in val_loader:
                   states, actions = states.cuda(), actions.cuda()
                   predicted_actions = policy(states)
                   loss = criterion(predicted_actions, actions)
                   val_losses.append(loss.item())
           
           train_loss = np.mean(train_losses)
           val_loss = np.mean(val_losses)
           
           print(f"Epoch {epoch+1}/100 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
           
           # Save best model
           if val_loss < best_val_loss:
               best_val_loss = val_loss
               torch.save(policy.state_dict(), 'models/bc/best_policy.pth')
               print("Saved new best model!")
       
       return policy
   
   if __name__ == '__main__':
       policy = train_bc_policy()
   ```

2. **Monitor training**
   - Plot training and validation loss curves
   - Check for overfitting (val loss increasing while train decreases)
   - Verify loss is decreasing (should reach <0.01 MSE)

3. **Data augmentation (if needed)**
   ```python
   def augment_demonstration(obs, action):
       """Apply data augmentation"""
       # Mirror left-right (flip y-axis movements)
       if np.random.rand() < 0.5:
           obs_aug = obs.copy()
           action_aug = action.copy()
           
           # Swap left-right legs (indices depend on robot config)
           # Example: swap indices [0,1,2] with [3,4,5] and [6,7,8] with [9,10,11]
           obs_aug[[0,1,2, 3,4,5, 6,7,8, 9,10,11]] = obs[[3,4,5, 0,1,2, 9,10,11, 6,7,8]]
           action_aug[[0,1,2, 3,4,5, 6,7,8, 9,10,11]] = action[[3,4,5, 0,1,2, 9,10,11, 6,7,8]]
           
           return obs_aug, action_aug
       
       return obs, action
   ```

#### **Member 2 Tasks** (Support)
1. **Test BC policy in simulation**
   ```python
   # test_bc_sim.py
   import torch
   from envs.go1_env import Go1Env
   from bc.train_bc import BCPolicy
   
   # Load trained BC policy
   policy = BCPolicy(state_dim=48, action_dim=12)
   policy.load_state_dict(torch.load('models/bc/best_policy.pth'))
   policy.eval()
   
   # Test in simulation
   env = Go1Env(render=True)
   obs = env.reset()
   
   total_reward = 0
   for step in range(1000):
       # Get action from BC policy
       with torch.no_grad():
           obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
           action = policy(obs_tensor).squeeze(0).numpy()
       
       # Execute action
       obs, reward, done, info = env.step(action)
       total_reward += reward
       
       if done:
           print(f"Episode ended at step {step}")
           break
   
   print(f"Total reward: {total_reward}")
   ```

2. **Evaluate BC policy quantitatively**
   - Does robot stand/walk without falling?
   - Forward velocity achieved
   - Gait naturalness (visual inspection)
   - Compare to demonstration videos

#### **Member 4 Tasks** (Support)
- Help monitor training
- Run BC evaluations in simulation
- Record metrics: success rate, avg velocity, stability

#### **Member 1 Tasks** (Support)
- Prepare for real robot BC testing
- Ensure robot is ready for deployment

**Deliverable Day 8**:
- Trained BC policy with low MSE loss (<0.01)
- BC working in simulation (robot walks naturally)
- Quantitative evaluation results

---

### **Day 9-11: Baseline RL from Scratch (Member 4 Lead)**

#### **Member 4 Tasks** (Primary)
1. **Set up RL training environment**
   ```python
   # rl/train_rl_from_scratch.py
   from stable_baselines3 import PPO
   from stable_baselines3.common.vec_env import SubprocVecEnv
   from envs.go1_env import Go1Env
   
   def make_env(rank):
       def _init():
           env = Go1Env(render=False)
           return env
       return _init
   
   # Create parallel environments
   n_envs = 16
   env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
   
   # Create PPO model
   model = PPO(
       "MlpPolicy",
       env,
       learning_rate=3e-4,
       n_steps=2048,
       batch_size=64,
       n_epochs=10,
       gamma=0.99,
       gae_lambda=0.95,
       clip_range=0.2,
       verbose=1,
       tensorboard_log="./experiments/logs/rl_scratch/"
   )
   
   # Train
   model.learn(total_timesteps=10_000_000)
   model.save("models/rl/rl_from_scratch")
   ```

2. **Implement reward function for RL**
   ```python
   # In Go1Env._compute_reward():
   
   def _compute_reward(self):
       """Reward function for RL from scratch"""
       base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
       base_vel, base_ang_vel = p.getBaseVelocity(self.robot)
       
       # Forward velocity reward
       forward_vel = base_vel[0]
       velocity_reward = forward_vel * 10.0
       
       # Stability penalty
       orn_euler = p.getEulerFromQuaternion(base_orn)
       roll, pitch, yaw = orn_euler
       stability_penalty = -(abs(roll) + abs(pitch)) * 2.0
       
       # Height penalty (maintain nominal height)
       height_penalty = -abs(base_pos[2] - 0.28) * 10.0
       
       # Energy penalty
       joint_states = p.getJointStates(self.robot, self.joint_indices)
       joint_torques = np.array([state[3] for state in joint_states])
       energy_penalty = -np.sum(np.abs(joint_torques)) * 0.001
       
       # Alive bonus
       alive_bonus = 0.1
       
       total_reward = (velocity_reward + stability_penalty + 
                      height_penalty + energy_penalty + alive_bonus)
       
       return total_reward
   ```

3. **Monitor RL training**
   ```bash
   tensorboard --logdir=./experiments/logs/rl_scratch/
   ```
   - Watch episode reward (should increase)
   - Monitor episode length (should increase initially)
   - Check for policy collapse (sudden drop in performance)

**Note**: RL from scratch typically takes 5-10M timesteps and may produce unnatural gaits. This is expected and serves as your baseline comparison.

#### **Member 2 Tasks** (Support)
- Help tune reward function if RL struggles
- Adjust environment parameters if robot falls too quickly
- Implement curriculum learning if needed (start easy, increase difficulty)

#### **Member 3 Tasks** (Support)
- Compare RL behavior to BC demonstrations
- Document differences in gait quality
- Help evaluate intermediate checkpoints

**Deliverable Day 11**:
- RL from scratch trained for 5-10M timesteps
- Baseline performance metrics
- Visual comparison: RL vs BC gaits
- Documentation of RL gait characteristics

---

## **WEEK 3: RL Fine-tuning with Hybrid Rewards**

### **Day 12-14: Pure Task Reward Fine-tuning (Member 4 Lead)**

#### **Member 4 Tasks** (Primary)
1. **Implement BC-initialized RL training**
   ```python
   # rl/train_rl_finetune.py
   from stable_baselines3 import PPO
   from stable_baselines3.common.policies import ActorCriticPolicy
   import torch
   
   # Load BC policy weights to initialize RL
   bc_policy_path = 'models/bc/best_policy.pth'
   bc_weights = torch.load(bc_policy_path)
   
   # Create PPO model
   model = PPO(
       "MlpPolicy",
       env,
       learning_rate=3e-4,
       # ... other hyperparameters
   )
   
   # Initialize actor network with BC weights
   # (This requires matching architectures)
   actor_state_dict = model.policy.mlp_extractor.state_dict()
   
   # Map BC weights to PPO actor
   # Assuming BC has same architecture as PPO's actor
   for key in bc_weights.keys():
       if key in actor_state_dict:
           actor_state_dict[key] = bc_weights[key]
   
   model.policy.mlp_extractor.load_state_dict(actor_state_dict)
   print("Initialized PPO with BC weights!")
   
   # Fine-tune (Experiment 1: Pure task reward)
   model.learn(total_timesteps=3_000_000)
   model.save("models/rl/bc_finetune_task_only")
   ```

2. **Run Experiment 1: BC + RL (Task Reward Only)**
   - Use only forward velocity reward
   - No gait preservation terms
   - This will likely break the natural gait but achieve high speed
   - Document: final velocity, gait quality, energy consumption

#### **All Members**
- Observe training progress
- Note when gait starts degrading
- Record videos at checkpoints (0, 1M, 2M, 3M steps)

**Deliverable Day 14**:
- BC fine-tuned with pure task reward
- Performance metrics
- Videos showing gait degradation
- Analysis of trade-off: speed vs. naturalness

---

### **Day 15-21: Hybrid Reward Experiments (Member 4 Lead, All Support)**

This is the **core research contribution** - testing different reward strategies.

#### **Member 4 Tasks** (Primary)
1. **Experiment 2: Fixed Weight Hybrid Reward**
   ```python
   # rl/hybrid_reward_env.py
   
   class HybridRewardGo1Env(Go1Env):
       def __init__(self, bc_policy, alpha_task=0.7, alpha_gait=0.3):
           super().__init__()
           self.bc_policy = bc_policy
           self.alpha_task = alpha_task
           self.alpha_gait = alpha_gait
       
       def _compute_reward(self):
           """Hybrid reward: task + gait preservation"""
           # Task reward (forward velocity)
           base_vel, _ = p.getBaseVelocity(self.robot)
           forward_vel = base_vel[0]
           task_reward = forward_vel * 10.0
           
           # Gait preservation reward
           current_obs = self._get_observation()
           
           with torch.no_grad():
               obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0)
               bc_action = self.bc_policy(obs_tensor).squeeze(0).numpy()
           
           # Compare current action to BC action
           action_diff = np.linalg.norm(self.current_action - bc_action)
           gait_reward = np.exp(-action_diff * 2.0) * 10.0
           
           # Combine with fixed weights
           total_reward = (self.alpha_task * task_reward + 
                          self.alpha_gait * gait_reward)
           
           return total_reward
   ```

2. **Run multiple weight combinations**
   - **Variant A**: α_task=0.7, α_gait=0.3
   - **Variant B**: α_task=0.5, α_gait=0.5
   - **Variant C**: α_task=0.3, α_gait=0.7
   
   Train each for 3M timesteps

3. **Experiment 3: Adaptive Weight Hybrid Reward**
   ```python
   class AdaptiveHybridRewardEnv(Go1Env):
       def __init__(self, bc_policy, alpha_decay=0.99999):
           super().__init__()
           self.bc_policy = bc_policy
           self.alpha_gait = 0.9  # Start high (preserve gait)
           self.alpha_task = 0.1  # Start low
           self.alpha_decay = alpha_decay
           self.step_count = 0
       
       def _compute_reward(self):
           """Adaptive reward: gradually shift from gait to task"""
           # Compute both reward components
           task_reward = self._compute_task_reward()
           gait_reward = self._compute_gait_reward()
           
           # Combine with adaptive weights
           total_reward = (self.alpha_task * task_reward + 
                          self.alpha_gait * gait_reward)
           
           # Decay gait weight, increase task weight
           self.step_count += 1
           if self.step_count % 1000 == 0:  # Update every 1000 steps
               self.alpha_gait = max(0.1, self.alpha_gait * self.alpha_decay)
               self.alpha_task = 1.0 - self.alpha_gait
               print(f"Step {self.step_count}: alpha_task={self.alpha_task:.3f}, alpha_gait={self.alpha_gait:.3f}")
           
           return total_reward
   ```

4. **Alternative gait preservation metrics to test**
   ```python
   # Option 1: Action similarity (already implemented)
   gait_reward_action = np.exp(-np.linalg.norm(action - bc_action) * 2.0)
   
   # Option 2: Foot contact pattern similarity
   def compute_foot_contact_reward(current_contacts, bc_contacts):
       """Reward for matching foot contact pattern"""
       contact_match = np.sum(current_contacts == bc_contacts) / 4.0
       return contact_match * 10.0
   
   # Option 3: Joint velocity smoothness
   def compute_smoothness_reward(joint_velocities, prev_joint_velocities):
       """Reward for smooth joint motions"""
       jerk = np.linalg.norm(joint_velocities - prev_joint_velocities)
       return np.exp(-jerk * 0.5) * 5.0
   
   # Option 4: KL divergence between policies
   def compute_kl_divergence_penalty(rl_logprobs, bc_logprobs):
       """Penalize deviation from BC policy distribution"""
       kl_div = torch.sum(torch.exp(bc_logprobs) * (bc_logprobs - rl_logprobs))
       return -kl_div * 0.1
   ```

#### **All Members Tasks**
- **Member 1**: Monitor training jobs, ensure no crashes
- **Member 2**: Evaluate checkpoints in simulation
- **Member 3**: Compare gaits visually, assess naturalness
- **Member 4**: Track metrics, plot learning curves

**Experiment Matrix** (7 total):
1. BC Only (no RL)
2. RL from Scratch
3. BC + RL (Task Only)
4. BC + RL (Hybrid 0.7/0.3)
5. BC + RL (Hybrid 0.5/0.5)
6. BC + RL (Hybrid 0.3/0.7)
7. BC + RL (Adaptive)

**Deliverable Week 3**:
- 7 trained policies
- Training curves for all experiments
- Preliminary comparison results

---

## **WEEK 4: Comprehensive Testing and Analysis**

### **Day 22-24: Simulation Testing (Member 3 & 4 Lead)**

#### **Member 3 & 4 Tasks** (Primary)
1. **Automated testing framework**
   ```python
   # testing/test_all_policies.py
   import json
   import numpy as np
   from envs.go1_env import Go1Env
   
   def test_policy(policy_name, policy, n_episodes=10):
       """Test a policy and collect metrics"""
       env = Go1Env(render=False)
       
       results = {
           'velocities': [],
           'episode_lengths': [],
           'falls': 0,
           'energy_consumption': [],
           'gait_smoothness': []
       }
       
       for episode in range(n_episodes):
           obs = env.reset()
           done = False
           step = 0
           episode_energy = 0
           velocities = []
           
           while not done and step < 1000:
               # Get action from policy
               action = policy.predict(obs)[0]
               
               # Step environment
               obs, reward, done, info = env.step(action)
               
               # Collect metrics
               velocities.append(info['forward_velocity'])
               episode_energy += info['energy']
               step += 1
           
           # Record episode metrics
           results['velocities'].append(np.mean(velocities))
           results['episode_lengths'].append(step)
           results['energy_consumption'].append(episode_energy)
           if done and step < 1000:
               results['falls'] += 1
       
       # Aggregate results
       summary = {
           'policy': policy_name,
           'avg_velocity': np.mean(results['velocities']),
           'std_velocity': np.std(results['velocities']),
           'avg_episode_length': np.mean(results['episode_lengths']),
           'fall_rate': results['falls'] / n_episodes,
           'avg_energy': np.mean(results['energy_consumption']),
           'success_rate': 1.0 - (results['falls'] / n_episodes)
       }
       
       return summary, results
   
   # Test all policies
   policies = {
       'BC_Only': load_bc_policy('models/bc/best_policy.pth'),
       'RL_Scratch': PPO.load('models/rl/rl_from_scratch'),
       'BC_RL_Task': PPO.load('models/rl/bc_finetune_task_only'),
       'BC_RL_Hybrid_73': PPO.load('models/rl/bc_hybrid_0.7_0.3'),
       'BC_RL_Hybrid_55': PPO.load('models/rl/bc_hybrid_0.5_0.5'),
       'BC_RL_Hybrid_37': PPO.load('models/rl/bc_hybrid_0.3_0.7'),
       'BC_RL_Adaptive': PPO.load('models/rl/bc_adaptive')
   }
   
   all_results = {}
   for name, policy in policies.items():
       print(f"Testing {name}...")
       summary, detailed = test_policy(name, policy, n_episodes=20)
       all_results[name] = summary
       print(f"  Avg Velocity: {summary['avg_velocity']:.3f} m/s")
       print(f"  Success Rate: {summary['success_rate']:.2%}")
   
   # Save results
   with open('results/simulation_test_results.json', 'w') as f:
       json.dump(all_results, f, indent=2)
   ```

2. **Gait quality assessment**
   ```python
   # testing/gait_analysis.py
   
   def analyze_gait_quality(policy, bc_policy, n_episodes=5):
       """Quantitative gait quality metrics"""
       env = Go1Env(render=False)
       
       metrics = {
           'action_similarity_to_bc': [],
           'foot_contact_regularity': [],
           'joint_smoothness': [],
           'energy_efficiency': []
       }
       
       for episode in range(n_episodes):
           obs = env.reset()
           done = False
           
           prev_joints = None
           step = 0
           
           while not done and step < 500:
               # Get RL action
               rl_action = policy.predict(obs)[0]
               
               # Get BC action for comparison
               with torch.no_grad():
                   bc_action = bc_policy(torch.FloatTensor(obs).unsqueeze(0))
                   bc_action = bc_action.squeeze(0).numpy()
               
               # Action similarity
               action_diff = np.linalg.norm(rl_action - bc_action)
               metrics['action_similarity_to_bc'].append(
                   np.exp(-action_diff * 2.0)
               )
               
               # Joint smoothness (low jerk)
               if prev_joints is not None:
                   jerk = np.linalg.norm(obs[12:24] - prev_joints)
                   metrics['joint_smoothness'].append(1.0 / (1.0 + jerk))
               
               prev_joints = obs[12:24].copy()
               
               # Step
               obs, reward, done, info = env.step(rl_action)
               metrics['energy_efficiency'].append(info.get('energy', 0))
               step += 1
       
       # Aggregate
       summary = {
           'action_similarity': np.mean(metrics['action_similarity_to_bc']),
           'smoothness_score': np.mean(metrics['joint_smoothness']),
           'avg_energy_per_step': np.mean(metrics['energy_efficiency'])
       }
       
       return summary
   ```

3. **Create comparison videos**
   ```python
   # testing/record_videos.py
   import imageio
   
   def record_policy_video(policy, filename, n_steps=500):
       """Record video of policy execution"""
       env = Go1Env(render=True)
       obs = env.reset()
       
       frames = []
       
       for step in range(n_steps):
           # Render and capture frame
           frame = env.render(mode='rgb_array')
           frames.append(frame)
           
           # Execute policy
           action = policy.predict(obs)[0]
           obs, reward, done, info = env.step(action)
           
           if done:
               break
       
       # Save video
       imageio.mimsave(filename, frames, fps=30)
       print(f"Saved video to {filename}")
   
   # Record all policies
   for name, policy in policies.items():
       record_policy_video(policy, f'results/videos/{name}.mp4')
   ```

#### **Member 1 & 2 Tasks** (Support)
- Help run tests in parallel
- Review videos and provide qualitative assessment
- Identify interesting behaviors to highlight

**Deliverable Day 24**:
- Complete simulation test results (all 7 policies)
- Gait quality metrics
- Comparison videos

---

### **Day 25-28: Real Robot Testing (ALL MEMBERS)**

#### **Safety Protocol (Member 1 Lead)**
1. **Pre-deployment checks**
   - Battery >70% charge
   - Emergency stop button accessible
   - Padded testing area prepared
   - Tether attached (for safety)
   - Camera setup for recording

2. **Progressive deployment strategy**
   - Day 1: BC Only (safest baseline)
   - Day 2: Best hybrid policies (2-3 selected)
   - Day 3: Comparison and additional tests
   - Day 4: Repeat tests and collect more data

#### **Testing Procedure**

**Test 1: Straight Walk (5 trials per policy)**
```python
# real_robot/test_straight_walk.py

def test_straight_walk(policy, duration=10.0):
    """Test forward walking on real robot"""
    robot = Robot('Go1')
    robot.connect()
    robot.stand()
    time.sleep(2)
    
    # Start logging
    logger = RobotDataLogger('results/real_robot/')
    logger.start()
    
    # Execute policy
    start_pos = robot.get_position()
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Get observation
        state = robot.get_state()
        obs = construct_observation(state)
        
        # Get action from policy
        action = policy.predict(obs)[0]
        
        # Send to robot
        robot.set_joint_targets(action)
        logger.log_state(state)
        
        time.sleep(1./16.)  # 16Hz control
    
    # Stop robot
    robot.stop()
    end_pos = robot.get_position()
    
    # Compute metrics
    distance = np.linalg.norm(end_pos - start_pos)
    avg_velocity = distance / duration
    
    logger.save(f'straight_walk_{policy_name}_{trial}.npz')
    
    return {
        'distance': distance,
        'avg_velocity': avg_velocity,
        'duration': duration
    }
```

**Test 2: Obstacle Course (if time permits)**
- Cones placed at intervals
- Navigate around obstacles
- Measure completion time and falls

**Test 3: Energy Comparison**
- Run each policy for fixed duration
- Measure battery consumption
- Compare energy efficiency

**Test 4: Gait Visual Assessment**
- Record high-quality videos from multiple angles
- Rate naturalness on scale 1-10 (human judges)
- Compare foot placement patterns to BC demonstrations

#### **All Members Roles**
- **Member 1**: Robot operator, safety monitor
- **Member 2**: Data logging, computer control
- **Member 3**: Video recording, note-taking
- **Member 4**: Metrics calculation, real-time analysis

#### **Data Collection**
For each policy:
- 5 straight walk trials
- 3 turn trials (left/right)
- 2 longer duration tests (30 seconds)
- Video from 3 angles
- Battery consumption data

**Deliverable Day 28**:
- Real robot test data for top 3-4 policies
- Videos of real robot execution
- Comparison: sim vs. real performance
- Qualitative assessment of gait quality

---

## **WEEK 5-6: Analysis, Writing, and Presentation**

### **Day 29-32: Data Analysis and Visualization (Member 4 Lead)**

#### **Member 4 Tasks** (Primary)
1. **Create comprehensive comparison plots**
   ```python
   # analysis/plot_all_results.py
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Plot 1: Velocity comparison
   fig, ax = plt.subplots(figsize=(10, 6))
   policies = list(results.keys())
   velocities = [results[p]['avg_velocity'] for p in policies]
   errors = [results[p]['std_velocity'] for p in policies]
   
   x = np.arange(len(policies))
   ax.bar(x, velocities, yerr=errors, capsize=5)
   ax.set_xticks(x)
   ax.set_xticklabels(policies, rotation=45, ha='right')
   ax.set_ylabel('Average Velocity (m/s)')
   ax.set_title('Task Performance: Forward Velocity')
   plt.tight_layout()
   plt.savefig('results/figures/velocity_comparison.png', dpi=300)
   
   # Plot 2: Pareto frontier (velocity vs. gait quality)
   fig, ax = plt.subplots(figsize=(8, 8))
   for policy in policies:
       vel = results[policy]['avg_velocity']
       gait = gait_quality[policy]['action_similarity']
       ax.scatter(vel, gait, s=200, label=policy, alpha=0.7)
   
   ax.set_xlabel('Forward Velocity (m/s)')
   ax.set_ylabel('Gait Similarity to BC')
   ax.set_title('Pareto Frontier: Performance vs. Gait Quality')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.savefig('results/figures/pareto_frontier.png', dpi=300)
   
   # Plot 3: Training curves
   # Plot 4: Energy efficiency comparison
   # Plot 5: Success rate comparison
   # Plot 6: Real vs. sim performance
   ```

2. **Statistical analysis**
   ```python
   from scipy import stats
   
   # T-test: BC+RL Hybrid vs. BC Only
   hybrid_vel = hybrid_results['velocities']
   bc_vel = bc_results['velocities']
   t_stat, p_value = stats.ttest_ind(hybrid_vel, bc_vel)
   
   print(f"Hybrid vs BC: t={t_stat:.3f}, p={p_value:.4f}")
   
   # Effect size (Cohen's d)
   pooled_std = np.sqrt((np.std(hybrid_vel)**2 + np.std(bc_vel)**2) / 2)
   cohens_d = (np.mean(hybrid_vel) - np.mean(bc_vel)) / pooled_std
   print(f"Effect size: d={cohens_d:.3f}")
   ```

3. **Create summary tables**
   ```python
   import pandas as pd
   
   # Summary table
   summary_df = pd.DataFrame({
       'Policy': policies,
       'Velocity (m/s)': [results[p]['avg_velocity'] for p in policies],
       'Success Rate': [results[p]['success_rate'] for p in policies],
       'Gait Quality': [gait_quality[p]['action_similarity'] for p in policies],
       'Energy (J)': [results[p]['avg_energy'] for p in policies]
   })
   
   summary_df = summary_df.round(3)
   summary_df.to_csv('results/summary_table.csv', index=False)
   print(summary_df.to_latex(index=False))
   ```

#### **All Members Tasks**
- Review figures and suggest improvements
- Interpret results and identify key findings
- Prepare data for report and presentation

**Key Findings to Extract**:
1. How much faster is BC+RL than BC only?
2. Which reward balance (task/gait) is optimal?
3. Does adaptive weighting outperform fixed?
4. How much better is BC+RL than RL from scratch?
5. What is the sim-to-real gap?
6. Visual gait quality assessment

**Deliverable Day 32**:
- 10-15 publication-quality figures
- Statistical analysis summary
- Summary tables
- Key findings documented

---

### **Day 33-38: Report Writing (ALL MEMBERS)**

#### **Report Structure** (20-25 pages)

**Section 1: Introduction** (Member 2, 2-3 pages)
- Problem statement: Why is quadruped locomotion learning hard?
- Motivation: Why BC+RL hybrid approach?
- Research question: Reward function design for gait preservation
- Contributions: What did you discover?

**Section 2: Related Work** (Member 3, 2-3 pages)
- Reinforcement learning for locomotion
- Imitation learning and behavior cloning
- Hybrid approaches (residual RL, learning from demonstrations)
- Reward shaping and multi-objective optimization
- Prior work on Unitree Go1

**Section 3: Methods** (Member 1 & 4, 6-8 pages)

**3.1 Hardware Platform** (Member 1)
- Unitree Go1 specifications
- Sensors and actuators
- SDK and control interface

**3.2 Behavior Cloning** (Member 3)
- Demonstration collection procedure
- Dataset statistics
- BC network architecture
- Training procedure and hyperparameters

**3.3 Reinforcement Learning Fine-tuning** (Member 4)
- PPO algorithm overview
- Environment design (observation/action spaces)
- Reward function variants:
  - Pure task reward
  - Hybrid reward (fixed weights)
  - Adaptive hybrid reward
- Gait preservation metrics
- Training procedure

**3.4 Evaluation Methodology** (Member 4)
- Simulation testing protocol
- Real robot testing protocol
- Metrics definition

**Section 4: Results** (Member 4, 5-6 pages)

**4.1 Simulation Results**
- Training curves
- Quantitative comparison (table + plots)
- Pareto frontier analysis
- Gait quality assessment

**4.2 Real Robot Results**
- Deployment success
- Performance comparison
- Sim-to-real gap analysis
- Qualitative gait assessment (with video stills)

**4.3 Ablation Studies**
- Effect of reward weights
- Adaptive vs. fixed weighting
- Different gait preservation metrics

**Section 5: Discussion** (ALL, 2-3 pages)
- Interpretation of results
- Why does hybrid reward work?
- Optimal balance between task and gait
- Limitations of approach
- Failure cases and lessons learned
- Sim-to-real transfer challenges

**Section 6: Conclusion** (Member 2, 1 page)
- Summary of key findings
- Practical recommendations
- Future work

**References** (ALL)
- 20-30 references

**Appendix**
- Hyperparameters
- Additional figures
- Code snippets

#### **Writing Schedule**
- Day 33-34: Draft sections 1-3
- Day 35-36: Draft sections 4-5
- Day 37: Draft section 6, integrate all
- Day 38: Revisions and polish

---

### **Day 39-42: Presentation Preparation (ALL MEMBERS)**

#### **Presentation Structure** (20 minutes + 5 min Q&A)

**Slides** (15-18 slides):
1. Title slide
2. Motivation: Why quadruped learning is hard
3. Problem statement
4. Approach overview (2-stage pipeline diagram)
5. Stage 1: Behavior Cloning (demo video)
6. Stage 2: RL Fine-tuning
7. Research question: Reward design
8. Experimental setup
9. Results 1: Training curves
10. Results 2: Velocity comparison
11. Results 3: Pareto frontier
12. Results 4: Real robot (video)
13. Key findings
14. Discussion: Optimal reward balance
15. Conclusions
16. Future work
17. Thank you + Q&A

**Demo Video** (3-4 minutes):
- BC demonstrations collection
- BC policy in simulation
- RL from scratch (bad gait)
- BC+RL hybrid (good gait + fast)
- Real robot deployment (side-by-side)

#### **Member Roles**
- **Member 1**: Hardware/setup slides (5 min)
- **Member 2**: Introduction/background (3 min)
- **Member 3**: BC methodology (3 min)
- **Member 4**: RL/results (7 min)
- **ALL**: Q&A preparation

#### **Rehearsal Schedule**
- Day 39: Create slides
- Day 40: Record demo video
- Day 41: Practice presentation (internal)
- Day 42: Final rehearsal, polish slides

**Deliverable Day 42**:
- Polished presentation slides
- Demo video
- Speaker notes
- Anticipated Q&A answers

---

##  Final Deliverables

### **1. Code Repository**
```
project/
├── README.md
├── requirements.txt
├── data/
│   ├── demonstrations/
│   └── processed/
├── envs/
│   └── go1_env.py
├── bc/
│   ├── train_bc.py
│   └── prepare_dataset.py
├── rl/
│   ├── train_rl_from_scratch.py
│   ├── train_rl_finetune.py
│   └── reward_functions.py
├── testing/
│   ├── test_all_policies.py
│   └── gait_analysis.py
├── real_robot/
│   ├── collect_demonstrations.py
│   └── test_real_robot.py
├── analysis/
│   └── plot_all_results.py
├── models/
│   ├── bc/
│   └── rl/
└── results/
    ├── figures/
    ├── videos/
    └── data/
```

### **2. Trained Models**
- BC policy (best_policy.pth)
- 7 RL policies (all experiment variants)
- Checkpoints at key training steps

### **3. Experimental Data**
- Demonstration dataset (20-40 min)
- Simulation test results (JSON)
- Real robot test data
- Training logs (TensorBoard)

### **4. Results**
- 10-15 figures (publication quality)
- Comparison videos (sim + real)
- Statistical analysis
- Summary tables

### **5. Written Report** (20-25 pages)
- PDF format
- LaTeX source (optional)
- All figures embedded
- Full bibliography

### **6. Presentation**
- Slides (PowerPoint/PDF)
- Demo video (MP4)
- Speaker notes

---

## 🔧 Troubleshooting Guide

### **Issue: BC policy doesn't work in simulation**
- **Symptoms**: Robot falls immediately, unnatural movements
- **Solutions**:
  - Check observation construction (match demonstration data)
  - Verify action scaling/normalization
  - Ensure simulation physics match real robot
  - Check for data processing errors
  - Add more demonstration data

### **Issue: RL destroys BC gait during fine-tuning**
- **Symptoms**: Fast movement but broken gait, high energy
- **Solutions**:
  - Increase gait preservation weight
  - Start with higher α_gait, decay slowly
  - Use KL divergence penalty
  - Lower learning rate
  - Clip policy updates more aggressively

### **Issue: Hybrid reward doesn't improve over BC**
- **Symptoms**: Similar performance to BC only
- **Solutions**:
  - Increase task reward weight
  - Train longer (need more timesteps)
  - Check if BC is already optimal (ceiling effect)
  - Try different task (BC might be good at speed, try turning)
  - Verify reward is actually changing policy

### **Issue: Large sim-to-real gap**
- **Symptoms**: Works in sim, fails on real robot
- **Solutions**:
  - Add domain randomization (mass, friction, motor delays)
  - Collect real demonstrations (not simulated)
  - Tune simulation physics parameters
  - Add sensor noise in simulation
  - Use more conservative policies (lower gains)
  - Fine-tune on real robot data

### **Issue: Real robot unstable/falls**
- **Symptoms**: Robot tips over, jerky movements
- **Solutions**:
  - Start with BC only (safest)
  - Reduce action magnitude (scale down)
  - Add safety constraints (joint limits, velocity limits)
  - Check control frequency (should be 16Hz+)
  - Ensure battery is charged (>70%)
  - Test on flat ground first

### **Issue: Training is too slow**
- **Symptoms**: Not enough compute time
- **Solutions**:
  - Reduce total timesteps (3M instead of 5M)
  - Increase parallel environments (16 → 32)
  - Use GPU acceleration
  - Simplify reward function
  - Focus on 3-4 key experiments instead of 7

---

## 📚 Key Resources

### **Documentation**
- [Unitree Go1 SDK](https://github.com/unitreerobotics/unitree_legged_sdk)
- [PyBullet Documentation](https://pybullet.org/wordpress/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

### **Papers to Read**
1. **"Learning Agile Robotic Locomotion Skills by Imitating Animals"** (2020) - Similar BC+RL approach
2. **"RMA: Rapid Motor Adaptation for Legged Robots"** (2021) - Adaptation in deployment
3. **"Learning Quadrupedal Locomotion over Challenging Terrain"** (2020) - Reward shaping
4. **"Emergence of Locomotion Behaviours in Rich Environments"** (2017) - RL from scratch
5. **"DeepMimic: Example-Guided Deep Reinforcement Learning"** (2018) - Imitation + RL

### **Video Tutorials**
- Unitree Go1 programming tutorials
- PPO algorithm explained
- Behavior cloning for robotics

### **Similar Projects**
- MIT Cheetah learning
- ANYmal locomotion
- Boston Dynamics research

---

##  Success Checklist

**Week 1:**
- [ ] SDK installed and robot controllable
- [ ] PyBullet simulation working
- [ ] Collected 20-40 min demonstrations
- [ ] BC dataset prepared

**Week 2:**
- [ ] BC policy trained (MSE <0.01)
- [ ] BC works in simulation
- [ ] RL from scratch baseline trained
- [ ] Code repository organized

**Week 3:**
- [ ] Pure task reward fine-tuning complete
- [ ] 3-5 hybrid reward variants trained
- [ ] Adaptive weighting implemented
- [ ] Training curves look good

**Week 4:**
- [ ] All 7 policies tested in simulation
- [ ] Gait quality metrics computed
- [ ] Top 3-4 policies deployed on real robot
- [ ] Videos recorded

**Week 5-6:**
- [ ] All figures created
- [ ] Statistical analysis complete
- [ ] Report drafted and revised
- [ ] Presentation prepared
- [ ] Demo video edited

---

##  Expected Findings

Based on literature and your experiments, you should observe:

### **Quantitative Results**
1. **BC Only**: 
   - Velocity: 0.4-0.6 m/s (moderate)
   - Gait quality: High (natural)
   - Success rate: >95%

2. **RL from Scratch**:
   - Velocity: 0.5-0.8 m/s (variable)
   - Gait quality: Low (unnatural, maybe hopping)
   - Success rate: 70-85%
   - Takes 10M+ timesteps to converge

3. **BC + RL (Task Only)**:
   - Velocity: 0.7-1.0 m/s (fast!)
   - Gait quality: Medium-Low (broken)
   - Energy: High
   - Achieves speed but destroys naturalness

4. **BC + RL (Hybrid - Optimal)**:
   - Velocity: 0.6-0.8 m/s (fast)
   - Gait quality: Medium-High (preserved)
   - Energy: Medium
   - **Best trade-off**
   - Converges in 2-3M timesteps (faster than scratch)

5. **BC + RL (Adaptive)**:
   - Similar or slightly better than fixed hybrid
   - More stable training
   - Requires less hyperparameter tuning

### **Key Insights**
1. **BC provides strong initialization**: Converges 3-5x faster than scratch
2. **Gait preservation is essential**: Pure task reward breaks naturalness
3. **Optimal balance exists**: Around α_task=0.5-0.7, α_gait=0.3-0.5
4. **Adaptive weighting helps**: Reduces hyperparameter sensitivity
5. **Sim-to-real transfer**: BC demonstrations from real robot crucial

### **Your Contribution**
- Systematic comparison of reward balancing strategies
- Identification of optimal task/gait weight ratio
- Demonstration that adaptive weighting improves robustness
- Real robot validation of approach

---

##  Tips for Success

1. **Start with BC first**: Get this working before RL
2. **Collect quality demonstrations**: Garbage in = garbage out
3. **Visualize constantly**: Watch videos, don't just trust metrics
4. **Be patient with RL**: Training takes time (days, not hours)
5. **Test incrementally**: Don't wait until end to deploy on real robot
6. **Safety first**: Always have emergency stop ready
7. **Document everything**: Take notes during experiments
8. **Communicate daily**: Short stand-ups to stay coordinated
9. **Manage expectations**: Some experiments will fail (that's research!)
10. **Have fun**: You're working with a cool robot!

---

##  Division of Labor Summary

| Week | Member 1 | Member 2 | Member 3 | Member 4 |
|------|----------|----------|----------|----------|
| **1** | SDK/Hardware (50%)<br>Data collection (50%) | Simulation setup (60%)<br>Support (40%) | BC prep (50%)<br>Data collection (50%) | RL prep (50%)<br>Support (50%) |
| **2** | Support BC (30%)<br>Real robot prep (70%) | Env design (60%)<br>BC testing (40%) | **BC training (80%)**<br>Evaluation (20%) | Baseline RL (70%)<br>Support (30%) |
| **3** | Monitor training (40%)<br>Support (60%) | Evaluation (50%)<br>Support (50%) | Gait analysis (60%)<br>Support (40%) | **RL experiments (80%)**<br>Coordination (20%) |
| **4** | **Real robot (70%)**<br>Support (30%) | Real robot (40%)<br>Analysis (60%) | **Testing (60%)**<br>Real robot (40%) | **Analysis (70%)**<br>Real robot (30%) |
| **5-6** | Report (25%)<br>Presentation (25%) | Report (25%)<br>Presentation (25%) | Report (25%)<br>Presentation (25%) | **Report (30%)**<br>**Presentation (20%)** |

**Everyone** participates in:
- Real robot demonstrations and testing
- Report writing (divided by expertise)
- Presentation preparation and rehearsal
- Debugging and problem-solving

---

## 🌟 Stretch Goals (If Time Permits)

1. **Terrain adaptation**: Test on uneven ground
2. **Dynamic obstacles**: Navigate around moving objects
3. **Multi-task learning**: Speed + turning + climbing
4. **Online adaptation**: Fine-tune during deployment
5. **Compare to other methods**: DAGGER, GAIL, SAC
6. **Longer deployments**: Multi-minute tests
7. **Energy optimization**: Minimize battery consumption
8. **Sim-to-real analysis**: Quantify domain gap sources

---

**Good luck! This project combines cutting-edge ML with real robotics - the results will be highly valuable. The Unitree Go1 is an amazing platform, and your systematic study of reward design will contribute real insights to the field.**

**Remember**: The goal isn't just to make the robot fast, but to understand HOW to balance multiple objectives (speed vs. naturalness). That's the research contribution!
