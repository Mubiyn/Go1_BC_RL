# Quick Start Guide

## For New Team Members

Follow these steps to get started with the project:

### Step 1: Clone Repository (or use existing folder)
```bash
cd TI/go1_bc_rl_project
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python3.8 -m venv venv
source venv/bin/activate  # On macOS/Linux

# OR using conda
conda env create -f environment.yml
conda activate go1_bc_rl
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Test Installation
```bash
python scripts/test_installation.py
```

You should see:
```
✓ All required packages installed
✓ PyTorch CUDA test passed
✓ PyBullet simulation test passed
✓ Directory structure correct
✓ INSTALLATION SUCCESSFUL
```

### Step 5: Read Documentation
1. **README.md** - Project overview and usage
2. **Task2_Quadruple_SOTA_Guide.md** - Detailed implementation guide
3. **PROJECT_STRUCTURE.md** - Directory structure and next steps

### Step 6: Start Development

Choose your role and start implementing:

**Member 1 (Robot Interface Lead):**
```bash
# Start with data logger
edit src/utils/data_logger.py
# Then demo collection
edit real_robot/collect_demonstrations.py
```

**Member 2 (Simulation Lead):**
```bash
# Start with Go1 environment
edit src/envs/go1_env.py
```

**Member 3 (BC Specialist):**
```bash
# Start with BC policy
edit src/bc/policy.py
edit src/bc/dataset.py
edit src/bc/trainer.py
```

**Member 4 (RL Lead):**
```bash
# Start with reward functions
edit src/rl/reward_functions.py
```

## Common Tasks

### Run Training (once implemented)
```bash
# BC training
python scripts/train_bc.py --config config/bc_config.yaml

# RL training
python scripts/train_rl_finetune.py --config config/rl_finetune_config.yaml
```

### Monitor Training
```bash
tensorboard --logdir results/logs/
```

### Run Tests
```bash
pytest tests/
```

### Check Code Style
```bash
black src/ scripts/
flake8 src/ scripts/
```

## Git Workflow

### First Time Setup
```bash
git init
git add .
git commit -m "Initial commit: Project structure"
git remote add origin YOUR_REPO_URL
git push -u origin main
```

### Daily Workflow
```bash
# Pull latest changes
git pull

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add your_files
git commit -m "Clear description of changes"

# Push to remote
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### Commit Message Convention
```
feat: Add new feature
fix: Bug fix
docs: Documentation changes
style: Code formatting
refactor: Code refactoring
test: Add tests
chore: Maintenance tasks
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root
cd TI/go1_bc_rl_project

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### CUDA Not Available
```bash
# Install CUDA-enabled PyTorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

### PyBullet GUI Issues on macOS
```bash
# Install additional dependencies
brew install freeglut
```

## Useful Commands

### View Project Structure
```bash
tree -L 3 -I '__pycache__|*.pyc|venv'
```

### Count Lines of Code
```bash
find src -name '*.py' | xargs wc -l
```

### Find TODOs
```bash
grep -r "TODO" src/ scripts/
```

## Resources

- **Main Guide**: `Task2_Quadruple_SOTA_Guide.md`
- **Documentation**: `README.md`
- **Structure**: `PROJECT_STRUCTURE.md`
- **Unitree SDK**: https://github.com/unitreerobotics/unitree_legged_sdk
- **PyBullet Docs**: https://pybullet.org/wordpress/
- **SB3 Docs**: https://stable-baselines3.readthedocs.io/

## Getting Help

1. Check the documentation first
2. Look at the example code in `Task2_Quadruple_SOTA_Guide.md`
3. Ask team members
4. Check GitHub issues (if repository is set up)

## Timeline Reminder

- **Week 1**: Setup, data collection
- **Week 2**: BC training
- **Week 3**: RL fine-tuning experiments
- **Week 4**: Testing and real robot deployment
- **Week 5-6**: Analysis, report, presentation

Good luck! 🚀
