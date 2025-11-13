#!/usr/bin/env python3
"""
Test installation and verify all dependencies are correctly installed.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    tests = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("pybullet", "PyBullet"),
        ("gym", "Gym"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("matplotlib", "Matplotlib"),
        ("yaml", "PyYAML"),
        ("tensorboard", "TensorBoard"),
    ]
    
    failed = []
    for module, name in tests:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    - Device: {torch.cuda.get_device_name(0)}")
            print(f"    - CUDA Version: {torch.version.cuda}")
        else:
            print("  ⚠ CUDA not available (CPU only mode)")
        return True
    except Exception as e:
        print(f"  ✗ CUDA test failed: {e}")
        return False


def test_pybullet():
    """Test PyBullet simulation."""
    print("\nTesting PyBullet...")
    try:
        import pybullet as p
        client = p.connect(p.DIRECT)
        print(f"  ✓ PyBullet initialized")
        print(f"    - Version: {p.getVersionInfo()}")
        p.disconnect()
        return True
    except Exception as e:
        print(f"  ✗ PyBullet test failed: {e}")
        return False


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    import os
    
    required_dirs = [
        "src",
        "scripts",
        "config",
        "models/bc",
        "models/rl",
        "data/demonstrations",
        "data/processed",
        "results/figures",
        "results/videos",
        "results/logs",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - missing!")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("=" * 60)
    print("Go1 BC+RL Project - Installation Test")
    print("=" * 60)
    
    # Test imports
    imports_ok, failed = test_imports()
    
    # Test CUDA
    cuda_ok = test_cuda()
    
    # Test PyBullet
    pybullet_ok = test_pybullet()
    
    # Test directories
    dirs_ok = test_directories()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if imports_ok:
        print("✓ All required packages installed")
    else:
        print(f"✗ Missing packages: {', '.join(failed)}")
        print("  Run: pip install -r requirements.txt")
    
    if cuda_ok:
        print("✓ PyTorch CUDA test passed")
    else:
        print("⚠ CUDA not available or failed")
    
    if pybullet_ok:
        print("✓ PyBullet simulation test passed")
    else:
        print("✗ PyBullet test failed")
    
    if dirs_ok:
        print("✓ Directory structure correct")
    else:
        print("✗ Some directories missing")
    
    # Overall result
    print("\n" + "=" * 60)
    if imports_ok and pybullet_ok and dirs_ok:
        print("✓ INSTALLATION SUCCESSFUL")
        print("You're ready to start training!")
        return 0
    else:
        print("✗ INSTALLATION INCOMPLETE")
        print("Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
