# 📖 Complete Setup Guide for Deep-Live-Cam Environment

## 🎯 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup](#quick-setup)
3. [Manual Setup Options](#manual-setup-options)
4. [Validation & Testing](#validation--testing)
5. [Troubleshooting](#troubleshooting)
6. [Performance Optimization](#performance-optimization)
7. [Advanced Configuration](#advanced-configuration)

---

## Prerequisites

### System Requirements

#### **Hardware**
- **CPU**: Modern x86_64 processor (Intel/AMD)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (recommended)
  - RTX 3070/4070+ for optimal performance
  - RTX 2060+ minimum for real-time processing
  - GTX 1060 6GB for basic functionality
- **RAM**: 16GB+ (32GB recommended for 4K processing)
- **Storage**: 10GB free space for environment + models

#### **Software**
- **OS**: Linux (Ubuntu 20.04+, Pop!_OS, Linux Mint)
- **NVIDIA Drivers**: Version 470.57+
- **CUDA Toolkit**: Version 11.8+ (optional but recommended)
- **Conda/Miniconda**: Latest version

### Pre-Installation Checks

```bash
# Check NVIDIA drivers
nvidia-smi

# Check conda installation
conda --version

# Check Python version (system Python, not conda)
python3 --version

# Check available disk space
df -h
```

---

## Quick Setup

### Method 1: One-Command Setup (Recommended)

```bash
# Navigate to the setup directory
cd /path/to/Deep-Live-Cam/faceswap_setup

# Run automated setup
chmod +x quick_setup_faceswap.sh
./quick_setup_faceswap.sh

# Activate environment
conda activate faceswap_github

# Validate installation
python validate_faceswap_env.py
```

### Method 2: Direct Environment Creation

```bash
# Create environment from YAML
conda env create -f faceswap_environment.yml

# Activate environment
conda activate faceswap_github

# Install PyTorch with CUDA support
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Validate installation
python validate_faceswap_env.py
```

---

## Manual Setup Options

### Option A: GPU-Optimized Setup

```bash
# Create base environment
conda create -n faceswap_github python=3.10.16

# Activate environment
conda activate faceswap_github

# Install GPU-optimized packages
pip install -r requirements_gpu.txt

# Validate installation
python validate_faceswap_env.py
```

### Option B: Core Dependencies Only

```bash
# Create base environment
conda create -n faceswap_github python=3.10.16

# Activate environment
conda activate faceswap_github

# Install core packages
pip install -r requirements_core.txt

# Validate installation
python validate_faceswap_env.py
```

### Option C: CPU-Only Installation

```bash
# Create base environment
conda create -n faceswap_github python=3.10.16

# Activate environment
conda activate faceswap_github

# Install CPU-only packages
pip install -r requirements_cpu_only.txt

# Validate installation
python validate_faceswap_env.py
```

### Option D: From Explicit Conda Specification

```bash
# For exact package reproduction
conda create --name faceswap_github --file faceswap_conda_explicit.txt

# Activate environment
conda activate faceswap_github

# Install PyTorch separately
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Validate installation
python validate_faceswap_env.py
```

---

## Validation & Testing

### Environment Validation

Run the comprehensive validation script:

```bash
conda activate faceswap_github
python validate_faceswap_env.py
```

Expected output for successful installation:
```
🔍 Faceswap Environment Validation for Deep-Live-Cam
===============================================================================

📋 System Information
--------------------
✅ Operating System: Linux 6.8.0-83-generic
✅ Python Version: Python 3.10.16
✅ Conda Environment: Environment: faceswap_github

📋 GPU & NVIDIA Support
-----------------------
✅ NVIDIA Driver: Version 570.172.08
⚠️  CUDA Toolkit: Version: Not found

📋 PyTorch & CUDA
-----------------
✅ PyTorch Installation: Version 2.3.0+cu118
✅ PyTorch CUDA Support: CUDA Available: True
✅ PyTorch CUDA Version: CUDA 11.8
✅ GPU Device Count: 1 GPU(s) detected
✅ Primary GPU: NVIDIA GeForce RTX 4080
✅ GPU Memory: 16.0 GB

📋 ONNX Runtime
---------------
✅ ONNX Installation: Version 1.16.0
✅ ONNX Runtime: Version 1.16.3
✅ ONNX GPU Providers: Available: TensorrtExecutionProvider, CUDAExecutionProvider

📋 Performance Tests
-------------------
✅ GPU Matrix Multiplication: 0.003 seconds
✅ GPU Memory Allocation: 7.6 MB allocated

📊 Validation Summary
====================
🎉 Overall Status: EXCELLENT
Tests Passed: 25/27 (92.6%)

🔧 Environment Status:
✅ PyTorch CUDA: Ready
❌ TensorFlow GPU: Issue detected
✅ ONNX GPU Runtime: Ready
✅ InsightFace: Ready
✅ OpenCV: Ready

💡 Recommendations:
• Environment is ready for Deep-Live-Cam!
• GPU acceleration should work properly
```

### Manual Validation Commands

```bash
# Test PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test TensorFlow GPU
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU devices: {len(tf.config.list_physical_devices(\"GPU\"))}')"

# Test InsightFace
python -c "import insightface; print(f'InsightFace: {insightface.__version__}')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Test ONNX Runtime GPU
python -c "import onnxruntime as ort; print(f'ONNX Runtime: {ort.__version__}'); print(f'GPU providers: {[p for p in ort.get_available_providers() if \"GPU\" in p or \"CUDA\" in p]}')"
```

### Deep-Live-Cam Testing

```bash
# Navigate to Deep-Live-Cam directory
cd /path/to/Deep-Live-Cam

# Test help command
python run.py --help

# Test basic functionality (with test images)
python run.py -s source_image.jpg -t target_image.jpg -o output.jpg --execution-provider cuda

# Test live camera mode (if available)
python run.py --execution-provider cuda
```

---

## Troubleshooting

### Common Issues

#### Issue 1: CUDA Not Available in PyTorch

**Symptoms:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions:**

1. **Reinstall PyTorch with correct CUDA version:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

2. **Check NVIDIA driver compatibility:**
```bash
nvidia-smi
# Should show CUDA Version >= 11.8
```

3. **Verify CUDA installation:**
```bash
nvcc --version
# or
/usr/local/cuda/bin/nvcc --version
```

#### Issue 2: TensorFlow GPU Not Detected

**Symptoms:**
```python
>>> import tensorflow as tf
>>> len(tf.config.list_physical_devices('GPU'))
0
```

**Solutions:**

1. **Install TensorFlow GPU dependencies:**
```bash
# For CUDA 11.8
pip install tensorflow[and-cuda]==2.19.0

# Or install cuDNN manually
conda install cudnn
```

2. **Set environment variables:**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=0
```

#### Issue 3: Environment Creation Fails

**Symptoms:**
```
CondaEnvException: Pip failed
ERROR: Could not find a version that satisfies the requirement...
```

**Solutions:**

1. **Use mamba for faster resolution:**
```bash
conda install mamba -n base -c conda-forge
mamba env create -f faceswap_environment.yml
```

2. **Create environment step by step:**
```bash
conda create -n faceswap_github python=3.10.16
conda activate faceswap_github
pip install -r requirements_core.txt
pip install -r requirements_gpu.txt
```

3. **Clear conda cache:**
```bash
conda clean --all
conda update --all
```

#### Issue 4: InsightFace Import Error

**Symptoms:**
```python
>>> import insightface
ImportError: ...
```

**Solutions:**

1. **Reinstall with specific version:**
```bash
pip uninstall insightface
pip install insightface==0.7.3
```

2. **Install build dependencies:**
```bash
sudo apt-get install cmake libopenblas-dev liblapack-dev
pip install --upgrade cython numpy
pip install insightface==0.7.3
```

#### Issue 5: OpenCV Issues

**Symptoms:**
```python
>>> import cv2
ImportError: libGL.so.1: cannot open shared object file
```

**Solutions:**

1. **Install system dependencies:**
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

2. **Use headless OpenCV:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless==4.11.0.86
```

#### Issue 6: Memory Issues

**Symptoms:**
- Out of memory errors
- Slow performance
- System freezing

**Solutions:**

1. **Set memory limits:**
```bash
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

2. **Monitor GPU memory:**
```bash
watch -n 1 nvidia-smi
```

3. **Optimize batch processing:**
```python
# In your code
torch.cuda.empty_cache()  # Clear PyTorch cache
```

### Performance Issues

#### Slow Processing

1. **Check GPU utilization:**
```bash
nvidia-smi dmon
```

2. **Verify CUDA execution:**
```bash
python -c "
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

3. **Optimize execution providers:**
```bash
# For Deep-Live-Cam, prefer CUDA provider
python run.py --execution-provider cuda
```

---

## Performance Optimization

### GPU Optimization

#### NVIDIA GPU Settings

```bash
# Set maximum performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 877,1480  # Adjust for your GPU

# Monitor temperatures
nvidia-smi dmon -s pucvmet -d 1
```

#### CUDA Optimization

```bash
# Add to ~/.bashrc
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Memory Optimization

#### System Memory

```bash
# Increase swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### GPU Memory

```python
# In Python code
import torch

# Clear cache periodically
torch.cuda.empty_cache()

# Use mixed precision
with torch.cuda.amp.autocast():
    # Your model inference here
    pass
```

### CPU Optimization

```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check CPU usage
htop
```

---

## Advanced Configuration

### Environment Customization

#### Custom Package Versions

```yaml
# Custom environment.yml
name: faceswap_custom
dependencies:
  - python=3.10.16
  - pip
  - pip:
    - torch==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    - torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    # Add your custom packages here
```

#### Development Mode

```bash
# Create development environment with additional tools
conda create -n faceswap_dev python=3.10.16
conda activate faceswap_dev

# Install core packages
pip install -r requirements_core.txt

# Add development tools
pip install jupyter ipython pytest black flake8 mypy
```

### Multiple Environment Management

```bash
# List all environments
conda env list

# Export current environment
conda env export > my_faceswap_env.yml

# Clone environment
conda create --name faceswap_backup --clone faceswap_github

# Remove environment
conda env remove -n faceswap_github
```

### Docker Container Setup

```dockerfile
# Dockerfile for faceswap environment
FROM nvidia/cuda:11.8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip wget curl git \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Copy environment files
COPY faceswap_environment.yml .
COPY requirements_gpu.txt .

# Create environment
RUN conda env create -f faceswap_environment.yml
RUN echo "conda activate faceswap_github" >> ~/.bashrc

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
```

### Batch Setup Script

```bash
#!/bin/bash
# batch_setup_multiple_envs.sh

ENVS=("faceswap_prod" "faceswap_dev" "faceswap_test")

for env_name in "${ENVS[@]}"; do
    echo "Creating environment: $env_name"

    # Modify environment.yml name
    sed "s/name: faceswap_github/name: $env_name/" faceswap_environment.yml > temp_env.yml

    # Create environment
    conda env create -f temp_env.yml

    # Install PyTorch
    conda activate $env_name
    pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118

    echo "Completed: $env_name"
done

rm temp_env.yml
```

---

## 🎯 Final Checklist

Before using Deep-Live-Cam, ensure:

- [ ] Environment validation passes (>90% tests)
- [ ] PyTorch CUDA support confirmed
- [ ] ONNX Runtime GPU providers available
- [ ] Deep-Live-Cam help command works
- [ ] GPU memory >4GB available
- [ ] NVIDIA drivers updated
- [ ] System has adequate cooling
- [ ] Backup environment created (optional)

## 🆘 Getting Help

If you encounter issues:

1. **Run validation script:** `python validate_faceswap_env.py`
2. **Check environment:** `conda activate faceswap_github && conda list`
3. **Monitor resources:** `nvidia-smi` and `htop`
4. **Review logs:** Check terminal output for specific error messages
5. **Search documentation:** This guide covers most common issues
6. **Community support:** GitHub issues, forums, Discord channels

---

**🎉 You're now ready to use Deep-Live-Cam with GPU acceleration!**

*Last updated: September 29, 2025*