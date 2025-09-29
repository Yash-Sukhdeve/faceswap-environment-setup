#!/bin/bash

# =============================================================================
# Faceswap Environment Quick Setup Script for Deep-Live-Cam
# =============================================================================
#
# This script automatically creates a complete faceswap_github conda environment
# for Deep-Live-Cam with GPU acceleration support.
#
# Requirements:
# - Linux system with NVIDIA GPU
# - NVIDIA drivers installed (470.57+)
# - Conda/Miniconda installed
# - CUDA toolkit 11.8+ (optional but recommended)
#
# Usage:
#   chmod +x quick_setup_faceswap.sh
#   ./quick_setup_faceswap.sh
#
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
ENV_NAME="faceswap_github"
PYTHON_VERSION="3.10.16"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        error "Conda is not installed or not in PATH. Please install Miniconda or Anaconda first."
    fi

    # Check if nvidia-smi is available (indicates NVIDIA drivers)
    if ! command -v nvidia-smi &> /dev/null; then
        warning "nvidia-smi not found. GPU acceleration may not work."
        warning "Please ensure NVIDIA drivers are installed for optimal performance."
    else
        log "NVIDIA drivers detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    fi

    # Check if required files exist
    if [[ ! -f "$SCRIPT_DIR/faceswap_environment.yml" ]]; then
        error "faceswap_environment.yml not found in script directory: $SCRIPT_DIR"
    fi

    success "Prerequisites check completed"
}

# Remove existing environment if it exists
remove_existing_env() {
    log "Checking for existing environment..."

    if conda env list | grep -q "^$ENV_NAME "; then
        warning "Environment '$ENV_NAME' already exists"
        read -p "Do you want to remove it and create a fresh one? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log "Removing existing environment..."
            conda env remove -n "$ENV_NAME" -y
            success "Existing environment removed"
        else
            log "Keeping existing environment. Attempting to update packages..."
            return 0
        fi
    fi
}

# Create conda environment
create_conda_environment() {
    log "Creating conda environment '$ENV_NAME'..."

    # Initialize conda for bash
    eval "$(conda shell.bash hook)"

    # Create environment from YAML file
    if conda env create -f "$SCRIPT_DIR/faceswap_environment.yml"; then
        success "Conda environment created successfully"
    else
        error "Failed to create conda environment"
    fi
}

# Install PyTorch with CUDA support
install_pytorch_cuda() {
    log "Installing PyTorch with CUDA 11.8 support..."

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # Install PyTorch with CUDA support
    if pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118; then
        success "PyTorch with CUDA support installed successfully"
    else
        warning "PyTorch CUDA installation had issues. Trying fallback installation..."
        pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
        warning "Installed CPU-only version of PyTorch"
    fi
}

# Validate environment
validate_environment() {
    log "Validating environment setup..."

    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # Test Python version
    PYTHON_VER=$(python --version)
    log "Python version: $PYTHON_VER"

    # Test PyTorch
    if python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
        success "PyTorch validation passed"
    else
        warning "PyTorch validation failed"
    fi

    # Test TensorFlow
    if python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>/dev/null; then
        success "TensorFlow validation passed"
    else
        warning "TensorFlow validation failed"
    fi

    # Test InsightFace
    if python -c "import insightface; print(f'InsightFace: {insightface.__version__}')" 2>/dev/null; then
        success "InsightFace validation passed"
    else
        warning "InsightFace validation failed"
    fi

    # Test OpenCV
    if python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>/dev/null; then
        success "OpenCV validation passed"
    else
        warning "OpenCV validation failed"
    fi

    # Test ONNX Runtime
    if python -c "import onnxruntime as ort; print(f'ONNX Runtime: {ort.__version__}'); print(f'GPU providers: {[p for p in ort.get_available_providers() if \"GPU\" in p or \"CUDA\" in p]}')" 2>/dev/null; then
        success "ONNX Runtime validation passed"
    else
        warning "ONNX Runtime validation failed"
    fi
}

# Run validation script if available
run_validation_script() {
    if [[ -f "$SCRIPT_DIR/validate_faceswap_env.py" ]]; then
        log "Running comprehensive validation script..."
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME"

        if python "$SCRIPT_DIR/validate_faceswap_env.py"; then
            success "Comprehensive validation passed"
        else
            warning "Some validation tests failed. Check output above."
        fi
    fi
}

# Print installation summary
print_summary() {
    echo
    echo "=========================================================================="
    echo -e "${GREEN}🎉 Faceswap Environment Setup Complete!${NC}"
    echo "=========================================================================="
    echo
    echo -e "${BLUE}Environment Details:${NC}"
    echo "  Name: $ENV_NAME"
    echo "  Python: $PYTHON_VERSION"
    echo "  Location: $(conda info --envs | grep $ENV_NAME | awk '{print $2}')"
    echo
    echo -e "${BLUE}Next Steps:${NC}"
    echo "  1. Activate the environment:"
    echo "     conda activate $ENV_NAME"
    echo
    echo "  2. Test with Deep-Live-Cam:"
    echo "     cd /path/to/Deep-Live-Cam"
    echo "     python run.py --help"
    echo
    echo "  3. For validation, run:"
    echo "     python validate_faceswap_env.py"
    echo
    echo -e "${BLUE}GPU Acceleration:${NC}"
    echo "  - PyTorch CUDA: Available if NVIDIA GPU detected"
    echo "  - TensorFlow GPU: May require additional setup"
    echo "  - ONNX Runtime: GPU providers included"
    echo
    echo -e "${GREEN}Environment ready for GPU-accelerated face swapping!${NC}"
    echo "=========================================================================="
}

# Main execution
main() {
    echo "=========================================================================="
    echo -e "${BLUE}🚀 Faceswap Environment Quick Setup for Deep-Live-Cam${NC}"
    echo "=========================================================================="
    echo

    check_prerequisites
    remove_existing_env
    create_conda_environment
    install_pytorch_cuda
    validate_environment
    run_validation_script
    print_summary
}

# Run main function
main "$@"