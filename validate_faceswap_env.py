#!/usr/bin/env python3
"""
Faceswap Environment Validation Script for Deep-Live-Cam

This script performs comprehensive validation of the faceswap_github conda environment
to ensure all components are properly installed and GPU acceleration is working.

Usage:
    python validate_faceswap_env.py

Requirements:
    - Must be run within the faceswap_github conda environment
    - NVIDIA GPU with CUDA support (optional but recommended)
"""

import sys
import subprocess
import importlib
import platform
import os
from typing import Dict, List, Tuple, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class EnvironmentValidator:
    """Comprehensive environment validation for faceswap_github"""

    def __init__(self):
        self.results = {
            'system': {},
            'python': {},
            'conda': {},
            'gpu': {},
            'pytorch': {},
            'tensorflow': {},
            'onnx': {},
            'opencv': {},
            'insightface': {},
            'utilities': {}
        }
        self.passed_tests = 0
        self.total_tests = 0

    def print_header(self):
        """Print validation header"""
        print(f"\n{Colors.BLUE}{'=' * 80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}🔍 Faceswap Environment Validation for Deep-Live-Cam{Colors.END}")
        print(f"{Colors.BLUE}{'=' * 80}{Colors.END}\n")

    def print_section(self, title: str):
        """Print section header"""
        print(f"{Colors.BOLD}{Colors.PURPLE}📋 {title}{Colors.END}")
        print(f"{Colors.PURPLE}{'-' * (len(title) + 4)}{Colors.END}")

    def test_result(self, test_name: str, success: bool, details: str = "", warning: bool = False) -> bool:
        """Print test result and update counters"""
        self.total_tests += 1

        if success:
            self.passed_tests += 1
            icon = "✅"
            color = Colors.GREEN
        elif warning:
            icon = "⚠️ "
            color = Colors.YELLOW
        else:
            icon = "❌"
            color = Colors.RED

        print(f"{color}{icon} {test_name}{Colors.END}", end="")
        if details:
            print(f": {details}")
        else:
            print()

        return success

    def validate_system_info(self):
        """Validate system information"""
        self.print_section("System Information")

        # Operating System
        os_info = f"{platform.system()} {platform.release()}"
        self.test_result("Operating System", True, os_info)
        self.results['system']['os'] = os_info

        # Python Version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        python_ok = sys.version_info >= (3, 10)
        self.test_result("Python Version", python_ok, f"Python {python_version}")
        self.results['python']['version'] = python_version

        # Conda Environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not activated')
        env_ok = conda_env == 'faceswap_github'
        self.test_result("Conda Environment", env_ok, f"Environment: {conda_env}", warning=not env_ok)
        self.results['conda']['environment'] = conda_env

        print()

    def validate_gpu_support(self):
        """Validate GPU and NVIDIA driver support"""
        self.print_section("GPU & NVIDIA Support")

        # NVIDIA Driver
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            nvidia_available = result.returncode == 0
            if nvidia_available:
                # Extract driver version
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Driver Version:' in line:
                        driver_version = line.split('Driver Version:')[1].split()[0]
                        self.test_result("NVIDIA Driver", True, f"Version {driver_version}")
                        self.results['gpu']['driver'] = driver_version
                        break
            else:
                self.test_result("NVIDIA Driver", False, "nvidia-smi not found")
                self.results['gpu']['driver'] = None
        except FileNotFoundError:
            self.test_result("NVIDIA Driver", False, "nvidia-smi not found")
            self.results['gpu']['driver'] = None

        # CUDA Toolkit
        cuda_version = os.environ.get('CUDA_VERSION', 'Not found')
        cuda_ok = cuda_version != 'Not found'
        self.test_result("CUDA Toolkit", cuda_ok, f"Version: {cuda_version}", warning=not cuda_ok)
        self.results['gpu']['cuda'] = cuda_version

        print()

    def validate_pytorch(self):
        """Validate PyTorch installation and CUDA support"""
        self.print_section("PyTorch & CUDA")

        try:
            import torch

            # PyTorch Version
            pytorch_version = torch.__version__
            self.test_result("PyTorch Installation", True, f"Version {pytorch_version}")
            self.results['pytorch']['version'] = pytorch_version

            # CUDA Support
            cuda_available = torch.cuda.is_available()
            self.test_result("PyTorch CUDA Support", cuda_available,
                           f"CUDA Available: {cuda_available}")
            self.results['pytorch']['cuda_available'] = cuda_available

            if cuda_available:
                # CUDA Version
                cuda_version = torch.version.cuda
                self.test_result("PyTorch CUDA Version", True, f"CUDA {cuda_version}")
                self.results['pytorch']['cuda_version'] = cuda_version

                # GPU Count
                gpu_count = torch.cuda.device_count()
                self.test_result("GPU Device Count", gpu_count > 0, f"{gpu_count} GPU(s) detected")
                self.results['pytorch']['gpu_count'] = gpu_count

                if gpu_count > 0:
                    # GPU Name
                    gpu_name = torch.cuda.get_device_name(0)
                    self.test_result("Primary GPU", True, gpu_name)
                    self.results['pytorch']['gpu_name'] = gpu_name

                    # GPU Memory
                    try:
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory
                        gpu_memory_gb = gpu_memory / (1024**3)
                        memory_ok = gpu_memory_gb >= 4.0  # Minimum 4GB for face swapping
                        self.test_result("GPU Memory", memory_ok, f"{gpu_memory_gb:.1f} GB")
                        self.results['pytorch']['gpu_memory_gb'] = gpu_memory_gb
                    except Exception as e:
                        self.test_result("GPU Memory", False, f"Error: {str(e)}")

        except ImportError:
            self.test_result("PyTorch Installation", False, "PyTorch not found")
            self.results['pytorch']['version'] = None
        except Exception as e:
            self.test_result("PyTorch Validation", False, f"Error: {str(e)}")

        print()

    def validate_tensorflow(self):
        """Validate TensorFlow installation"""
        self.print_section("TensorFlow")

        try:
            import tensorflow as tf

            # TensorFlow Version
            tf_version = tf.__version__
            self.test_result("TensorFlow Installation", True, f"Version {tf_version}")
            self.results['tensorflow']['version'] = tf_version

            # GPU Support
            try:
                gpu_devices = tf.config.list_physical_devices('GPU')
                gpu_available = len(gpu_devices) > 0
                self.test_result("TensorFlow GPU Support", gpu_available,
                               f"{len(gpu_devices)} GPU device(s)", warning=not gpu_available)
                self.results['tensorflow']['gpu_devices'] = len(gpu_devices)
            except Exception as e:
                self.test_result("TensorFlow GPU Check", False, f"Error: {str(e)}")

        except ImportError:
            self.test_result("TensorFlow Installation", False, "TensorFlow not found")
            self.results['tensorflow']['version'] = None
        except Exception as e:
            self.test_result("TensorFlow Validation", False, f"Error: {str(e)}")

        print()

    def validate_onnx(self):
        """Validate ONNX and ONNX Runtime"""
        self.print_section("ONNX Runtime")

        try:
            import onnx
            onnx_version = onnx.__version__
            self.test_result("ONNX Installation", True, f"Version {onnx_version}")
            self.results['onnx']['version'] = onnx_version
        except ImportError:
            self.test_result("ONNX Installation", False, "ONNX not found")

        try:
            import onnxruntime as ort

            # ONNX Runtime Version
            ort_version = ort.__version__
            self.test_result("ONNX Runtime", True, f"Version {ort_version}")
            self.results['onnx']['runtime_version'] = ort_version

            # Available Providers
            providers = ort.get_available_providers()
            gpu_providers = [p for p in providers if any(gpu in p for gpu in ['CUDA', 'TensorRT', 'GPU'])]

            self.test_result("ONNX GPU Providers", len(gpu_providers) > 0,
                           f"Available: {', '.join(gpu_providers) if gpu_providers else 'CPU only'}")
            self.results['onnx']['gpu_providers'] = gpu_providers
            self.results['onnx']['all_providers'] = providers

        except ImportError:
            self.test_result("ONNX Runtime", False, "ONNX Runtime not found")
        except Exception as e:
            self.test_result("ONNX Runtime Validation", False, f"Error: {str(e)}")

        print()

    def validate_opencv(self):
        """Validate OpenCV installation"""
        self.print_section("OpenCV")

        try:
            import cv2

            # OpenCV Version
            cv_version = cv2.__version__
            self.test_result("OpenCV Installation", True, f"Version {cv_version}")
            self.results['opencv']['version'] = cv_version

            # Test basic functionality
            try:
                # Test image creation
                import numpy as np
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
                self.test_result("OpenCV Basic Functions", True, "Color conversion working")
                self.results['opencv']['basic_functions'] = True
            except Exception as e:
                self.test_result("OpenCV Basic Functions", False, f"Error: {str(e)}")
                self.results['opencv']['basic_functions'] = False

        except ImportError:
            self.test_result("OpenCV Installation", False, "OpenCV not found")
            self.results['opencv']['version'] = None
        except Exception as e:
            self.test_result("OpenCV Validation", False, f"Error: {str(e)}")

        print()

    def validate_insightface(self):
        """Validate InsightFace installation"""
        self.print_section("InsightFace")

        try:
            import insightface

            # InsightFace Version
            if_version = insightface.__version__
            self.test_result("InsightFace Installation", True, f"Version {if_version}")
            self.results['insightface']['version'] = if_version

            # Test model loading (without actually downloading)
            try:
                # This tests if the package structure is correct
                from insightface.app import FaceAnalysis
                self.test_result("InsightFace App Module", True, "FaceAnalysis class available")
                self.results['insightface']['app_module'] = True
            except ImportError as e:
                self.test_result("InsightFace App Module", False, f"Error: {str(e)}")
                self.results['insightface']['app_module'] = False

        except ImportError:
            self.test_result("InsightFace Installation", False, "InsightFace not found")
            self.results['insightface']['version'] = None
        except Exception as e:
            self.test_result("InsightFace Validation", False, f"Error: {str(e)}")

        print()

    def validate_utilities(self):
        """Validate utility packages"""
        self.print_section("Utility Packages")

        # Core packages
        packages = [
            ('numpy', 'NumPy'),
            ('PIL', 'Pillow'),
            ('matplotlib', 'Matplotlib'),
            ('scipy', 'SciPy'),
            ('sklearn', 'Scikit-learn'),
            ('customtkinter', 'CustomTkinter'),
            ('tqdm', 'TQDM'),
            ('requests', 'Requests'),
            ('yaml', 'PyYAML')
        ]

        for module_name, display_name in packages:
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'Unknown version')
                self.test_result(f"{display_name}", True, f"Version {version}")
                self.results['utilities'][display_name.lower()] = version
            except ImportError:
                self.test_result(f"{display_name}", False, "Not found")
                self.results['utilities'][display_name.lower()] = None

        print()

    def run_performance_test(self):
        """Run basic performance tests"""
        self.print_section("Performance Tests")

        try:
            import torch
            import time

            if torch.cuda.is_available():
                # GPU tensor operations
                device = torch.device('cuda')
                start_time = time.time()

                # Create tensors on GPU
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)

                # Perform matrix multiplication
                c = torch.mm(a, b)
                torch.cuda.synchronize()  # Wait for GPU operations

                gpu_time = time.time() - start_time
                self.test_result("GPU Matrix Multiplication", True, f"{gpu_time:.3f} seconds")
                self.results['gpu']['performance_test'] = gpu_time

                # Memory test
                try:
                    gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)  # MB
                    self.test_result("GPU Memory Allocation", True, f"{gpu_memory_allocated:.1f} MB allocated")
                    self.results['gpu']['memory_allocated_mb'] = gpu_memory_allocated
                except Exception as e:
                    self.test_result("GPU Memory Test", False, f"Error: {str(e)}")
            else:
                self.test_result("GPU Performance Test", False, "CUDA not available", warning=True)

        except Exception as e:
            self.test_result("Performance Test", False, f"Error: {str(e)}")

        print()

    def print_summary(self):
        """Print validation summary"""
        print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}📊 Validation Summary{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")

        # Test Results
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0

        if success_rate >= 90:
            color = Colors.GREEN
            status = "EXCELLENT"
            emoji = "🎉"
        elif success_rate >= 75:
            color = Colors.YELLOW
            status = "GOOD"
            emoji = "✅"
        else:
            color = Colors.RED
            status = "NEEDS ATTENTION"
            emoji = "⚠️"

        print(f"{color}{emoji} Overall Status: {status}{Colors.END}")
        print(f"{color}Tests Passed: {self.passed_tests}/{self.total_tests} ({success_rate:.1f}%){Colors.END}\n")

        # Environment Status
        print(f"{Colors.BOLD}🔧 Environment Status:{Colors.END}")

        # Critical components
        pytorch_ok = self.results.get('pytorch', {}).get('cuda_available', False)
        tensorflow_ok = self.results.get('tensorflow', {}).get('gpu_devices', 0) > 0
        onnx_ok = len(self.results.get('onnx', {}).get('gpu_providers', [])) > 0
        insightface_ok = self.results.get('insightface', {}).get('version') is not None
        opencv_ok = self.results.get('opencv', {}).get('version') is not None

        components = [
            ("PyTorch CUDA", pytorch_ok),
            ("TensorFlow GPU", tensorflow_ok),
            ("ONNX GPU Runtime", onnx_ok),
            ("InsightFace", insightface_ok),
            ("OpenCV", opencv_ok)
        ]

        for component, status in components:
            icon = "✅" if status else "❌"
            color = Colors.GREEN if status else Colors.RED
            print(f"{color}{icon} {component}: {'Ready' if status else 'Issue detected'}{Colors.END}")

        print()

        # Recommendations
        print(f"{Colors.BOLD}💡 Recommendations:{Colors.END}")

        if not pytorch_ok:
            print(f"{Colors.YELLOW}• Install PyTorch with CUDA support for GPU acceleration{Colors.END}")

        if not tensorflow_ok:
            print(f"{Colors.YELLOW}• TensorFlow GPU support may require additional setup{Colors.END}")

        if not onnx_ok:
            print(f"{Colors.YELLOW}• Install onnxruntime-gpu for optimized inference{Colors.END}")

        if success_rate >= 90:
            print(f"{Colors.GREEN}• Environment is ready for Deep-Live-Cam!{Colors.END}")
            print(f"{Colors.GREEN}• GPU acceleration should work properly{Colors.END}")
        elif success_rate >= 75:
            print(f"{Colors.YELLOW}• Environment is mostly ready, but some optimizations possible{Colors.END}")
        else:
            print(f"{Colors.RED}• Environment needs attention before running Deep-Live-Cam{Colors.END}")

        print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")

        return success_rate >= 75

    def run_validation(self) -> bool:
        """Run complete validation suite"""
        self.print_header()

        self.validate_system_info()
        self.validate_gpu_support()
        self.validate_pytorch()
        self.validate_tensorflow()
        self.validate_onnx()
        self.validate_opencv()
        self.validate_insightface()
        self.validate_utilities()
        self.run_performance_test()

        return self.print_summary()

def main():
    """Main validation function"""
    validator = EnvironmentValidator()

    try:
        success = validator.run_validation()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Validation failed with error: {str(e)}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()