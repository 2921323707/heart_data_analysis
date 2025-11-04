"""
GPU检测诊断脚本
用于检查GPU可用性和PyTorch CUDA支持
"""
import sys

print("=" * 60)
print("GPU检测诊断")
print("=" * 60)

# 1. 检查Python版本
print(f"\n1. Python版本: {sys.version}")

# 2. 检查PyTorch是否安装
try:
    import torch
    print(f"2. PyTorch版本: {torch.__version__}")
except ImportError:
    print("2. ❌ PyTorch未安装")
    print("   请运行: pip install torch torchvision")
    sys.exit(1)

# 3. 检查CUDA是否可用
print(f"\n3. CUDA可用性:")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   ✅ CUDA可用！")
    print(f"   CUDA版本: {torch.version.cuda}")
    print(f"   cuDNN版本: {torch.backends.cudnn.version()}")
    
    # 检查GPU信息
    print(f"\n4. GPU信息:")
    print(f"   GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"     总内存: {props.total_memory / 1024**3:.2f} GB")
        print(f"     计算能力: {props.major}.{props.minor}")
    
    # 测试GPU计算
    print(f"\n5. GPU计算测试:")
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = torch.mm(x, y)
        print(f"   ✅ GPU计算测试成功")
        print(f"   测试结果形状: {z.shape}")
    except Exception as e:
        print(f"   ❌ GPU计算测试失败: {e}")
else:
    print(f"   ❌ CUDA不可用")
    
    # 诊断问题
    print(f"\n4. 问题诊断:")
    
    # 检查是否有CUDA支持
    if hasattr(torch.version, 'cuda'):
        if torch.version.cuda is None:
            print(f"   ⚠️  PyTorch安装的是CPU版本，没有CUDA支持")
            print(f"   解决方案:")
            print(f"   1. 卸载当前PyTorch: pip uninstall torch torchvision")
            print(f"   2. 根据您的CUDA版本安装对应的PyTorch:")
            print(f"      - CUDA 11.8: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print(f"      - CUDA 12.1: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print(f"      - 或者访问: https://pytorch.org/get-started/locally/")
        else:
            print(f"   PyTorch编译时支持CUDA: {torch.version.cuda}")
            print(f"   但运行时CUDA不可用，可能原因:")
            print(f"   - CUDA驱动未安装或版本不匹配")
            print(f"   - GPU驱动未正确安装")
            print(f"   - 环境变量配置问题")
    else:
        print(f"   PyTorch版本过旧，无法检测CUDA支持")
    
    # 检查nvidia-smi
    print(f"\n5. 检查NVIDIA驱动:")
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"   ✅ nvidia-smi可用，NVIDIA驱动已安装")
            print(f"   GPU信息:")
            lines = result.stdout.split('\n')
            for line in lines[:5]:  # 只显示前几行
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"   ❌ nvidia-smi不可用")
    except FileNotFoundError:
        print(f"   ❌ nvidia-smi未找到，可能NVIDIA驱动未安装")
        print(f"   请访问 https://www.nvidia.com/Download/index.aspx 安装驱动")
    except subprocess.TimeoutExpired:
        print(f"   ⚠️  nvidia-smi超时")
    except Exception as e:
        print(f"   ⚠️  无法运行nvidia-smi: {e}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)

