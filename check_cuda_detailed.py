"""
详细检查CUDA安装情况
检查CUDA驱动、CUDA Toolkit和PyTorch的CUDA支持
"""
import subprocess
import sys
import os

print("=" * 70)
print("CUDA详细检查")
print("=" * 70)

# 1. 检查nvidia-smi（NVIDIA驱动）
print("\n【1. NVIDIA驱动检查】")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ nvidia-smi可用")
        # 提取CUDA版本
        import re
        cuda_match = re.search(r'CUDA Version:\s+(\d+\.\d+)', result.stdout)
        if cuda_match:
            driver_cuda_version = cuda_match.group(1)
            print(f"   驱动支持CUDA版本: {driver_cuda_version}")
        
        # 显示GPU信息
        gpu_match = re.search(r'\|.*\|\s+(\S+.*?)\s+\|', result.stdout)
        if gpu_match:
            print(f"   GPU信息: {result.stdout.split(chr(10))[8] if len(result.stdout.split(chr(10))) > 8 else 'N/A'}")
    else:
        print("❌ nvidia-smi不可用")
except FileNotFoundError:
    print("❌ nvidia-smi未找到")
except Exception as e:
    print(f"❌ 错误: {e}")

# 2. 检查CUDA Toolkit（nvcc）
print("\n【2. CUDA Toolkit检查】")
cuda_toolkit_installed = False
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ CUDA Toolkit已安装")
        cuda_toolkit_installed = True
        # 提取版本
        import re
        version_match = re.search(r'release\s+(\d+\.\d+)', result.stdout, re.IGNORECASE)
        if version_match:
            toolkit_version = version_match.group(1)
            print(f"   CUDA Toolkit版本: {toolkit_version}")
        print("   输出:")
        for line in result.stdout.split('\n')[:3]:
            if line.strip():
                print(f"   {line}")
    else:
        print("⚠️  nvcc命令执行失败")
except FileNotFoundError:
    print("⚠️  CUDA Toolkit可能未安装（nvcc未找到）")
    print("   说明: 有NVIDIA驱动不等于有CUDA Toolkit")
    print("   对于PyTorch，通常只需要驱动，不需要完整安装CUDA Toolkit")
except Exception as e:
    print(f"⚠️  检查nvcc时出错: {e}")

# 3. 检查环境变量
print("\n【3. CUDA环境变量检查】")
cuda_path = os.environ.get('CUDA_PATH')
cuda_path_v11 = os.environ.get('CUDA_PATH_V11_8')
cuda_path_v12 = os.environ.get('CUDA_PATH_V12_1')
path_env = os.environ.get('PATH', '')

if cuda_path:
    print(f"✅ CUDA_PATH: {cuda_path}")
else:
    print("⚠️  CUDA_PATH环境变量未设置")

if cuda_path_v11 or cuda_path_v12:
    print(f"   检测到CUDA版本路径: {cuda_path_v11 or cuda_path_v12}")

# 检查PATH中是否有CUDA
if 'CUDA' in path_env.upper() or 'cuda' in path_env:
    print("✅ PATH中包含CUDA相关路径")
else:
    print("⚠️  PATH中未发现CUDA路径")

# 4. 检查PyTorch
print("\n【4. PyTorch CUDA支持检查】")
try:
    import torch
    print(f"   PyTorch版本: {torch.__version__}")
    
    if '+cpu' in torch.__version__:
        print("   ❌ PyTorch是CPU版本（没有CUDA支持）")
        print("   ⚠️  即使系统有CUDA，CPU版本的PyTorch也无法使用GPU")
    elif '+cu' in torch.__version__:
        cuda_version_in_torch = torch.__version__.split('+cu')[1].split('+')[0]
        print(f"   ✅ PyTorch包含CUDA支持: CUDA {cuda_version_in_torch}")
    else:
        print("   ⚠️  无法从版本号判断PyTorch是否支持CUDA")
    
    print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("   ✅ PyTorch可以访问GPU")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("   ❌ PyTorch无法访问GPU")
        
        # 诊断原因
        print("\n   可能的原因:")
        if '+cpu' in torch.__version__:
            print("   1. PyTorch安装的是CPU版本（最可能）")
            print("      → 需要安装GPU版本的PyTorch")
        elif not cuda_toolkit_installed:
            print("   2. CUDA Toolkit未安装或版本不匹配")
        else:
            print("   2. CUDA版本不匹配")
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                print(f"      PyTorch编译时使用的CUDA: {torch.version.cuda}")
        
except ImportError:
    print("   ❌ PyTorch未安装")

# 5. 检查PyTorch能否找到CUDA库
print("\n【5. PyTorch CUDA库检查】")
try:
    import torch
    if hasattr(torch.version, 'cuda'):
        if torch.version.cuda:
            print(f"   PyTorch编译时CUDA版本: {torch.version.cuda}")
        else:
            print("   ⚠️  PyTorch编译时未包含CUDA支持")
    
    # 尝试加载CUDA
    try:
        torch.cuda.init()
        print("   ✅ PyTorch可以初始化CUDA")
    except Exception as e:
        print(f"   ❌ PyTorch无法初始化CUDA: {e}")
        
except Exception as e:
    print(f"   ❌ 检查时出错: {e}")

# 6. 总结和建议
print("\n" + "=" * 70)
print("【总结】")
print("=" * 70)

try:
    import torch
    if torch.cuda.is_available():
        print("✅ 一切正常！PyTorch可以使用GPU")
    else:
        if '+cpu' in torch.__version__:
            print("❌ 问题: PyTorch安装的是CPU版本")
            print("\n解决方案:")
            print("1. 卸载CPU版本:")
            print("   pip uninstall torch torchvision")
            print("2. 安装GPU版本（根据您的驱动CUDA 12.7，推荐安装CUDA 12.1）:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print("\n注意: 有NVIDIA驱动就够了，通常不需要完整安装CUDA Toolkit")
        else:
            print("❌ PyTorch无法使用GPU，可能需要:")
            print("   1. 检查CUDA版本兼容性")
            print("   2. 重新安装GPU版本的PyTorch")
except:
    print("请先安装PyTorch")

print("=" * 70)

