# GPU版本PyTorch安装指南

根据您的诊断结果，您的系统已安装NVIDIA驱动（CUDA Version 12.7），但PyTorch安装的是CPU版本。

## 快速安装（推荐）

### 方法1：使用自动安装脚本

```bash
python install_gpu_pytorch.py
```

脚本会自动：
1. 检测CUDA版本
2. 卸载CPU版本的PyTorch
3. 安装GPU版本的PyTorch

### 方法2：手动安装

根据您的系统（CUDA 12.7），安装CUDA 12.1版本的PyTorch（兼容12.7）：

```bash
# 1. 卸载CPU版本
pip uninstall torch torchvision

# 2. 安装GPU版本（CUDA 12.1，兼容12.7）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 验证安装

安装完成后，运行验证：

```bash
python check_gpu.py
```

应该看到：
```
torch.cuda.is_available(): True
✅ CUDA可用！
GPU 0: [您的GPU名称]
```

## 如果安装失败

### 问题1：网络连接问题

如果下载速度慢或失败，可以：
1. 使用国内镜像源（如果可用）
2. 检查网络连接
3. 重试安装命令

### 问题2：版本冲突

如果遇到依赖冲突：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall
```

### 问题3：CUDA版本不匹配

如果PyTorch安装成功但CUDA仍不可用，可能需要：
1. 检查CUDA Toolkit是否正确安装
2. 确保PyTorch的CUDA版本与系统兼容（通常PyTorch CUDA 12.1可以兼容CUDA 12.7）

## 其他CUDA版本

如果您的系统是其他CUDA版本：

**CUDA 11.8:**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.4:**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 安装完成后

安装完成后，运行训练脚本：

```bash
python train.py
```

应该会看到GPU信息：
```
检测到GPU: [GPU名称]
GPU内存: [内存大小] GB
使用设备: cuda:0
```

## 参考链接

- PyTorch官方安装页面: https://pytorch.org/get-started/locally/
- CUDA兼容性说明: https://pytorch.org/get-started/previous-versions/

