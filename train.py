"""
ECG心律失常分类模型训练脚本
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_loader import MITBIHLoader
from preprocess import ECGBeatPreprocessor
from model import get_model

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ECGDataset(Dataset):
    """ECG数据集类"""
    
    def __init__(self, beats: np.ndarray, labels: np.ndarray):
        """
        初始化数据集
        
        Args:
            beats: 心拍数组 (n_beats, beat_length)
            labels: 标签数组 (n_beats,)
        """
        self.beats = torch.FloatTensor(beats).unsqueeze(1)  # (n_beats, 1, beat_length)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.beats)
    
    def __getitem__(self, idx):
        return self.beats[idx], self.labels[idx]


class ECGClassifierTrainer:
    """ECG分类器训练器"""
    
    def __init__(self, model, device=None):
        """
        初始化训练器
        
        Args:
            model: PyTorch模型
            device: 训练设备，None表示自动选择
        """
        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                device = 'cpu'
                print("未检测到GPU，使用CPU训练")
        
        self.device = device
        self.model = model.to(self.device)
        
        # 如果使用GPU，打印设备信息
        if self.device == 'cuda':
            print(f"使用设备: {self.device}:{torch.cuda.current_device()}")
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for beats, labels in tqdm(train_loader, desc='训练中'):
            beats = beats.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(beats)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for beats, labels in tqdm(val_loader, desc='验证中'):
                beats = beats.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(beats)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=50, 
              learning_rate=0.001, weight_decay=1e-5):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # 注意：verbose参数在某些PyTorch版本中可能不支持，改用手动打印
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=5)
        
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 10
        
        print(f"\n开始训练，使用设备: {self.device}")
        if self.device == 'cuda':
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc, _, _ = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 学习率调度
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # 如果学习率发生变化，打印信息
            if new_lr < old_lr:
                print(f"学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")
            
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"保存最佳模型 (验证准确率: {best_val_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"早停：验证准确率连续{max_patience}轮未提升")
                    break
    
    def evaluate(self, test_loader):
        """评估模型"""
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, all_preds, all_labels = self.validate(test_loader, criterion)
        
        # 计算详细指标
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"\n测试集评估结果:")
        print(f"损失: {test_loss:.4f}")
        print(f"准确率: {test_acc:.2f}%")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        return {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失', linewidth=2)
        ax1.plot(self.val_losses, label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练过程损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(self.train_accs, label='训练准确率', linewidth=2)
        ax2.plot(self.val_accs, label='验证准确率', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title('训练过程准确率曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")


def prepare_data(data_dir='heart_data', max_records=None, 
                 target_classes=['N', 'L', 'R', 'A', 'V'], 
                 beat_length=360):
    """
    准备训练数据
    
    Args:
        data_dir: 数据目录
        max_records: 最大加载记录数
        target_classes: 目标类别列表
        beat_length: 心拍长度
        
    Returns:
        beats: 心拍数组
        labels: 标签数组
        label_encoder: 标签编码器
    """
    print("=" * 60)
    print("步骤1: 加载MIT-BIH数据集")
    print("=" * 60)
    
    # 加载数据
    loader = MITBIHLoader(data_dir)
    records = loader.load_all_records(max_records=max_records)
    
    print("\n" + "=" * 60)
    print("步骤2: 预处理数据")
    print("=" * 60)
    
    # 初始化预处理器
    preprocessor = ECGBeatPreprocessor(fs=360, beat_length=beat_length)
    
    # 获取标签映射
    label_map = loader.get_heartbeat_labels([])
    
    all_beats = []
    all_labels = []
    
    for signal, record_info, ann_info in tqdm(records, desc='处理记录'):
        if signal is None or ann_info is None:
            continue
        
        # 提取心拍
        beats, valid_indices = preprocessor.process_record(
            signal, ann_info['sample'], channel=0, denoise=True
        )
        
        if len(beats) == 0:
            continue
        
        # 获取对应的标签
        # 确保symbol是numpy数组，并使用正确的索引方式
        symbols_array = np.array(ann_info['symbol'])
        symbols = symbols_array[valid_indices]
        labels = []
        
        for sym in symbols:
            # 处理字节字符串（如果wfdb返回的是bytes）
            if isinstance(sym, bytes):
                sym = sym.decode('utf-8')
            elif isinstance(sym, np.bytes_):
                sym = str(sym)
            
            if sym in target_classes and sym in label_map:
                labels.append(label_map[sym])
            else:
                # 如果不在目标类别中，标记为-1（后续过滤）
                labels.append(-1)
        
        labels = np.array(labels)
        
        # 只保留目标类别的数据
        valid_mask = labels >= 0
        if np.any(valid_mask):
            all_beats.append(beats[valid_mask])
            all_labels.append(labels[valid_mask])
    
    if len(all_beats) == 0:
        raise ValueError("未能提取到有效的心拍数据！")
    
    # 合并所有数据
    all_beats = np.vstack(all_beats)
    all_labels = np.concatenate(all_labels)
    
    print(f"\n提取到 {len(all_beats)} 个心拍")
    print(f"类别分布:")
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    label_names = {v: k for k, v in label_map.items()}
    for label, count in zip(unique_labels, counts):
        name = label_names.get(label, f'类别{label}')
        print(f"  {name} (标签{label}): {count} 个")
    
    # 创建标签编码器（将标签映射到0,1,2,...）
    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(all_labels)
    
    return all_beats, all_labels, label_encoder


def main():
    """主函数"""
    # 配置参数
    config = {
        'data_dir': 'heart_data',
        'max_records': None,  # None表示加载所有记录
        'target_classes': ['N', 'L', 'R', 'A', 'V'],  # 常见的心律失常类型
        'beat_length': 360,
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'model_type': 'cnn',  # 'cnn' 或 'cnn_lstm'
        'test_size': 0.2,
        'val_size': 0.1,
    }
    
    # 准备数据
    beats, labels, label_encoder = prepare_data(
        data_dir=config['data_dir'],
        max_records=config['max_records'],
        target_classes=config['target_classes'],
        beat_length=config['beat_length']
    )
    
    num_classes = len(np.unique(labels))
    print(f"\n分类任务: {num_classes} 个类别")
    
    # 划分数据集
    print("\n" + "=" * 60)
    print("步骤3: 划分数据集")
    print("=" * 60)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        beats, labels, test_size=config['test_size'] + config['val_size'], 
        random_state=42, stratify=labels
    )
    
    val_size_adjusted = config['val_size'] / (config['test_size'] + config['val_size'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_size_adjusted, 
        random_state=42, stratify=y_temp
    )
    
    print(f"训练集: {len(X_train)} 个样本")
    print(f"验证集: {len(X_val)} 个样本")
    print(f"测试集: {len(X_test)} 个样本")
    
    # 创建数据加载器
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)
    
    # 根据是否有GPU设置num_workers（Windows上建议使用0，Linux可以设置更大）
    import platform
    if platform.system() == 'Windows':
        num_workers = 0  # Windows上多进程可能有问题
    else:
        num_workers = 4 if torch.cuda.is_available() else 2  # GPU训练时使用更多worker
    
    # 使用pin_memory加速GPU数据传输
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=num_workers, 
                             pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, num_workers=num_workers,
                          pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)
    
    # 创建模型
    print("\n" + "=" * 60)
    print("步骤4: 创建模型")
    print("=" * 60)
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"✓ CUDA版本: {torch.version.cuda}")
    else:
        print("⚠ 未检测到GPU，将使用CPU训练（速度较慢）")
    
    model = get_model(
        model_type=config['model_type'],
        num_classes=num_classes,
        input_length=config['beat_length']
    )
    
    # 训练模型
    print("\n" + "=" * 60)
    print("步骤5: 训练模型")
    print("=" * 60)
    
    trainer = ECGClassifierTrainer(model)
    trainer.train(
        train_loader, val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate']
    )
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    # 加载最佳模型并评估
    print("\n" + "=" * 60)
    print("步骤6: 评估模型")
    print("=" * 60)
    
    model.load_state_dict(torch.load('best_model.pth'))
    results = trainer.evaluate(test_loader)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(results['confusion_matrix'], 
                         label_encoder.classes_,
                         save_path='confusion_matrix.png')
    
    print("\n训练完成！")
    return trainer, results


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {save_path}")


if __name__ == "__main__":
    main()

