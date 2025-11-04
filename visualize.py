"""
ECG数据可视化模块
使用matplotlib绘制ECG信号、心拍、训练过程等
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ECGVisualizer:
    """ECG数据可视化器"""
    
    def __init__(self, fs=360):
        """
        初始化可视化器
        
        Args:
            fs: 采样率 (Hz)
        """
        self.fs = fs
    
    def plot_signal(self, signal: np.ndarray, title: str = "ECG信号", 
                   save_path: str = None, duration: float = None):
        """
        绘制ECG信号
        
        Args:
            signal: ECG信号数组
            title: 图表标题
            save_path: 保存路径
            duration: 显示时长（秒），None表示显示全部
        """
        if duration:
            n_samples = int(duration * self.fs)
            signal = signal[:n_samples]
        
        time = np.arange(len(signal)) / self.fs
        
        plt.figure(figsize=(15, 5))
        plt.plot(time, signal, linewidth=0.8, color='#2E86AB')
        plt.xlabel('时间 (秒)', fontsize=12)
        plt.ylabel('幅值 (mV)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"信号图已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_signal_with_rpeaks(self, signal: np.ndarray, r_peaks: np.ndarray,
                               title: str = "ECG信号与R波", save_path: str = None,
                               duration: float = 10.0):
        """
        绘制ECG信号并标记R波位置
        
        Args:
            signal: ECG信号
            r_peaks: R波位置数组
            title: 图表标题
            save_path: 保存路径
            duration: 显示时长（秒）
        """
        n_samples = int(duration * self.fs)
        signal_segment = signal[:n_samples]
        time = np.arange(len(signal_segment)) / self.fs
        
        # 过滤R波位置
        r_peaks_in_range = r_peaks[r_peaks < n_samples]
        r_times = r_peaks_in_range / self.fs
        r_values = signal[r_peaks_in_range]
        
        plt.figure(figsize=(15, 6))
        plt.plot(time, signal_segment, linewidth=0.8, color='#2E86AB', label='ECG信号')
        plt.scatter(r_times, r_values, color='red', s=50, zorder=5, 
                   marker='v', label='R波')
        
        # 添加垂直线标记R波
        for r_time in r_times:
            plt.axvline(x=r_time, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        
        plt.xlabel('时间 (秒)', fontsize=12)
        plt.ylabel('幅值 (mV)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"信号与R波图已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_beats(self, beats: np.ndarray, labels: np.ndarray = None,
                   n_samples: int = 20, title: str = "心拍示例",
                   save_path: str = None):
        """
        绘制多个心拍
        
        Args:
            beats: 心拍数组 (n_beats, beat_length)
            labels: 标签数组（可选）
            n_samples: 显示的心拍数量
            title: 图表标题
            save_path: 保存路径
        """
        n_samples = min(n_samples, len(beats))
        indices = np.linspace(0, len(beats) - 1, n_samples, dtype=int)
        
        time = np.arange(beats.shape[1]) / self.fs
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2 * n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            axes[i].plot(time, beats[idx], linewidth=1.5, color='#2E86AB')
            
            if labels is not None:
                label_name = f"标签: {labels[idx]}"
                axes[i].set_title(label_name, fontsize=10)
            
            axes[i].set_ylabel('幅值', fontsize=9)
            axes[i].grid(True, alpha=0.3)
            
            if i == len(indices) - 1:
                axes[i].set_xlabel('时间 (秒)', fontsize=10)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"心拍图已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_beat_annotation(self, beat: np.ndarray, title: str = "心拍标注",
                            save_path: str = None):
        """
        绘制单个心拍并标注P波、QRS波群、T波
        
        Args:
            beat: 单个心拍数组
            title: 图表标题
            save_path: 保存路径
        """
        time = np.arange(len(beat)) / self.fs
        
        # 找到R波位置（最大值）
        r_peak_idx = np.argmax(beat)
        r_peak_time = r_peak_idx / self.fs
        
        plt.figure(figsize=(12, 6))
        plt.plot(time, beat, linewidth=2, color='#2E86AB', label='ECG信号')
        
        # 标记R波
        plt.scatter([r_peak_time], [beat[r_peak_idx]], color='red', 
                   s=100, zorder=5, marker='v', label='R波')
        
        # 估计QRS波群位置（R波前后各约0.08秒）
        qrs_start = max(0, r_peak_idx - int(0.08 * self.fs))
        qrs_end = min(len(beat), r_peak_idx + int(0.08 * self.fs))
        
        # 估计P波位置（R波前约0.2秒）
        p_start = max(0, r_peak_idx - int(0.25 * self.fs))
        p_end = max(0, r_peak_idx - int(0.15 * self.fs))
        
        # 估计T波位置（R波后约0.2秒）
        t_start = min(len(beat), r_peak_idx + int(0.15 * self.fs))
        t_end = min(len(beat), r_peak_idx + int(0.35 * self.fs))
        
        # 添加文本标注
        plt.text(time[p_start], beat[p_start] + 0.1, 'P波', 
                fontsize=10, color='green', fontweight='bold')
        plt.text(time[qrs_start], beat[qrs_start] - 0.2, 'QRS波群', 
                fontsize=10, color='red', fontweight='bold')
        plt.text(time[t_start], beat[t_start] + 0.1, 'T波', 
                fontsize=10, color='orange', fontweight='bold')
        
        # 添加区域高亮
        plt.axvspan(time[p_start], time[p_end], alpha=0.2, color='green', label='P波区域')
        plt.axvspan(time[qrs_start], time[qrs_end], alpha=0.2, color='red', label='QRS波群区域')
        plt.axvspan(time[t_start], time[t_end], alpha=0.2, color='orange', label='T波区域')
        
        plt.xlabel('时间 (秒)', fontsize=12)
        plt.ylabel('幅值 (mV)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"心拍标注图已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_class_distribution(self, labels: np.ndarray, label_names: dict = None,
                               title: str = "类别分布", save_path: str = None):
        """
        绘制类别分布图
        
        Args:
            labels: 标签数组
            label_names: 标签名称字典
            title: 图表标题
            save_path: 保存路径
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if label_names:
            labels_str = [label_names.get(l, f'类别{l}') for l in unique_labels]
        else:
            labels_str = [f'类别{l}' for l in unique_labels]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels_str, counts, color='#2E86AB', alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({100*count/len(labels):.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('类别', fontsize=12)
        plt.ylabel('样本数量', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"类别分布图已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_beats_by_class(self, beats: np.ndarray, labels: np.ndarray,
                           label_names: dict = None, n_per_class: int = 5,
                           title: str = "各类别心拍示例", save_path: str = None):
        """
        按类别绘制心拍示例
        
        Args:
            beats: 心拍数组
            labels: 标签数组
            label_names: 标签名称字典
            n_per_class: 每个类别显示的心拍数
            title: 图表标题
            save_path: 保存路径
        """
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        time = np.arange(beats.shape[1]) / self.fs
        
        fig, axes = plt.subplots(n_classes, 1, figsize=(12, 3 * n_classes))
        if n_classes == 1:
            axes = [axes]
        
        for i, label in enumerate(unique_labels):
            class_beats = beats[labels == label]
            n_samples = min(n_per_class, len(class_beats))
            indices = np.random.choice(len(class_beats), n_samples, replace=False)
            
            for idx in indices:
                axes[i].plot(time, class_beats[idx], linewidth=1, alpha=0.7)
            
            label_name = label_names.get(label, f'类别{label}') if label_names else f'类别{label}'
            axes[i].set_title(f'{label_name} (共{len(class_beats)}个样本)', 
                            fontsize=11, fontweight='bold')
            axes[i].set_ylabel('幅值', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            
            if i == len(unique_labels) - 1:
                axes[i].set_xlabel('时间 (秒)', fontsize=10)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分类心拍图已保存到: {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    print("ECG可视化模块")
    print("请使用data_loader.py加载数据后，使用此模块进行可视化")

