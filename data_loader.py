"""
MIT-BIH数据集加载模块
使用wfdb库读取MIT-BIH心律失常数据库
"""
import wfdb
import numpy as np
import os
from typing import List, Tuple

class MITBIHLoader:
    """MIT-BIH数据集加载器"""
    
    def __init__(self, data_dir: str = "heart_data"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据文件夹路径
        """
        self.data_dir = data_dir
        self.records = []
        self._load_record_list()
    
    def _load_record_list(self):
        """从RECORDS文件加载记录列表"""
        records_file = os.path.join(self.data_dir, "RECORDS")
        if os.path.exists(records_file):
            with open(records_file, 'r') as f:
                self.records = [line.strip() for line in f if line.strip()]
        else:
            # 如果没有RECORDS文件，从文件夹中自动发现
            files = os.listdir(self.data_dir)
            dat_files = [f.replace('.dat', '') for f in files if f.endswith('.dat')]
            self.records = sorted(set(dat_files))
    
    def load_record(self, record_name: str) -> Tuple[np.ndarray, dict, dict]:
        """
        加载单个记录
        
        Args:
            record_name: 记录名称（如'100'）
            
        Returns:
            signal: ECG信号数据 (n_samples, n_channels)
            record_info: 记录信息字典
            annotation: 注释信息字典
        """
        record_path = os.path.join(self.data_dir, record_name)
        
        try:
            # 读取信号数据
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal  # (n_samples, n_channels)
            
            # 读取注释数据
            annotation = wfdb.rdann(record_path, 'atr')
            
            # 提取记录信息
            record_info = {
                'fs': record.fs,  # 采样率
                'sig_len': record.sig_len,  # 信号长度
                'n_sig': record.n_sig,  # 通道数
                'sig_name': record.sig_name,  # 信号名称
                'comments': record.comments
            }
            
            # 提取注释信息
            # 确保symbol是numpy数组格式
            symbol_array = np.array(annotation.symbol)
            ann_info = {
                'sample': np.array(annotation.sample),  # R波位置
                'symbol': symbol_array,  # 心跳类型符号
                'aux_note': annotation.aux_note,  # 辅助注释
                'fs': annotation.fs
            }
            
            return signal, record_info, ann_info
            
        except Exception as e:
            print(f"加载记录 {record_name} 时出错: {e}")
            return None, None, None
    
    def load_all_records(self, max_records: int = None) -> List[Tuple]:
        """
        加载所有记录
        
        Args:
            max_records: 最大加载记录数，None表示加载所有
            
        Returns:
            records: 记录列表，每个元素为(signal, record_info, ann_info)
        """
        all_records = []
        records_to_load = self.records[:max_records] if max_records else self.records
        
        print(f"开始加载 {len(records_to_load)} 条记录...")
        for i, record_name in enumerate(records_to_load):
            print(f"加载进度: {i+1}/{len(records_to_load)} - {record_name}")
            signal, record_info, ann_info = self.load_record(record_name)
            if signal is not None:
                all_records.append((signal, record_info, ann_info))
        
        print(f"成功加载 {len(all_records)} 条记录")
        return all_records
    
    def get_heartbeat_labels(self, symbols: np.ndarray) -> dict:
        """
        获取心跳类型标签映射
        
        MIT-BIH标准心跳类型：
        N: Normal beat (正常心跳)
        L: Left bundle branch block beat (左束支传导阻滞)
        R: Right bundle branch block beat (右束支传导阻滞)
        A: Atrial premature beat (房性早搏)
        V: Premature ventricular contraction (室性早搏)
        /: Paced beat (起搏心跳)
        E: Ventricular escape beat (室性逸搏)
        
        Returns:
            label_map: 标签映射字典
        """
        # 常见的心律失常类型
        label_map = {
            'N': 0,  # 正常心跳
            'L': 1,  # 左束支传导阻滞
            'R': 2,  # 右束支传导阻滞
            'A': 3,  # 房性早搏
            'V': 4,  # 室性早搏
            '/': 5,  # 起搏心跳
            'E': 6,  # 室性逸搏
            'a': 7,  # 融合的房性早搏
            'J': 8,  # 交界性早搏
            'S': 9,  # 室上性早搏
            'e': 10,  # 房性逸搏
            'j': 11,  # 交界性逸搏
        }
        
        return label_map

if __name__ == "__main__":
    # 测试数据加载
    loader = MITBIHLoader("heart_data")
    print(f"发现 {len(loader.records)} 条记录")
    
    # 加载第一条记录进行测试
    if loader.records:
        signal, record_info, ann_info = loader.load_record(loader.records[0])
        if signal is not None:
            print(f"\n记录信息:")
            print(f"采样率: {record_info['fs']} Hz")
            print(f"信号长度: {record_info['sig_len']} 采样点")
            print(f"通道数: {record_info['n_sig']}")
            print(f"信号名称: {record_info['sig_name']}")
            print(f"\n注释信息:")
            print(f"R波数量: {len(ann_info['sample'])}")
            print(f"心跳类型: {np.unique(ann_info['symbol'])}")

