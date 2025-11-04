"""
MIT-BIH数据集使用示例
演示如何加载数据、预处理、可视化和训练模型
"""
import numpy as np
from data_loader import MITBIHLoader
from preprocess import ECGBeatPreprocessor
from visualize import ECGVisualizer
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def example_load_and_visualize():
    """示例：加载数据并可视化"""
    print("=" * 60)
    print("示例1: 加载MIT-BIH数据并可视化")
    print("=" * 60)
    
    # 1. 初始化数据加载器
    loader = MITBIHLoader("heart_data")
    print(f"发现 {len(loader.records)} 条记录")
    
    # 2. 加载第一条记录
    if loader.records:
        record_name = loader.records[0]
        print(f"\n加载记录: {record_name}")
        signal, record_info, ann_info = loader.load_record(record_name)
        
        if signal is not None:
            print(f"采样率: {record_info['fs']} Hz")
            print(f"信号长度: {record_info['sig_len']} 采样点")
            print(f"通道数: {record_info['n_sig']}")
            print(f"R波数量: {len(ann_info['sample'])}")
            print(f"心跳类型: {np.unique(ann_info['symbol'])}")
            
            # 3. 可视化
            visualizer = ECGVisualizer(fs=record_info['fs'])
            
            # 可视化前10秒的信号
            print("\n生成可视化图表...")
            visualizer.plot_signal(
                signal[:, 0], 
                title=f"记录{record_name} - ECG信号（前10秒）",
                duration=10.0,
                save_path=f"ecg_signal_{record_name}.png"
            )
            
            # 可视化信号和R波
            visualizer.plot_signal_with_rpeaks(
                signal[:, 0],
                ann_info['sample'],
                title=f"记录{record_name} - ECG信号与R波标记（前10秒）",
                duration=10.0,
                save_path=f"ecg_with_rpeaks_{record_name}.png"
            )

def example_preprocess():
    """示例：预处理数据"""
    print("\n" + "=" * 60)
    print("示例2: 数据预处理")
    print("=" * 60)
    
    # 1. 加载数据
    loader = MITBIHLoader("heart_data")
    if not loader.records:
        print("未找到数据记录！")
        return
    
    record_name = loader.records[0]
    signal, record_info, ann_info = loader.load_record(record_name)
    
    if signal is None:
        print("数据加载失败！")
        return
    
    # 2. 预处理
    preprocessor = ECGBeatPreprocessor(fs=360, beat_length=360)
    beats, valid_indices = preprocessor.process_record(
        signal, ann_info['sample'], channel=0, denoise=True
    )
    
    print(f"\n提取到 {len(beats)} 个有效心拍")
    print(f"心拍形状: {beats.shape}")
    
    # 3. 获取对应的标签
    symbols_array = np.array(ann_info['symbol'])
    symbols = symbols_array[valid_indices]
    label_map = loader.get_heartbeat_labels(symbols)
    
    # 处理符号类型（可能是bytes）
    processed_symbols = []
    for sym in symbols:
        if isinstance(sym, bytes):
            processed_symbols.append(sym.decode('utf-8'))
        elif isinstance(sym, np.bytes_):
            processed_symbols.append(str(sym))
        else:
            processed_symbols.append(str(sym))
    
    labels = np.array([label_map.get(sym, -1) for sym in processed_symbols])
    
    # 过滤无效标签
    valid_mask = labels >= 0
    beats = beats[valid_mask]
    labels = labels[valid_mask]
    
    print(f"有效标签心拍数: {len(beats)}")
    print(f"标签分布: {np.unique(labels, return_counts=True)}")
    
    # 4. 可视化心拍
    visualizer = ECGVisualizer(fs=360)
    
    # 可视化单个心拍并标注
    if len(beats) > 0:
        visualizer.plot_beat_annotation(
            beats[0],
            title="心拍示例（标注P波、QRS波群、T波）",
            save_path="beat_annotation_example.png"
        )
        
        # 可视化多个心拍
        visualizer.plot_beats(
            beats[:20],
            labels[:20] if len(labels) > 0 else None,
            n_samples=20,
            title="心拍示例",
            save_path="beats_example.png"
        )
        
        # 按类别可视化
        if len(np.unique(labels)) > 0:
            visualizer.plot_beats_by_class(
                beats,
                labels,
                label_names={v: k for k, v in label_map.items()},
                n_per_class=5,
                title="各类别心拍示例",
                save_path="beats_by_class.png"
            )
            
            # 类别分布
            visualizer.plot_class_distribution(
                labels,
                label_names={v: k for k, v in label_map.items()},
                title="心拍类别分布",
                save_path="class_distribution.png"
            )

def example_batch_process():
    """示例：批量处理多条记录"""
    print("\n" + "=" * 60)
    print("示例3: 批量处理多条记录")
    print("=" * 60)
    
    # 1. 加载多条记录
    loader = MITBIHLoader("heart_data")
    records = loader.load_all_records(max_records=5)  # 只加载前5条记录
    
    # 2. 预处理
    preprocessor = ECGBeatPreprocessor(fs=360, beat_length=360)
    label_map = loader.get_heartbeat_labels([])
    
    all_beats = []
    all_labels = []
    target_classes = ['N', 'L', 'R', 'A', 'V']
    
    for signal, record_info, ann_info in records:
        if signal is None or ann_info is None:
            continue
        
        beats, valid_indices = preprocessor.process_record(
            signal, ann_info['sample'], channel=0, denoise=True
        )
        
        if len(beats) == 0:
            continue
        
        symbols_array = np.array(ann_info['symbol'])
        symbols = symbols_array[valid_indices]
        labels = []
        for sym in symbols:
            # 处理字节字符串（如果wfdb返回的是bytes）
            if isinstance(sym, bytes):
                sym = sym.decode('utf-8')
            elif isinstance(sym, np.bytes_):
                sym = str(sym)
            else:
                sym = str(sym)
            
            if sym in target_classes and sym in label_map:
                labels.append(label_map[sym])
            else:
                labels.append(-1)
        
        labels = np.array(labels)
        valid_mask = labels >= 0
        
        if np.any(valid_mask):
            all_beats.append(beats[valid_mask])
            all_labels.append(labels[valid_mask])
    
    if len(all_beats) > 0:
        all_beats = np.vstack(all_beats)
        all_labels = np.concatenate(all_labels)
        
        print(f"\n总共提取到 {len(all_beats)} 个有效心拍")
        print(f"类别分布:")
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        label_names = {v: k for k, v in label_map.items()}
        for label, count in zip(unique_labels, counts):
            name = label_names.get(label, f'类别{label}')
            print(f"  {name}: {count} 个")
        
        # 可视化整体类别分布
        visualizer = ECGVisualizer(fs=360)
        visualizer.plot_class_distribution(
            all_labels,
            label_names={v: k for k, v in label_map.items()},
            title="所有记录的类别分布",
            save_path="all_class_distribution.png"
        )

if __name__ == "__main__":
    print("MIT-BIH数据集使用示例")
    print("=" * 60)
    
    try:
        # 示例1: 加载和可视化
        example_load_and_visualize()
        
        # 示例2: 预处理
        example_preprocess()
        
        # 示例3: 批量处理
        example_batch_process()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

