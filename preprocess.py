"""
MIT-BIH数据预处理模块
包括信号去噪、心拍分割、特征提取等
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt
import pywt
from typing import Tuple, List

class ECGBeatPreprocessor:
    """ECG心拍预处理器"""
    
    def __init__(self, fs: int = 360, beat_length: int = 360):
        """
        初始化预处理器
        
        Args:
            fs: 采样率 (Hz)
            beat_length: 每个心拍的长度（采样点数）
        """
        self.fs = fs
        self.beat_length = beat_length
        self.half_length = beat_length // 2
    
    def denoise_signal(self, ecg_signal: np.ndarray, method: str = 'wavelet') -> np.ndarray:
        """
        对ECG信号进行去噪
        
        Args:
            ecg_signal: 原始ECG信号
            method: 去噪方法 ('wavelet' 或 'filter')
            
        Returns:
            denoised_signal: 去噪后的信号
        """
        if method == 'wavelet':
            # 小波去噪
            return self._wavelet_denoise(ecg_signal)
        elif method == 'filter':
            # 带通滤波
            return self._bandpass_filter(ecg_signal)
        else:
            return ecg_signal
    
    def _wavelet_denoise(self, signal: np.ndarray) -> np.ndarray:
        """小波去噪"""
        # 使用Daubechies小波，level根据信号长度自适应
        wavelet = 'db6'
        max_level = pywt.dwt_max_level(len(signal), wavelet)
        level = min(9, max_level)
        
        # 小波分解
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # 计算阈值并去噪
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # 对细节系数进行软阈值处理
        coeffs_thresh = [coeffs[0]]
        for i in range(1, len(coeffs)):
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        
        # 小波重构
        denoised = pywt.waverec(coeffs_thresh, wavelet)
        
        # 确保长度一致
        if len(denoised) > len(signal):
            denoised = denoised[:len(signal)]
        elif len(denoised) < len(signal):
            denoised = np.pad(denoised, (0, len(signal) - len(denoised)), 'constant')
        
        return denoised
    
    def _bandpass_filter(self, signal: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
        """带通滤波器"""
        nyquist = self.fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # 设计Butterworth带通滤波器
        b, a = butter(4, [low, high], btype='band')
        
        # 应用滤波器
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal
    
    def extract_beats(self, ecg_signal: np.ndarray, r_peaks: np.ndarray, 
                     channel: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        从ECG信号中提取心拍
        
        Args:
            ecg_signal: ECG信号 (n_samples, n_channels) 或 (n_samples,)
            r_peaks: R波位置数组
            channel: 使用的通道索引（如果信号是多通道）
            
        Returns:
            beats: 心拍数组 (n_beats, beat_length)
            valid_indices: 有效心拍的索引
        """
        # 处理多通道信号
        if ecg_signal.ndim > 1:
            signal = ecg_signal[:, channel]
        else:
            signal = ecg_signal
        
        beats = []
        valid_indices = []
        
        for i, r_peak in enumerate(r_peaks):
            # 确保R波位置在有效范围内
            start = r_peak - self.half_length
            end = r_peak + self.half_length
            
            if start >= 0 and end < len(signal):
                beat = signal[start:end]
                # 确保长度正确
                if len(beat) == self.beat_length:
                    beats.append(beat)
                    valid_indices.append(i)
        
        if len(beats) == 0:
            return np.array([]), np.array([])
        
        return np.array(beats), np.array(valid_indices)
    
    def normalize_beats(self, beats: np.ndarray) -> np.ndarray:
        """
        标准化心拍数据
        
        Args:
            beats: 心拍数组 (n_beats, beat_length)
            
        Returns:
            normalized_beats: 标准化后的心拍数组
        """
        # Z-score标准化
        mean = np.mean(beats, axis=1, keepdims=True)
        std = np.std(beats, axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)  # 避免除零
        normalized = (beats - mean) / std
        
        return normalized
    
    def process_record(self, ecg_signal: np.ndarray, r_peaks: np.ndarray,
                      channel: int = 0, denoise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单条记录：去噪 -> 提取心拍 -> 标准化
        
        Args:
            ecg_signal: ECG信号
            r_peaks: R波位置
            channel: 通道索引
            denoise: 是否去噪
            
        Returns:
            processed_beats: 处理后的心拍 (n_beats, beat_length)
            valid_indices: 有效心拍索引
        """
        # 处理多通道信号
        if ecg_signal.ndim > 1:
            signal = ecg_signal[:, channel]
        else:
            signal = ecg_signal
        
        # 去噪
        if denoise:
            signal = self.denoise_signal(signal, method='wavelet')
        
        # 提取心拍
        beats, valid_indices = self.extract_beats(signal, r_peaks, channel=0)
        
        if len(beats) == 0:
            return np.array([]), np.array([])
        
        # 标准化
        beats = self.normalize_beats(beats)
        
        return beats, valid_indices

if __name__ == "__main__":
    # 测试预处理
    print("ECG预处理模块测试")
    print("请使用data_loader.py加载数据后，使用此模块进行预处理")

