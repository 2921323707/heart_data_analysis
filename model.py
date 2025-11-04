"""
基于PyTorch的ECG心律失常分类模型
包括CNN和CNN-LSTM混合架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGCNN(nn.Module):
    """
    用于ECG心律失常分类的CNN模型
    """
    
    def __init__(self, num_classes: int = 5, input_length: int = 360):
        """
        初始化模型
        
        Args:
            num_classes: 分类类别数
            input_length: 输入信号长度（采样点数）
        """
        super(ECGCNN, self).__init__()
        self.num_classes = num_classes
        self.input_length = input_length
        
        # 第一层卷积块
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 第二层卷积块
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 第三层卷积块
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 第四层卷积块
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算全连接层输入尺寸
        # 经过4次池化，每次长度减半：input_length / 16
        fc_input_size = 256 * (input_length // 16)
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 1, signal_length)
            
        Returns:
            output: 分类输出 (batch_size, num_classes)
        """
        # 卷积层1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 卷积层2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 卷积层3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # 卷积层4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class ECGCNNLSTM(nn.Module):
    """
    CNN-LSTM混合模型，用于ECG心律失常分类
    使用CNN提取空间特征，LSTM捕获时序依赖
    """
    
    def __init__(self, num_classes: int = 5, input_length: int = 360, 
                 hidden_size: int = 128, num_layers: int = 2):
        """
        初始化模型
        
        Args:
            num_classes: 分类类别数
            input_length: 输入信号长度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
        """
        super(ECGCNNLSTM, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # CNN特征提取器
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # LSTM层
        self.lstm = nn.LSTM(128, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 1, signal_length)
            
        Returns:
            output: 分类输出 (batch_size, num_classes)
        """
        # CNN特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 转换维度：从(batch, channels, length)到(batch, length, channels)
        x = x.transpose(1, 2)
        
        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        x = lstm_out[:, -1, :]
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model(model_type: str = 'cnn', num_classes: int = 5, 
              input_length: int = 360, **kwargs):
    """
    获取模型实例
    
    Args:
        model_type: 模型类型 ('cnn' 或 'cnn_lstm')
        num_classes: 分类类别数
        input_length: 输入信号长度
        **kwargs: 其他模型参数
        
    Returns:
        model: 模型实例
    """
    if model_type.lower() == 'cnn':
        model = ECGCNN(num_classes=num_classes, input_length=input_length)
    elif model_type.lower() == 'cnn_lstm':
        hidden_size = kwargs.get('hidden_size', 128)
        num_layers = kwargs.get('num_layers', 2)
        model = ECGCNNLSTM(num_classes=num_classes, input_length=input_length,
                          hidden_size=hidden_size, num_layers=num_layers)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    return model

if __name__ == "__main__":
    # 测试模型
    print("测试ECG分类模型...")
    
    # 测试CNN模型
    model_cnn = get_model('cnn', num_classes=5, input_length=360)
    x = torch.randn(32, 1, 360)  # batch_size=32
    output = model_cnn(x)
    print(f"CNN模型输出形状: {output.shape}")
    print(f"CNN模型参数数量: {sum(p.numel() for p in model_cnn.parameters())}")
    
    # 测试CNN-LSTM模型
    model_cnn_lstm = get_model('cnn_lstm', num_classes=5, input_length=360)
    output = model_cnn_lstm(x)
    print(f"CNN-LSTM模型输出形状: {output.shape}")
    print(f"CNN-LSTM模型参数数量: {sum(p.numel() for p in model_cnn_lstm.parameters())}")

