import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    """注意力机制模块"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / np.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        
    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch_size, hidden_size]
        encoder_outputs: [batch_size, seq_len, hidden_size]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # 扩展hidden为 [batch_size, seq_len, hidden_size]
        h = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # 计算注意力权重
        attn_energies = self.score(h, encoder_outputs)  # [batch_size, seq_len]
        
        # 归一化注意力权重
        return F.softmax(attn_energies, dim=1)
        
    def score(self, hidden, encoder_outputs):
        # [batch_size, seq_len, 2*hidden_size]
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        
        # [batch_size, seq_len]
        energy = energy.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [batch_size, 1, hidden_size]
        energy = torch.bmm(v, energy)  # [batch_size, 1, seq_len]
        return energy.squeeze(1)  # [batch_size, seq_len]

class EnhancedLSTMModel(nn.Module):
    """增强型双向LSTM模型，包含注意力机制"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = Attention(hidden_size)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 批归一化层
        self.bn = nn.BatchNorm1d(hidden_size)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size = x.size(0)
        
        # LSTM层: [batch_size, seq_len, 2*hidden_size]
        lstm_out, _ = self.lstm(x)
        
        # 获取最后一个时间步的隐藏状态
        h_n = lstm_out[:, -1, :self.hidden_size]  # 前向LSTM的最后输出
        
        # 使用注意力机制
        attn_weights = self.attention(h_n, lstm_out)  # [batch_size, seq_len]
        
        # 将注意力权重应用到LSTM输出
        context = attn_weights.unsqueeze(1).bmm(lstm_out).squeeze(1)  # [batch_size, 2*hidden_size]
        
        # 全连接层
        out = self.fc1(context)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

class TransformerModel(nn.Module):
    """基于Transformer架构的预测模型"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, n_heads=4, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        
        # 投影到hidden_size维度
        x = self.input_projection(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        
        # 应用批归一化
        x = self.bn1(x)
        
        # 全连接层
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

class PositionalEncoding(nn.Module):
    """Transformer位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class HybridModel(nn.Module):
    """混合模型：结合LSTM和Transformer的优点"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, n_heads=4, dropout=0.2):
        super(HybridModel, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers//2,  # 减少LSTM层数
            batch_first=True,
            dropout=dropout if num_layers//2 > 1 else 0,
            bidirectional=True
        )
        
        # Transformer部分
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size*2,  # 双向LSTM输出
            nhead=n_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers//2)
        
        # 注意力机制
        self.attention = Attention(hidden_size*2)
        
        # 输出层
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_size*2)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        
        # LSTM层
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, 2*hidden_size]
        
        # Transformer层
        trans_out = self.transformer_encoder(lstm_out)  # [batch_size, seq_len, 2*hidden_size]
        
        # 注意力机制
        h_n = trans_out[:, -1, :]  # 最后时间步输出
        attn_weights = self.attention(h_n, trans_out)  # [batch_size, seq_len]
        context = attn_weights.unsqueeze(1).bmm(trans_out).squeeze(1)  # [batch_size, 2*hidden_size]
        
        # 批归一化
        context = self.bn1(context)
        
        # 全连接层
        out = self.fc1(context)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

class TimeSeriesDataset(torch.utils.data.Dataset):
    """时间序列数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] 