import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from binance.client import Client
from binance.exceptions import BinanceAPIException
import schedule
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ta
import logging
import pickle
import argparse
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from imblearn.over_sampling import SMOTE
import sys
import select

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bitcoin_prediction.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 配置文件
CONFIG_FILE = 'config.json'

# 默认配置 - 🔧 技术改进：增加历史数据获取范围
DEFAULT_CONFIG = {
    'api_key': '',
    'api_secret': '',
    'symbol': 'BTCUSDT',
    'intervals': ['1m', '5m', '15m'],
    'lookback_hours': 72,  # 🔧 从24小时增加到72小时，确保足够的历史数据
    'prediction_minutes': 10,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'sequence_length': 10,
    'trade_amount': 100,
    'stop_loss': 0.02,
    'take_profit': 0.03,
    'min_confidence_threshold': 60,  # 🔧 新增：最低置信度门槛
    'enhanced_sentiment_enabled': True,  # 🔧 新增：启用增强情绪分析
    'soft_confidence_floor': 15,  # 🔧 新增：置信度软下限
    'fear_greed_weight': 0.1,  # 🔧 新增：恐慌贪婪指数权重
    'bet_amount': 5,  # 固定投注金额为5u
    'payout_ratio': 0.8,  # 事件合约盈利率80%
    'big_trade_threshold': 0.01,  # 大单交易阈值，单位BTC (约1000美元)
}

# 加载或创建配置
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # 合并默认配置和已有配置
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
    else:
        config = DEFAULT_CONFIG
        save_config(config)
    return config

# 保存配置
def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

# LSTM模型定义 - 升级为Bi-LSTM + Attention
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-LSTM层
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True  # 双向LSTM
        )
        
        # Attention机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # 双向LSTM输出维度
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 初始化双向LSTM隐藏状态
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # Bi-LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))  # [batch, seq_len, hidden_size*2]
        
        # Self-Attention机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后时间步的输出
        last_output = attn_out[:, -1, :]  # [batch, hidden_size*2]
        
        # 特征融合
        fused_features = self.feature_fusion(last_output)
        
        # 最终预测
        output = self.fc(fused_features)
        output = self.sigmoid(output)
        
        return output

# 保持向后兼容
class LSTMModel(EnhancedLSTMModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout)

# 新增：多分类标签系统
class EnhancedLabelProcessor:
    """增强的标签处理器 - 支持多维度分类"""
    
    def __init__(self):
        pass
    
    def create_enhanced_labels(self, df, prediction_minutes=10):
        """创建增强的多维度标签"""
        labels = {}
        
        # 1. 基础涨跌标签 (二分类)
        future_prices = df['close'].shift(-prediction_minutes)
        current_prices = df['close']
        price_changes = (future_prices - current_prices) / current_prices
        
        # 处理NaN值
        valid_mask = ~(price_changes.isna() | future_prices.isna())
        
        labels['direction'] = (price_changes > 0).astype(int)
        labels['direction'] = labels['direction'].fillna(0)  # NaN填充为0
        
        # 2. 涨幅大小标签 (多分类)
        price_change_pct = price_changes * 100
        price_change_pct = price_change_pct.fillna(0)  # 先填充NaN
        
        labels['magnitude'] = pd.cut(price_change_pct, 
                                   bins=[-float('inf'), -0.5, -0.2, 0.2, 0.5, float('inf')],
                                   labels=[0, 1, 2, 3, 4], include_lowest=True)
        labels['magnitude'] = labels['magnitude'].fillna(2).astype(int)  # NaN填充为震荡(2)
        
        # 3. 波动率标签 (三分类)
        rolling_std = df['close'].rolling(20).std()
        volatility_pct = rolling_std / df['close'] * 100
        volatility_pct = volatility_pct.fillna(1)  # 填充默认波动率
        
        labels['volatility'] = pd.cut(volatility_pct,
                                    bins=[0, 1, 2, float('inf')],
                                    labels=[0, 1, 2], include_lowest=True)
        labels['volatility'] = labels['volatility'].fillna(1).astype(int)  # NaN填充为中等波动(1)
        
        # 4. 趋势强度标签 (三分类)
        ma_short = df['close'].rolling(5).mean()
        ma_long = df['close'].rolling(20).mean()
        trend_strength = (ma_short - ma_long) / ma_long * 100
        trend_strength = trend_strength.fillna(0)  # 填充为无趋势
        
        labels['trend'] = pd.cut(trend_strength,
                               bins=[-float('inf'), -1, 1, float('inf')],
                               labels=[0, 1, 2], include_lowest=True)
        labels['trend'] = labels['trend'].fillna(1).astype(int)  # NaN填充为震荡(1)
        
        return labels

# 新增：情绪分析模块 - 🔧 技术改进：更丰富的情绪指标
class MarketSentimentAnalyzer:
    """市场情绪分析器 - 增强版"""
    
    def __init__(self, api):
        self.api = api
        
    def analyze_order_book_sentiment(self, symbol='BTCUSDT'):
        """分析订单簿情绪 - 🔧 增强买墙/卖墙分析"""
        try:
            depth = self.api.get_order_book(symbol, limit=100)
            
            bids = depth.get('bids', [])
            asks = depth.get('asks', [])
            
            if not bids or not asks:
                return self._get_default_sentiment()
            
            # 基础买卖盘分析
            bid_volume = sum([float(bid[1]) for bid in bids[:10]])  # 前10档买单
            ask_volume = sum([float(ask[1]) for ask in asks[:10]])  # 前10档卖单
            
            # 🔧 新增：买墙/卖墙分析
            # 分析不同价格层级的订单分布
            current_price = float(bids[0][0]) if bids else 0
            
            # 买墙分析：距离当前价格1%以内的大额买单
            buy_wall_volume = 0
            for bid in bids:
                price, volume = float(bid[0]), float(bid[1])
                if current_price - price <= current_price * 0.01:  # 1%以内
                    if volume > 10:  # 大额订单
                        buy_wall_volume += volume
            
            # 卖墙分析：距离当前价格1%以内的大额卖单
            sell_wall_volume = 0
            for ask in asks:
                price, volume = float(ask[0]), float(ask[1])
                if price - current_price <= current_price * 0.01:  # 1%以内
                    if volume > 10:  # 大额订单
                        sell_wall_volume += volume
            
            # 🔧 新增：深度不平衡分析
            # 分析5个价格层级的订单分布
            deep_bid_volume = sum([float(bid[1]) for bid in bids[:50]])  # 前50档
            deep_ask_volume = sum([float(ask[1]) for ask in asks[:50]])
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return self._get_default_sentiment()
                
            sentiment_score = bid_volume / total_volume  # 买盘占比
            book_imbalance = (bid_volume - ask_volume) / total_volume
            
            # 🔧 新增：买墙/卖墙强度
            wall_ratio = buy_wall_volume / max(sell_wall_volume, 1)
            deep_imbalance = (deep_bid_volume - deep_ask_volume) / max(deep_bid_volume + deep_ask_volume, 1)
            
            return {
                'sentiment_score': sentiment_score,
                'book_imbalance': book_imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'buy_wall_volume': buy_wall_volume,  # 🔧 新增
                'sell_wall_volume': sell_wall_volume,  # 🔧 新增
                'wall_ratio': wall_ratio,  # 🔧 新增
                'deep_imbalance': deep_imbalance,  # 🔧 新增
            }
        except Exception as e:
            logging.warning(f"⚠️ 订单簿情绪分析失败: {e}")
            return self._get_default_sentiment()
    
    def analyze_funding_rate_sentiment(self, symbol='BTCUSDT'):
        """分析资金费率情绪（需要期货API）"""
        try:
            # 🔧 可以集成币安期货API获取资金费率
            # 资金费率为正表示多头情绪，为负表示空头情绪
            return {'funding_rate': 0.0}  # 占位符
        except:
            return {'funding_rate': 0.0}
    
    def get_fear_greed_index(self):
        """🔧 新增：获取恐慌贪婪指数（模拟实现）"""
        try:
            # 这里可以集成真实的Fear & Greed Index API
            # 例如：Alternative.me API
            # 目前使用基于技术指标的模拟实现
            
            # 获取最近价格数据计算简化的恐慌贪婪指数
            klines = self.api.client.get_klines(symbol='BTCUSDT', interval='1h', limit=24)
            if not klines:
                return 50  # 中性值
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # 简化的恐慌贪婪指数计算
            # 基于价格动量、波动率、成交量等因素
            
            # 1. 价格动量 (0-25分)
            price_change_24h = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            momentum_score = min(25, max(0, (price_change_24h + 0.1) * 125))
            
            # 2. 波动率 (0-25分) - 低波动率表示贪婪
            volatility = df['close'].pct_change().std()
            volatility_score = min(25, max(0, 25 - volatility * 1000))
            
            # 3. 成交量 (0-25分) - 高成交量表示市场活跃
            avg_volume = df['volume'].mean()
            recent_volume = df['volume'].iloc[-6:].mean()  # 最近6小时
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            volume_score = min(25, max(0, volume_ratio * 12.5))
            
            # 4. 市场主导地位 (0-25分) - 简化为固定值
            dominance_score = 12.5  # 中性值
            
            fear_greed_index = momentum_score + volatility_score + volume_score + dominance_score
            fear_greed_index = max(0, min(100, fear_greed_index))
            
            return fear_greed_index
            
        except Exception as e:
            logging.warning(f"⚠️ 恐慌贪婪指数计算失败: {e}")
            return 50  # 返回中性值
    
    def get_social_sentiment(self):
        """🔧 新增：社交媒体情绪分析（占位符）"""
        try:
            # 这里可以集成Twitter API、Telegram频道分析等
            # 目前返回中性值
            return {
                'twitter_sentiment': 0.5,  # 0-1，0.5为中性
                'telegram_mentions': 0,
                'reddit_sentiment': 0.5,
                'social_volume': 0
            }
        except:
            return {
                'twitter_sentiment': 0.5,
                'telegram_mentions': 0,
                'reddit_sentiment': 0.5,
                'social_volume': 0
            }
    
    def _get_default_sentiment(self):
        """返回默认情绪数据"""
        return {
            'sentiment_score': 0.5, 
            'book_imbalance': 0, 
            'bid_volume': 0, 
            'ask_volume': 0,
            'buy_wall_volume': 0,
            'sell_wall_volume': 0,
            'wall_ratio': 1.0,
            'deep_imbalance': 0,
        }

# 数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 增强的币安API交互类
class EnhancedBinanceAPI:
    def __init__(self, api_key, api_secret, config):
        self.client = Client(api_key, api_secret)
        self.config = config
        self.websocket_data = {}
        self.big_trades = []
        self.market_depth = {}
        
    def get_multi_timeframe_data(self, symbol, intervals, start_str):
        """获取多时间周期数据"""
        data = {}
        for interval in intervals:
            try:
                klines = self.client.get_historical_klines(symbol, interval, start_str)
                if klines:
                    data[interval] = klines
                    logging.info(f"📊 获取{interval}数据: {len(klines)}条记录")
            except BinanceAPIException as e:
                logging.error(f"❌ 获取{interval}数据失败: {e}")
        return data
    
    def get_recent_trades(self, symbol, limit=1000):
        """获取最近成交记录，检测大单"""
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            big_trades = []
            
            # 确保config中有big_trade_threshold，如果没有则设置默认值
            if 'big_trade_threshold' not in self.config:
                self.config['big_trade_threshold'] = 0.01  # 默认0.01 BTC为大单阈值
                logging.info(f"⚠️ 未找到大单阈值配置，设置默认值为 {self.config['big_trade_threshold']} BTC")
            
            # 确保阈值在合理范围内（0.001-10 BTC）
            if self.config['big_trade_threshold'] > 10:
                logging.warning(f"⚠️ 大单阈值过高 ({self.config['big_trade_threshold']} BTC)，调整为 0.05 BTC")
                self.config['big_trade_threshold'] = 0.05
            
            for trade in trades:
                qty = float(trade['qty'])
                if qty >= self.config['big_trade_threshold']:
                    big_trades.append({
                        'time': int(trade['time']),
                        'price': float(trade['price']),
                        'qty': qty,
                        'is_buyer_maker': trade['isBuyerMaker']
                    })
            
            if big_trades:
                logging.info(f"📈 检测到 {len(big_trades)} 笔大单交易 (>{self.config['big_trade_threshold']} BTC)")
            else:
                logging.info(f"📊 未检测到大于 {self.config['big_trade_threshold']} BTC 的大单交易")
            
            return big_trades
        except Exception as e:
            logging.warning(f"获取交易数据失败: {e}")
            return []
    
    def get_order_book(self, symbol, limit=100):
        """获取订单簿深度"""
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            
            # 计算买卖盘比率
            bid_volume = sum([float(bid[1]) for bid in depth['bids'][:10]])
            ask_volume = sum([float(ask[1]) for ask in depth['asks'][:10]])
            ratio = bid_volume / ask_volume if ask_volume > 0 else 1
            
            logging.info(f"📈 买卖盘比率: {ratio:.2f}")
            return depth
        except Exception as e:
            logging.warning(f"获取订单簿失败: {e}")
            return {}
    
    def get_latest_price(self, symbol):
        """获取最新价格"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logging.error(f"获取价格失败: {e}")
            return None

# 增强的数据处理器
class EnhancedDataProcessor:
    def __init__(self, config):
        self.config = config
        self.feature_scalers = {}
        self.label_processor = EnhancedLabelProcessor()
        self.sentiment_analyzer = None
        
        # 初始化features属性，防止预测时出错
        self.features = None
        
        # 新增技术指标列表
        self.enhanced_indicators = [
            'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_pct',
            'sma_10', 'sma_20', 'ema_12', 'ema_26', 'adx',
            'stoch_k', 'stoch_d', 'cci', 'williams_r',
            'obv', 'atr', 'volume_sma', 'volume_ratio',
            'price_vs_sma20', 'ema_cross', 'momentum_5', 'momentum_10',
            'volatility_ratio', 'vwap', 'price_vs_vwap', 'high_low_pct',
            'rsi_smooth', 'volume_momentum', 'price_oscillation'
        ]
        
    def fetch_and_prepare_enhanced_data(self, api, retrain=False):
        """获取并准备增强的数据"""
        try:
            if self.sentiment_analyzer is None:
                self.sentiment_analyzer = MarketSentimentAnalyzer(api)
                
            # 获取多时间周期数据
            lookback_time = (datetime.now() - timedelta(hours=self.config['lookback_hours'])).strftime("%d %b, %Y")
            klines = api.get_multi_timeframe_data(self.config['symbol'], self.config['intervals'], lookback_time)
            
            if not klines:
                logging.error("❌ 无法获取K线数据")
                return None
            
            # 处理不同时间周期的数据
            processed_data = {}
            for interval, kline_data in klines.items():
                df = self.process_klines_to_dataframe(kline_data, interval)
                if df is not None and not df.empty:
                    processed_data[interval] = df
                    logging.info(f"✅ {interval}数据处理完成: {len(df)}条记录")
            
            if not processed_data:
                logging.error("❌ 所有时间周期数据处理失败")
                return None
            
            # 获取市场深度和大单数据
            big_trades = []
            try:
                big_trades = api.get_recent_trades(self.config['symbol'])
            except Exception as e:
                logging.warning(f"⚠️ 获取交易数据失败: {e}")
                big_trades = []
            
            market_depth = api.get_order_book(self.config['symbol'])
            
            # 获取情绪分析数据
            sentiment_data = self.sentiment_analyzer.analyze_order_book_sentiment()
            
            # 合并特征
            combined_df = self.combine_timeframe_features(processed_data, big_trades, market_depth, sentiment_data)
            
            if combined_df is None or combined_df.empty:
                logging.error("❌ 特征合并失败")
                return None
            
            if retrain:
                # 创建增强标签
                labels = self.label_processor.create_enhanced_labels(combined_df, self.config['prediction_minutes'])
                return self.prepare_training_data(combined_df, labels)
            else:
                return self.prepare_prediction_data(combined_df)
                
        except Exception as e:
            logging.error(f"❌ 数据准备失败: {e}")
            return None
    
    def process_klines_to_dataframe(self, klines, interval):
        """处理K线数据为DataFrame"""
        try:
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 转换数据类型
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_asset_volume', 'number_of_trades',
                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            # 添加时间周期后缀
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[f'{col}_{interval}'] = df[col]
            
            # 创建技术指标
            df = self.add_enhanced_technical_indicators(df, interval)
            
            # 丢弃NaN值
            df.dropna(inplace=True)
            
            return df
        except Exception as e:
            logging.error(f"❌ 处理{interval}数据失败: {e}")
            return None
    
    def combine_timeframe_features(self, processed_data, big_trades, market_depth, sentiment_data):
        """合并多时间周期特征"""
        try:
            # 以1分钟数据为基准
            base_df = processed_data.get('1m')
            if base_df is None:
                logging.error("❌ 缺少1分钟基准数据")
                return None
            
            combined_df = base_df.copy()
            
            # 添加其他时间周期的关键指标
            for interval in ['5m', '15m']:
                if interval in processed_data:
                    interval_df = processed_data[interval]
                    # 通过时间对齐合并数据
                    combined_df = self.merge_by_time(combined_df, interval_df, interval)
            
            # 添加市场微观结构特征
            if big_trades:
                combined_df = self.add_big_trade_features(combined_df, big_trades)
            
            if market_depth:
                combined_df = self.add_market_depth_features(combined_df, market_depth)
            
            # 添加情绪分析特征
            if sentiment_data:
                combined_df = self.add_sentiment_features(combined_df, sentiment_data)
            
            return combined_df
        except Exception as e:
            logging.error(f"❌ 合并多时间周期特征失败: {e}")
            return None
    
    def merge_by_time(self, base_df, interval_df, interval):
        """通过时间对齐合并数据 - 🔧 优化性能，避免DataFrame碎片化"""
        try:
            # 🔧 获取所有需要合并的特征，而不只是选择几个关键指标
            exclude_cols = ['open_time', 'close_time']  # 排除时间列
            
            # 重新采样到1分钟
            interval_df_resampled = interval_df.set_index('open_time').resample('1min').ffill()
            
            # 🔧 性能优化：批量处理所有特征，避免逐个添加列
            new_columns = {}
            features_to_merge = [col for col in interval_df.columns if col not in exclude_cols and col != 'open_time']
            
            for col in features_to_merge:
                # 为所有特征添加时间周期后缀
                new_col_name = f"{col}_{interval}" if not col.endswith(f"_{interval}") else col
                
                # 将特征映射到基准数据的时间索引
                if col in interval_df_resampled.columns:
                    new_columns[new_col_name] = base_df['open_time'].map(
                        interval_df_resampled[col].to_dict()
                    )
            
            # 🔧 一次性合并所有新列，避免碎片化
            if new_columns:
                new_df = pd.DataFrame(new_columns, index=base_df.index)
                base_df = pd.concat([base_df, new_df], axis=1)
            
            logging.info(f"📊 成功合并{interval}时间周期的{len(features_to_merge)}个特征")
            return base_df
        except Exception as e:
            logging.error(f"❌ 时间对齐合并失败: {e}")
            return base_df
    
    def add_big_trade_features(self, df, big_trades):
        """添加大单特征"""
        try:
            # 计算最近1分钟内的大单特征
            current_time = time.time() * 1000  # 转换为毫秒
            recent_big_trades = [t for t in big_trades 
                               if current_time - t['time'] <= 60000]  # 1分钟内
            
            df['big_trade_count'] = len(recent_big_trades)
            df['big_trade_volume'] = sum(t['qty'] for t in recent_big_trades)
            df['big_buy_ratio'] = (sum(t['qty'] for t in recent_big_trades if not t['is_buyer_maker']) / 
                                 max(1, sum(t['qty'] for t in recent_big_trades)))
            
            # 使用与get_recent_trades相同的格式显示大单信息
            threshold = self.config.get('big_trade_threshold', 0.01)
            if recent_big_trades:
                logging.info(f"💰 检测到 {len(recent_big_trades)} 笔最近1分钟内的大单交易 (>{threshold} BTC)")
            else:
                logging.info(f"💰 最近1分钟内无大单交易 (>{threshold} BTC)")
                
            return df
        except Exception as e:
            logging.error(f"❌ 添加大单特征失败: {e}")
            return df
    
    def add_market_depth_features(self, df, market_depth):
        """添加市场深度特征 - 🔧 优化性能"""
        try:
            if not market_depth or 'bids' not in market_depth or 'asks' not in market_depth:
                # 如果没有市场深度数据，批量填充默认值
                default_depth_features = {
                    'bid_ask_spread': 0,
                    'depth_ratio': 1.0,
                    'price_impact': 0
                }
                for key, value in default_depth_features.items():
                    df[key] = value
                return df
            
            bids = market_depth['bids']
            asks = market_depth['asks']
            
            if not bids or not asks:
                default_depth_features = {
                    'bid_ask_spread': 0,
                    'depth_ratio': 1.0,
                    'price_impact': 0
                }
                for key, value in default_depth_features.items():
                    df[key] = value
                return df
            
            # 买卖价差
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid * 100
            
            # 深度比率
            bid_volume = sum([float(bid[1]) for bid in bids[:5]])
            ask_volume = sum([float(ask[1]) for ask in asks[:5]])
            depth_ratio = bid_volume / ask_volume if ask_volume > 0 else 1
            
            # 价格冲击（5万USDT成交的价格影响）
            target_volume = 50000
            cumulative_volume = 0
            price_impact = 0
            
            for ask in asks:
                price = float(ask[0])
                volume = float(ask[1])
                if cumulative_volume + volume * price >= target_volume:
                    price_impact = (price - best_ask) / best_ask * 100
                    break
                cumulative_volume += volume * price
            
            # 🔧 批量添加所有市场深度特征
            depth_features = {
                'bid_ask_spread': spread,
                'depth_ratio': depth_ratio,
                'price_impact': price_impact
            }
            for key, value in depth_features.items():
                df[key] = value
            
            return df
        except Exception as e:
            logging.warning(f"⚠️ 添加市场深度特征失败: {e}")
            # 批量添加默认值
            default_depth_features = {
                'bid_ask_spread': 0,
                'depth_ratio': 1.0,
                'price_impact': 0
            }
            for key, value in default_depth_features.items():
                df[key] = value
            return df
    
    def add_sentiment_features(self, df, sentiment_data):
        """添加情绪特征 - 🔧 支持增强的情绪指标，优化性能"""
        try:
            # 🔧 性能优化：批量添加所有情绪特征
            sentiment_features = {
                # 基础情绪特征
                'sentiment_score': sentiment_data.get('sentiment_score', 0.5),
                'book_imbalance': sentiment_data.get('book_imbalance', 0),
                'bid_ask_ratio': sentiment_data.get('bid_volume', 0) / max(sentiment_data.get('ask_volume', 1), 1),
                
                # 🔧 新增：买墙/卖墙特征
                'buy_wall_volume': sentiment_data.get('buy_wall_volume', 0),
                'sell_wall_volume': sentiment_data.get('sell_wall_volume', 0),
                'wall_ratio': sentiment_data.get('wall_ratio', 1.0),
                'deep_imbalance': sentiment_data.get('deep_imbalance', 0),
                
                # 🔧 新增：情绪强度指标
                'sentiment_strength': abs(sentiment_data.get('sentiment_score', 0.5) - 0.5) * 2,  # 0-1
                'wall_dominance': 1 if sentiment_data.get('wall_ratio', 1.0) > 1.2 else (-1 if sentiment_data.get('wall_ratio', 1.0) < 0.8 else 0)
            }
            
            # 🔧 一次性添加所有情绪特征
            for key, value in sentiment_features.items():
                df[key] = value
            
            return df
        except Exception as e:
            logging.warning(f"⚠️ 添加情绪特征失败: {e}")
            # 添加默认值
            default_features = {
                'sentiment_score': 0.5,
                'book_imbalance': 0,
                'bid_ask_ratio': 1.0,
                'buy_wall_volume': 0,
                'sell_wall_volume': 0,
                'wall_ratio': 1.0,
                'deep_imbalance': 0,
                'sentiment_strength': 0,
                'wall_dominance': 0
            }
            for key, value in default_features.items():
                df[key] = value
            return df
    
    def prepare_training_data(self, df, labels):
        """准备训练数据 - 支持多标签"""
        try:
            # 🔧 优化特征选择 - 确保所有特征都包含
            feature_cols = []
            
            # 基础价格特征
            base_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in base_cols:
                if col in df.columns:
                    feature_cols.append(col)
            
            # 🔧 收集所有时间周期的技术指标
            intervals = ['1m', '5m', '15m']
            tech_indicators = [
                'rsi_7', 'rsi_14',  # 🔧 包含rsi_7（新增）和rsi_14
                'ema_fast', 'ema_slow', 'macd_simple', 'macd_signal',  # 🔧 使用新的EMA和MACD
                'macd', 'macd_histogram',
                'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',  # 🔧 包含sma_5
                'adx', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_pct',
                'atr', 'obv', 'volume_sma', 'volume_ratio',
                'price_vs_sma10', 'price_vs_sma20', 'ema_cross',  # 🔧 包含price_vs_sma10
                'momentum_3', 'momentum_5', 'momentum_10',  # 🔧 包含momentum_3
                'volatility_ratio', 'vwap', 'price_vs_vwap',
                'high_low_pct', 'rsi_smooth', 'volume_momentum',
                'price_oscillation', 'trend_fast', 'trend_strength'  # 🔧 包含新指标
            ]
            
            # 🔧 动态添加所有可用的技术指标
            missing_tech_features = {}  # 🔧 批量收集缺失的技术指标
            
            for interval in intervals:
                for indicator in tech_indicators:
                    col_name = f"{indicator}_{interval}"
                    if col_name in df.columns:
                        feature_cols.append(col_name)
                    else:
                        # 🔧 如果缺失技术指标，添加默认值而不是跳过
                        missing_features = getattr(self, '_missing_features', set())
                        if col_name not in missing_features:
                            logging.warning(f"⚠️ 缺少特征: {col_name}")
                            missing_features.add(col_name)
                            self._missing_features = missing_features
                        
                        # 🔧 为缺失的技术指标收集合理的默认值
                        if 'rsi' in indicator:
                            missing_tech_features[col_name] = 50  # RSI中位值
                        elif 'macd' in indicator:
                            missing_tech_features[col_name] = 0   # MACD中性值
                        elif 'ema' in indicator or 'sma' in indicator or 'bb_' in indicator or 'vwap' in indicator:
                            missing_tech_features[col_name] = df['close']  # 使用当前价格作为默认值
                        elif 'stoch' in indicator:
                            missing_tech_features[col_name] = 50  # 随机振荡器中位值
                        elif 'adx' in indicator:
                            missing_tech_features[col_name] = 25  # ADX中等强度
                        elif 'cci' in indicator:
                            missing_tech_features[col_name] = 0   # CCI中性值
                        elif 'williams_r' in indicator:
                            missing_tech_features[col_name] = -50 # Williams %R中位值
                        elif 'atr' in indicator:
                            missing_tech_features[col_name] = (df['high'] - df['low']).mean()  # 使用平均波幅
                        elif 'volume' in indicator:
                            missing_tech_features[col_name] = df['volume'].mean()  # 使用平均成交量
                        elif 'momentum' in indicator:
                            missing_tech_features[col_name] = 0   # 动量中性值
                        elif 'volatility' in indicator:
                            missing_tech_features[col_name] = 1   # 波动率比率中性值
                        elif 'trend' in indicator:
                            missing_tech_features[col_name] = 0.5 # 趋势中性值
                        elif indicator in ['ema_cross', 'trend_fast']:
                            missing_tech_features[col_name] = 0   # 二元指标默认值
                        else:
                            missing_tech_features[col_name] = 0   # 其他指标默认值
                        
                        feature_cols.append(col_name)
            
            # 🔧 批量添加所有缺失的技术指标特征
            if missing_tech_features:
                for col_name, default_value in missing_tech_features.items():
                    df[col_name] = default_value
            
            # 🔧 情绪特征 - 包含所有新增的情绪指标
            sentiment_cols = [
                'sentiment_score', 'book_imbalance', 'bid_ask_ratio',
                'buy_wall_volume', 'sell_wall_volume', 'wall_ratio', 'deep_imbalance',
                'sentiment_strength', 'wall_dominance'
            ]
            for col in sentiment_cols:
                if col in df.columns:
                    feature_cols.append(col)
                else:
                    # 🔧 如果缺失情绪特征，添加默认值
                    if col == 'sentiment_score':
                        df[col] = 0.5
                    elif col in ['buy_wall_volume', 'sell_wall_volume', 'book_imbalance', 'deep_imbalance']:
                        df[col] = 0
                    elif col == 'bid_ask_ratio' or col == 'wall_ratio':
                        df[col] = 1.0
                    elif col == 'sentiment_strength':
                        df[col] = 0
                    elif col == 'wall_dominance':
                        df[col] = 0
                    feature_cols.append(col)
            
            # 🔧 市场深度特征
            depth_cols = ['bid_ask_spread', 'depth_ratio', 'price_impact']
            for col in depth_cols:
                if col in df.columns:
                    feature_cols.append(col)
                else:
                    # 🔧 如果缺失深度特征，添加默认值
                    if col == 'bid_ask_spread' or col == 'price_impact':
                        df[col] = 0
                    elif col == 'depth_ratio':
                        df[col] = 1.0
                    feature_cols.append(col)
            
            # 🔧 大单特征
            trade_cols = ['big_trade_count', 'big_trade_volume', 'big_buy_ratio']
            for col in trade_cols:
                if col in df.columns:
                    feature_cols.append(col)
                else:
                    # 🔧 如果缺失大单特征，添加默认值
                    if col == 'big_trade_count' or col == 'big_trade_volume':
                        df[col] = 0
                    elif col == 'big_buy_ratio':
                        df[col] = 0.5
                    feature_cols.append(col)
            
            if not feature_cols:
                logging.error("❌ 没有可用的特征列")
                return None
            
            # 特征数据
            feature_data = df[feature_cols].dropna()
            
            if len(feature_data) < 50:
                logging.error("❌ 特征数据不足")
                return None
            
            # 保存特征列名
            self.features = feature_cols
            
            # 特征缩放
            if '1m' not in self.feature_scalers:
                self.feature_scalers['1m'] = MinMaxScaler()
                scaled_features = self.feature_scalers['1m'].fit_transform(feature_data)
            else:
                scaled_features = self.feature_scalers['1m'].transform(feature_data)
            
            # 创建序列数据 - 只使用基础方向标签进行兼容
            X, y = self.create_sequences(scaled_features, labels['direction'].loc[feature_data.index].values)
            
            return X, y
            
        except Exception as e:
            logging.error(f"❌ 准备训练数据时出错: {e}")
            return None
    
    def prepare_prediction_data(self, df):
        """准备预测数据 - 🔧 确保特征一致性"""
        try:
            # 🔧 使用与训练相同的特征选择逻辑
            feature_cols = []
            
            # 基础价格特征
            base_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in base_cols:
                if col in df.columns:
                    feature_cols.append(col)
            
            # 🔧 收集所有时间周期的技术指标（与训练时相同）
            intervals = ['1m', '5m', '15m']
            tech_indicators = [
                'rsi_7', 'rsi_14',  # 🔧 包含rsi_7（新增）和rsi_14
                'ema_fast', 'ema_slow', 'macd_simple', 'macd_signal',  # 🔧 使用新的EMA和MACD
                'macd', 'macd_histogram',
                'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',  # 🔧 包含sma_5
                'adx', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_pct',
                'atr', 'obv', 'volume_sma', 'volume_ratio',
                'price_vs_sma10', 'price_vs_sma20', 'ema_cross',  # 🔧 包含price_vs_sma10
                'momentum_3', 'momentum_5', 'momentum_10',  # 🔧 包含momentum_3
                'volatility_ratio', 'vwap', 'price_vs_vwap',
                'high_low_pct', 'rsi_smooth', 'volume_momentum',
                'price_oscillation', 'trend_fast', 'trend_strength'  # 🔧 包含新指标
            ]
            
            # 🔧 动态添加所有可用的技术指标
            missing_tech_features = {}  # 🔧 批量收集缺失的技术指标
            
            for interval in intervals:
                for indicator in tech_indicators:
                    col_name = f"{indicator}_{interval}"
                    if col_name in df.columns:
                        feature_cols.append(col_name)
                    else:
                        # 🔧 如果缺失技术指标，添加默认值而不是跳过
                        missing_features = getattr(self, '_missing_features', set())
                        if col_name not in missing_features:
                            logging.warning(f"⚠️ 缺少特征: {col_name}")
                            missing_features.add(col_name)
                            self._missing_features = missing_features
                        
                        # 🔧 为缺失的技术指标收集合理的默认值
                        if 'rsi' in indicator:
                            missing_tech_features[col_name] = 50  # RSI中位值
                        elif 'macd' in indicator:
                            missing_tech_features[col_name] = 0   # MACD中性值
                        elif 'ema' in indicator or 'sma' in indicator or 'bb_' in indicator or 'vwap' in indicator:
                            missing_tech_features[col_name] = df['close']  # 使用当前价格作为默认值
                        elif 'stoch' in indicator:
                            missing_tech_features[col_name] = 50  # 随机振荡器中位值
                        elif 'adx' in indicator:
                            missing_tech_features[col_name] = 25  # ADX中等强度
                        elif 'cci' in indicator:
                            missing_tech_features[col_name] = 0   # CCI中性值
                        elif 'williams_r' in indicator:
                            missing_tech_features[col_name] = -50 # Williams %R中位值
                        elif 'atr' in indicator:
                            missing_tech_features[col_name] = (df['high'] - df['low']).mean()  # 使用平均波幅
                        elif 'volume' in indicator:
                            missing_tech_features[col_name] = df['volume'].mean()  # 使用平均成交量
                        elif 'momentum' in indicator:
                            missing_tech_features[col_name] = 0   # 动量中性值
                        elif 'volatility' in indicator:
                            missing_tech_features[col_name] = 1   # 波动率比率中性值
                        elif 'trend' in indicator:
                            missing_tech_features[col_name] = 0.5 # 趋势中性值
                        elif indicator in ['ema_cross', 'trend_fast']:
                            missing_tech_features[col_name] = 0   # 二元指标默认值
                        else:
                            missing_tech_features[col_name] = 0   # 其他指标默认值
                        
                        feature_cols.append(col_name)
            
            # 🔧 批量添加所有缺失的技术指标特征
            if missing_tech_features:
                for col_name, default_value in missing_tech_features.items():
                    df[col_name] = default_value
            
            # 🔧 情绪特征 - 包含所有新增的情绪指标
            sentiment_cols = [
                'sentiment_score', 'book_imbalance', 'bid_ask_ratio',
                'buy_wall_volume', 'sell_wall_volume', 'wall_ratio', 'deep_imbalance',
                'sentiment_strength', 'wall_dominance'
            ]
            for col in sentiment_cols:
                if col in df.columns:
                    feature_cols.append(col)
                else:
                    # 🔧 如果缺失情绪特征，添加默认值
                    if col == 'sentiment_score':
                        df[col] = 0.5
                    elif col in ['buy_wall_volume', 'sell_wall_volume', 'book_imbalance', 'deep_imbalance']:
                        df[col] = 0
                    elif col == 'bid_ask_ratio' or col == 'wall_ratio':
                        df[col] = 1.0
                    elif col == 'sentiment_strength':
                        df[col] = 0
                    elif col == 'wall_dominance':
                        df[col] = 0
                    feature_cols.append(col)
            
            # 🔧 市场深度特征
            depth_cols = ['bid_ask_spread', 'depth_ratio', 'price_impact']
            for col in depth_cols:
                if col in df.columns:
                    feature_cols.append(col)
                else:
                    # 🔧 如果缺失深度特征，添加默认值
                    if col == 'bid_ask_spread' or col == 'price_impact':
                        df[col] = 0
                    elif col == 'depth_ratio':
                        df[col] = 1.0
                    feature_cols.append(col)
            
            # 🔧 大单特征
            trade_cols = ['big_trade_count', 'big_trade_volume', 'big_buy_ratio']
            for col in trade_cols:
                if col in df.columns:
                    feature_cols.append(col)
                else:
                    # 🔧 如果缺失大单特征，添加默认值
                    if col == 'big_trade_count' or col == 'big_trade_volume':
                        df[col] = 0
                    elif col == 'big_buy_ratio':
                        df[col] = 0.5
                    feature_cols.append(col)
            
            if not feature_cols:
                logging.error("❌ 没有可用的特征列")
                return None, None
            
            # 🔧 错误日志：显示最终使用的特征
            logging.info(f"✅ 预测特征数量: {len(feature_cols)}")
            
            # 检查是否有缺失的关键特征
            missing_key_features = []
            key_features = ['rsi_14_1m', 'close', 'volume']
            for key_feature in key_features:
                if key_feature not in feature_cols:
                    missing_key_features.append(key_feature)
            
            if missing_key_features:
                logging.error(f"❌ 缺少关键特征: {missing_key_features}")
                return None, None
            
            # 选择特征数据
            feature_data = df[feature_cols].copy()
            
            # 🔧 优化NaN处理
            # 首先检查NaN情况
            nan_counts = feature_data.isnull().sum()
            total_nans = nan_counts.sum()
            if total_nans > 0:
                logging.info(f"📊 检测到 {total_nans} 个NaN值，进行处理...")
                
                # 🔧 修复FutureWarning：使用新的方法替代过时的method参数
                feature_data = feature_data.ffill().bfill().fillna(0)
                
                # 再次检查
                remaining_nans = feature_data.isnull().sum().sum()
                if remaining_nans > 0:
                    logging.warning(f"⚠️ 仍有 {remaining_nans} 个NaN值，使用均值填充")
                    feature_data = feature_data.fillna(feature_data.mean()).fillna(0)
            
            # 特征缩放
            if hasattr(self, 'feature_scalers') and self.feature_scalers:
                try:
                    # 🔧 修复：从字典中获取正确的缩放器
                    if '1m' in self.feature_scalers:
                        feature_data_scaled = self.feature_scalers['1m'].transform(feature_data)
                    elif isinstance(self.feature_scalers, dict) and len(self.feature_scalers) > 0:
                        # 如果没有'1m'键，使用第一个可用的缩放器
                        scaler_key = list(self.feature_scalers.keys())[0]
                        feature_data_scaled = self.feature_scalers[scaler_key].transform(feature_data)
                    else:
                        # 兼容旧版本：如果feature_scalers本身就是缩放器对象
                        feature_data_scaled = self.feature_scalers.transform(feature_data)
                except Exception as e:
                    logging.error(f"❌ 特征缩放失败: {e}")
                    return None, None
            else:
                logging.warning("⚠️ 没有找到特征缩放器，使用原始数据")
                feature_data_scaled = feature_data.values
            
            # 创建时间序列
            sequence_length = self.config.get('sequence_length', 10)
            if len(feature_data_scaled) < sequence_length:
                logging.error(f"❌ 数据长度不足：需要{sequence_length}，实际{len(feature_data_scaled)}")
                return None, None
            
            # 取最后sequence_length个时间步
            X = feature_data_scaled[-sequence_length:].reshape(1, sequence_length, -1)
            
            return X, df
            
        except Exception as e:
            logging.error(f"❌ 准备预测数据失败: {e}")
            return None, None
    
    def add_enhanced_technical_indicators(self, df, interval):
        """添加增强的技术指标 - 🔧 优化以减少NaN值"""
        try:
            suffix = f"_{interval}"
            
            # 🔵 1. 趋势指标 - 🔧 优化计算窗口
            # RSI (使用较短窗口减少NaN)
            df[f'rsi_7{suffix}'] = ta.momentum.rsi(df['close'], window=7)  # 🔧 从14改为7
            df[f'rsi_14{suffix}'] = ta.momentum.rsi(df['close'], window=14)
            
            # 🔧 EMA替代MACD减少NaN - 使用更短窗口
            # 快速EMA和慢速EMA
            df[f'ema_fast{suffix}'] = ta.trend.ema_indicator(df['close'], window=8)  # 🔧 从12改为8
            df[f'ema_slow{suffix}'] = ta.trend.ema_indicator(df['close'], window=21)  # 🔧 从26改为21
            
            # 简化的MACD信号
            df[f'macd_simple{suffix}'] = df[f'ema_fast{suffix}'] - df[f'ema_slow{suffix}']
            df[f'macd_signal{suffix}'] = ta.trend.ema_indicator(df[f'macd_simple{suffix}'], window=6)  # 🔧 从9改为6
            
            # 保留原始MACD用于兼容性（但使用更短窗口）
            try:
                macd = ta.trend.MACD(df['close'], window_slow=21, window_fast=8, window_sign=6)  # 🔧 优化窗口
                df[f'macd{suffix}'] = macd.macd()
                df[f'macd_histogram{suffix}'] = macd.macd_diff()
            except:
                # 如果MACD计算失败，使用简化版本
                df[f'macd{suffix}'] = df[f'macd_simple{suffix}']
                df[f'macd_histogram{suffix}'] = df[f'macd_simple{suffix}'] - df[f'macd_signal{suffix}']
            
            # 移动平均线 - 🔧 使用更短窗口
            df[f'sma_5{suffix}'] = ta.trend.sma_indicator(df['close'], window=5)   # 🔧 新增短期SMA
            df[f'sma_10{suffix}'] = ta.trend.sma_indicator(df['close'], window=10)
            df[f'sma_20{suffix}'] = ta.trend.sma_indicator(df['close'], window=20)
            df[f'ema_12{suffix}'] = ta.trend.ema_indicator(df['close'], window=12)
            df[f'ema_26{suffix}'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # ADX (平均方向指数) - 🔧 使用更短窗口
            try:
                df[f'adx{suffix}'] = ta.trend.adx(df['high'], df['low'], df['close'], window=10)  # 🔧 从14改为10
            except:
                df[f'adx{suffix}'] = 50  # 默认中性值
            
            # 🔵 2. 动量指标 - 🔧 优化窗口大小
            # 随机振荡器 - 使用更短窗口
            try:
                stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=10, smooth_window=3)  # 🔧 优化窗口
                df[f'stoch_k{suffix}'] = stoch.stoch()
                df[f'stoch_d{suffix}'] = stoch.stoch_signal()
            except:
                df[f'stoch_k{suffix}'] = 50
                df[f'stoch_d{suffix}'] = 50
            
            # CCI (商品频道指标) - 🔧 使用更短窗口
            try:
                df[f'cci{suffix}'] = ta.trend.cci(df['high'], df['low'], df['close'], window=14)  # 🔧 从20改为14
            except:
                df[f'cci{suffix}'] = 0
            
            # Williams %R - 🔧 使用更短窗口
            try:
                df[f'williams_r{suffix}'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=10)  # 🔧 从14改为10
            except:
                df[f'williams_r{suffix}'] = -50
            
            # 🔵 3. 波动率指标 - 🔧 优化窗口
            # 布林带 - 使用更短窗口
            try:
                bollinger = ta.volatility.BollingerBands(df['close'], window=15, window_dev=2)  # 🔧 从20改为15
                df[f'bb_upper{suffix}'] = bollinger.bollinger_hband()
                df[f'bb_middle{suffix}'] = bollinger.bollinger_mavg()
                df[f'bb_lower{suffix}'] = bollinger.bollinger_lband()
                df[f'bb_pct{suffix}'] = bollinger.bollinger_pband()
            except:
                # 如果布林带计算失败，使用简化版本
                sma_15 = df['close'].rolling(15).mean()
                std_15 = df['close'].rolling(15).std()
                df[f'bb_upper{suffix}'] = sma_15 + 2 * std_15
                df[f'bb_middle{suffix}'] = sma_15
                df[f'bb_lower{suffix}'] = sma_15 - 2 * std_15
                df[f'bb_pct{suffix}'] = (df['close'] - df[f'bb_lower{suffix}']) / (df[f'bb_upper{suffix}'] - df[f'bb_lower{suffix}'])
            
            # ATR (平均真实波幅) - 🔧 使用更短窗口
            try:
                df[f'atr{suffix}'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=10)  # 🔧 从14改为10
            except:
                df[f'atr{suffix}'] = (df['high'] - df['low']).rolling(10).mean()
            
            # 🔵 4. 成交量指标
            # OBV (能量潮)
            try:
                df[f'obv{suffix}'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            except:
                df[f'obv{suffix}'] = df['volume'].cumsum()
            
            # 成交量均线和比率 - 🔧 使用更短窗口
            df[f'volume_sma{suffix}'] = ta.trend.sma_indicator(df['volume'], window=10)  # 🔧 从20改为10
            df[f'volume_ratio{suffix}'] = df['volume'] / df[f'volume_sma{suffix}'].replace(0, 1)
            
            # 🔵 5. 自定义指标 (避免过拟合) - 🔧 优化计算
            # 价格相对位置
            df[f'price_vs_sma10{suffix}'] = (df['close'] - df[f'sma_10{suffix}']) / df[f'sma_10{suffix}'].replace(0, 1) * 100  # 🔧 使用SMA10
            df[f'price_vs_sma20{suffix}'] = (df['close'] - df[f'sma_20{suffix}']) / df[f'sma_20{suffix}'].replace(0, 1) * 100
            
            # EMA交叉信号 - 🔧 使用优化的EMA
            df[f'ema_cross{suffix}'] = (df[f'ema_fast{suffix}'] > df[f'ema_slow{suffix}']).astype(int)
            
            # 价格动量 (多周期) - 🔧 使用更短周期
            df[f'momentum_3{suffix}'] = (df['close'] / df['close'].shift(3) - 1) * 100  # 🔧 新增3周期动量
            df[f'momentum_5{suffix}'] = (df['close'] / df['close'].shift(5) - 1) * 100
            df[f'momentum_10{suffix}'] = (df['close'] / df['close'].shift(10) - 1) * 100
            
            # 波动率变化率 - 🔧 使用更短窗口
            rolling_std_short = df['close'].rolling(3).std()  # 🔧 从5改为3
            rolling_std_long = df['close'].rolling(10).std()   # 🔧 从20改为10
            df[f'volatility_ratio{suffix}'] = rolling_std_short / rolling_std_long.replace(0, 1)
            
            # 成交量加权价格 - 🔧 使用更短窗口
            df[f'vwap{suffix}'] = (df['close'] * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum()  # 🔧 从20改为10
            df[f'price_vs_vwap{suffix}'] = (df['close'] - df[f'vwap{suffix}']) / df[f'vwap{suffix}'].replace(0, 1) * 100
            
            # 高低点距离
            df[f'high_low_pct{suffix}'] = (df['high'] - df['low']) / df['close'] * 100
            
            # 🔵 6. 噪声指标 (增加多样性) - 🔧 优化计算
            # RSI平滑
            df[f'rsi_smooth{suffix}'] = df[f'rsi_7{suffix}'].rolling(2).mean()  # 🔧 使用RSI7和更短平滑窗口
            df[f'volume_momentum{suffix}'] = (df['volume'] / df['volume'].shift(1) - 1) * 100
            
            # 价格振荡强度 - 🔧 使用更短窗口
            df[f'price_oscillation{suffix}'] = df['close'].rolling(5).apply(  # 🔧 从10改为5
                lambda x: (x.max() - x.min()) / x.mean() * 100 if x.mean() != 0 else 0
            )
            
            # 🔧 新增：快速趋势指标（减少NaN）
            df[f'trend_fast{suffix}'] = (df['close'] > df[f'sma_5{suffix}']).astype(int)
            df[f'trend_strength{suffix}'] = (df[f'sma_5{suffix}'] - df[f'sma_10{suffix}']) / df[f'sma_10{suffix}'] * 100
            
            return df
            
        except Exception as e:
            logging.warning(f"⚠️ 计算{interval}技术指标时出错: {e}")
            return df
    
    def create_sequences(self, data, targets):
        X, y = [], []
        seq_length = 10  # 使用10个时间点的数据预测
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            if targets is not None:
                y.append(targets[i+seq_length])
        
        X_tensor = torch.FloatTensor(np.array(X))
        
        if targets is not None:
            y_tensor = torch.FloatTensor(np.array(y).reshape(-1, 1))
            return X_tensor, y_tensor
        else:
            # 预测模式，只返回X
            return X_tensor

# 模型训练和预测类
class BitcoinPredictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.api = EnhancedBinanceAPI(config['api_key'], config['api_secret'], config)
        self.data_processor = EnhancedDataProcessor(config)
        self.simulation_records = []
        # 添加马丁格尔策略变量
        self.current_bet_level = 0  # 当前投注级别：0=5U, 1=10U, 2=30U, 3=90U, 4=250U
        self.martingale_bet_amounts = [5, 10, 30, 90, 250]  # 马丁格尔投注金额序列
        self.load_model()
    
    def load_model(self):
        if os.path.exists('model.pth'):
            try:
                # 添加MinMaxScaler到安全加载列表
                from torch.serialization import safe_globals
                with safe_globals([MinMaxScaler]):
                    model_state = torch.load('model.pth', weights_only=False, map_location=self.device)
                
                # 从模型参数实际形状中推断真实的hidden_size
                model_state_dict = model_state['model_state_dict']
                
                # 检查attention.in_proj_weight的形状，这是最可靠的判断方式
                if 'attention.in_proj_weight' in model_state_dict:
                    attention_shape = model_state_dict['attention.in_proj_weight'].shape
                    # attention shape通常是 [hidden_size*3, hidden_size*2]
                    real_hidden_size = attention_shape[1] // 2
                    logging.info(f"📐 从attention权重形状{attention_shape}推断hidden_size={real_hidden_size}")
                # 检查lstm.weight_hh_l0形状
                elif 'lstm.weight_hh_l0' in model_state_dict:
                    lstm_shape = model_state_dict['lstm.weight_hh_l0'].shape
                    # lstm.weight_hh_l0形状通常是 [hidden_size*4, hidden_size]
                    real_hidden_size = lstm_shape[1]
                    logging.info(f"📐 从LSTM权重形状{lstm_shape}推断hidden_size={real_hidden_size}")
                # 如果无法从权重推断，使用状态字典中记录的值
                else:
                    real_hidden_size = model_state['hidden_size']
                    logging.warning(f"⚠️ 无法从权重形状推断，使用模型记录的hidden_size={real_hidden_size}")
                
                # 使用正确推断的hidden_size
                input_size = model_state['input_size']
                num_layers = model_state['num_layers']
                dropout = model_state['dropout']
                
                logging.info(f"📝 使用推断的参数: hidden_size={real_hidden_size}")
                
                # 创建与权重形状完全匹配的模型
                self.model = LSTMModel(input_size, real_hidden_size, num_layers, 1, dropout)
                self.model.load_state_dict(model_state['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # 加载数据处理器状态 - 兼容新旧版本
                self.data_processor.features = model_state['features']
                if 'feature_scalers' in model_state:
                    self.data_processor.feature_scalers = model_state['feature_scalers']
                elif 'feature_scaler' in model_state:
                    # 兼容旧版本：将单个scaler转换为字典格式
                    self.data_processor.feature_scalers = {'1m': model_state['feature_scaler']}
                else:
                    logging.warning("⚠️ 模型文件缺少scaler信息，需要重新训练")
                    return False
                
                # 🧠 加载集成模型
                if 'ensemble_models' in model_state:
                    self.ensemble_models = []
                    for ensemble_state in model_state['ensemble_models']:
                        # 从集成模型状态字典中获取权重形状
                        ensemble_dict = ensemble_state['model_state_dict']
                        
                        # 同样，从权重形状推断真实hidden_size
                        if 'attention.in_proj_weight' in ensemble_dict:
                            attn_shape = ensemble_dict['attention.in_proj_weight'].shape
                            ensemble_hidden_size = attn_shape[1] // 2
                            logging.info(f"📐 集成模型从attention权重形状{attn_shape}推断hidden_size={ensemble_hidden_size}")
                        elif 'lstm.weight_hh_l0' in ensemble_dict:
                            lstm_shape = ensemble_dict['lstm.weight_hh_l0'].shape
                            ensemble_hidden_size = lstm_shape[1]
                            logging.info(f"📐 集成模型从LSTM权重形状{lstm_shape}推断hidden_size={ensemble_hidden_size}")
                        else:
                            # 如果无法推断，使用模型记录的值
                            ensemble_hidden_size = ensemble_state.get('hidden_size', real_hidden_size)
                            logging.warning(f"⚠️ 无法推断集成模型hidden_size，使用记录值={ensemble_hidden_size}")
                        
                        # 重建集成模型
                        ensemble_model = EnhancedLSTMModel(
                            input_size, 
                            ensemble_hidden_size, 
                            num_layers, 
                            1, 
                            dropout
                        )
                        ensemble_model.load_state_dict(ensemble_dict)
                        ensemble_model.to(self.device)
                        ensemble_model.eval()
                        
                        self.ensemble_models.append({
                            'model': ensemble_model,
                            'accuracy': ensemble_state['accuracy'],
                            'weight': ensemble_state['weight']
                        })
                    
                    logging.info(f"✅ 成功加载 {len(self.ensemble_models)} 个集成模型")
                else:
                    logging.info("📝 旧版本模型，无集成模型信息")
                
                logging.info(f"✅ 模型加载成功 (hidden_size={real_hidden_size})")
                return True
            except Exception as e:
                logging.error(f"❌ 加载模型时出错: {e}")
                
                # 检查是否是参数不匹配错误
                if "size mismatch" in str(e):
                    logging.error("💥 模型参数不匹配，这通常是因为:")
                    logging.error("   1. 模型架构已更新")
                    logging.error("   2. 配置参数发生变化")
                    logging.error("   3. 特征数量发生变化")
                    logging.error("")
                    logging.error("🔧 解决方法:")
                    logging.error("   方法1: 删除旧模型重新训练")
                    logging.error("         del model.pth")
                    logging.error("         python bitcoin_prediction.py --train")
                    logging.error("")
                    logging.error("   方法2: 使用PowerShell删除")
                    logging.error("         Remove-Item model.pth")
                    logging.error("         python bitcoin_prediction.py --train")
                    logging.error("")
                    return False
                
                # 如果加载失败，尝试旧版本兼容方式加载
                try:
                    logging.info("🔄 尝试兼容模式加载模型...")
                    model_state = torch.load('model.pth', weights_only=False, map_location=self.device, pickle_module=pickle)
                    
                    # 从模型参数实际形状中推断真实的hidden_size
                    model_state_dict = model_state['model_state_dict']
                    
                    # 检查attention.in_proj_weight的形状，这是最可靠的判断方式
                    if 'attention.in_proj_weight' in model_state_dict:
                        attention_shape = model_state_dict['attention.in_proj_weight'].shape
                        # attention shape通常是 [hidden_size*3, hidden_size*2]
                        real_hidden_size = attention_shape[1] // 2
                        logging.info(f"📐 兼容模式: 从attention权重形状{attention_shape}推断hidden_size={real_hidden_size}")
                    # 检查lstm.weight_hh_l0形状
                    elif 'lstm.weight_hh_l0' in model_state_dict:
                        lstm_shape = model_state_dict['lstm.weight_hh_l0'].shape
                        # lstm.weight_hh_l0形状通常是 [hidden_size*4, hidden_size]
                        real_hidden_size = lstm_shape[1]
                        logging.info(f"📐 兼容模式: 从LSTM权重形状{lstm_shape}推断hidden_size={real_hidden_size}")
                    # 如果无法从权重推断，使用状态字典中记录的值
                    else:
                        real_hidden_size = model_state['hidden_size']
                        logging.warning(f"⚠️ 兼容模式: 无法从权重形状推断，使用模型记录的hidden_size={real_hidden_size}")
                    
                    # 使用正确推断的hidden_size
                    input_size = model_state['input_size']
                    num_layers = model_state['num_layers']
                    dropout = model_state['dropout']
                    
                    logging.info(f"📝 兼容模式: 使用推断的参数 hidden_size={real_hidden_size}")
                    
                    self.model = LSTMModel(input_size, real_hidden_size, num_layers, 1, dropout)
                    self.model.load_state_dict(model_state['model_state_dict'])
                    self.model.to(self.device)
                    self.model.eval()
                    
                    # 加载数据处理器状态 - 兼容新旧版本
                    self.data_processor.features = model_state['features']
                    if 'feature_scalers' in model_state:
                        self.data_processor.feature_scalers = model_state['feature_scalers']
                    elif 'feature_scaler' in model_state:
                        # 兼容旧版本：将单个scaler转换为字典格式
                        self.data_processor.feature_scalers = {'1m': model_state['feature_scaler']}
                    else:
                        logging.warning("⚠️ 模型文件缺少scaler信息，需要重新训练")
                        return False
                    
                    logging.info("✅ 模型兼容模式加载成功")
                    return True
                except Exception as e2:
                    logging.error(f"❌ 兼容模式加载模型失败: {e2}")
                    if "size mismatch" in str(e2):
                        logging.error("💥 即使兼容模式也无法解决参数不匹配问题")
                        logging.error("🔧 请删除模型文件并重新训练:")
                        logging.error("   del model.pth && python bitcoin_prediction.py --train")
                    return False
        else:
            logging.warning("⚠️ 没有找到模型文件，需要先训练模型")
            return False
    
    def save_model(self, input_size):
        # 准备保存集成模型的状态字典
        ensemble_states = []
        if hasattr(self, 'ensemble_models') and self.ensemble_models:
            for model_info in self.ensemble_models:
                ensemble_states.append({
                    'model_state_dict': model_info['model'].state_dict(),
                    'accuracy': model_info['accuracy'],
                    'weight': model_info['weight'],
                    'hidden_size': model_info['model'].hidden_size  # 保存每个模型的hidden_size
                })
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'input_size': input_size,
            'hidden_size': self.config['hidden_size'],
            'num_layers': self.config['num_layers'],
            'dropout': self.config['dropout'],
            'features': self.data_processor.features,
            'feature_scalers': self.data_processor.feature_scalers,
            'ensemble_models': ensemble_states  # 保存集成模型
        }
        # 使用pickle模块保存，确保兼容性
        torch.save(model_state, 'model.pth', pickle_module=pickle)
        logging.info("模型保存成功")
    
    def train(self):
        logging.info("开始训练增强模型...")
        X, y = self.data_processor.fetch_and_prepare_enhanced_data(self.api, retrain=True)
        
        if X is None or y is None:
            logging.error("数据准备失败，无法训练模型")
            return False
        
        # 📊 数据平衡处理
        try:
            # 将3D数据重塑为2D进行SMOTE处理
            X_reshaped = X.reshape(X.shape[0], -1)
            y_flat = y.flatten()
            
            # 检查类别分布
            unique, counts = np.unique(y_flat, return_counts=True)
            logging.info(f"训练数据类别分布: {dict(zip(unique, counts))}")
            
            # 只有在类别不平衡时才使用SMOTE
            if len(unique) > 1 and min(counts) / max(counts) < 0.7:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min(counts)-1))
                X_balanced, y_balanced = smote.fit_resample(X_reshaped, y_flat)
                
                # 重塑回3D格式
                X = X_balanced.reshape(-1, X.shape[1], X.shape[2])
                y = y_balanced.reshape(-1, 1)
                logging.info(f"✅ SMOTE平衡后数据量: {X.shape[0]}")
                
                # 再次检查平衡后的类别分布
                unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)
                logging.info(f"平衡后类别分布: {dict(zip(unique_balanced, counts_balanced))}")
            else:
                logging.info("📊 数据已经相对平衡，无需SMOTE处理")
                
        except Exception as e:
            logging.warning(f"⚠️ SMOTE处理失败，使用原始数据: {e}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.config.get('train_size', 0.8), shuffle=False
        )
        
        # 创建数据加载器
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # 🧠 模型集成 - 训练多个模型
        self.ensemble_models = []
        best_accuracy = 0
        
        # 获取训练参数
        input_size = X.shape[2]
        base_hidden_size = self.config['hidden_size']  # 使用配置中的hidden_size作为基础
        logging.info(f"📝 训练新模型，使用配置中的hidden_size={base_hidden_size}")
        
        # 🆕 添加训练曲线可视化数据收集
        all_train_losses = []
        all_val_accuracies = []
        
        for model_idx in range(3):  # 训练3个不同的模型
            logging.info(f"🔄 训练第 {model_idx + 1}/3 个集成模型...")
            
            # 初始化模型 - 每个模型使用稍微不同的参数
            hidden_size = base_hidden_size + (model_idx * 16)  # 不同模型不同隐藏层大小
            
            model = EnhancedLSTMModel(
                input_size, 
                hidden_size, 
                self.config['num_layers'], 
                1, 
                self.config['dropout']
            )
            model.to(self.device)
            
            # 定义损失函数和优化器
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'] * (1 + model_idx * 0.1))
            
            # 🆕 添加学习率调度器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            # 📅 训练循环
            best_model_accuracy = 0
            model_train_losses = []
            model_val_accuracies = []
            
            for epoch in range(self.config['epochs']):
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # 前向传播
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # 计算验证集准确率
                model.eval()
                correct = 0
                total = 0
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()
                        predicted = (outputs > 0.5).float()
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                accuracy = correct / total if total > 0 else 0
                
                # 记录训练指标
                avg_train_loss = train_loss/len(train_loader)
                avg_val_loss = val_loss/len(test_loader)
                model_train_losses.append(avg_train_loss)
                model_val_accuracies.append(accuracy)
                
                # 更新学习率调度器
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(avg_val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                
                # 手动记录学习率变化
                if new_lr != old_lr:
                    logging.info(f"学习率从 {old_lr:.6f} 调整为 {new_lr:.6f}")
                
                if accuracy > best_model_accuracy:
                    best_model_accuracy = accuracy
                    # 保存最佳模型状态
                    best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': accuracy,
                    }
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logging.info(f"模型{model_idx+1} Epoch [{epoch+1}/{self.config['epochs']}], "
                               f"Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                               f"Accuracy: {accuracy:.4f}")
            
            # 收集该模型的训练曲线数据
            all_train_losses.append(model_train_losses)
            all_val_accuracies.append(model_val_accuracies)
            
            # 🎯 每个模型训练完成后添加到集成
            # 加载最佳模型状态
            if 'best_model_state' in locals():
                model.load_state_dict(best_model_state['model_state_dict'])
                logging.info(f"加载最佳模型状态 (Epoch {best_model_state['epoch']+1}, Accuracy: {best_model_state['accuracy']:.4f})")
            
            self.ensemble_models.append({
                'model': model,
                'accuracy': best_model_accuracy,
                'weight': best_model_accuracy  # 使用准确率作为权重
            })
            
            # 更新最佳单模型
            if best_model_accuracy > best_accuracy:
                best_accuracy = best_model_accuracy
                self.model = model
        
        # 🔧 集成权重归一化
        total_weight = sum([m['weight'] for m in self.ensemble_models])
        for model_info in self.ensemble_models:
            model_info['weight'] /= total_weight
            
        logging.info(f"✅ 模型集成训练完成!")
        logging.info(f"📊 集成模型权重: {[f'{m['weight']:.3f}' for m in self.ensemble_models]}")
        logging.info(f"🎯 最佳单模型准确率: {best_accuracy:.4f}")
        
        # 🆕 生成训练曲线可视化
        try:
            self.plot_training_curves(all_train_losses, all_val_accuracies)
        except Exception as e:
            logging.error(f"生成训练曲线可视化失败: {e}")
        
        # 保存最佳模型
        self.save_model(input_size)
        return True
    
    def plot_training_curves(self, all_train_losses, all_val_accuracies):
        """生成训练曲线可视化图表"""
        try:
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 绘制训练损失曲线
            ax1.set_title('训练损失曲线', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            
            colors = ['#00BFFF', '#00FF7F', '#FFD700']  # 蓝色, 绿色, 黄色
            
            for i, losses in enumerate(all_train_losses):
                ax1.plot(losses, label=f'模型 {i+1}', color=colors[i], linewidth=2)
            
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 绘制验证准确率曲线
            ax2.set_title('验证准确率曲线', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(0, 1)
            
            for i, accuracies in enumerate(all_val_accuracies):
                ax2.plot(accuracies, label=f'模型 {i+1}', color=colors[i], linewidth=2)
            
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('training_curves.png', dpi=300)
            logging.info("✅ 训练曲线可视化已保存至 training_curves.png")
        except Exception as e:
            logging.error(f"绘制训练曲线失败: {e}")
    
    def predict(self):
        """使用增强预测系统进行预测 - 保持完整准确率"""
        try:
            # 获取实时数据
            data = self.data_processor.fetch_and_prepare_enhanced_data(self.api, retrain=False)
            if data is None:
                return None
            
            X, df = data
            if X is None or len(X) == 0:
                return None
            
            # 使用最新的数据点进行预测
            last_sequence = torch.FloatTensor(X[-1:]).to(self.device)
            
            predictions = []
            model_confidences = []
            
            # 🧠 集成预测
            if hasattr(self, 'ensemble_models') and self.ensemble_models:
                with torch.no_grad():
                    for i, model_info in enumerate(self.ensemble_models):
                        try:
                            model = model_info['model']
                            weight = model_info['weight']
                            
                            model.eval()
                            pred = model(last_sequence)
                            confidence = float(pred.cpu().numpy()[0][0])
                            
                            predictions.append(confidence)
                            model_confidences.append(confidence)
                        except Exception as e:
                            continue
                
                if not predictions:
                    return None
                
                # 检查预测结果中的NaN值
                valid_predictions = [p for p in predictions if not np.isnan(p)]
                if len(valid_predictions) == 0:
                    return None
                elif len(valid_predictions) < len(predictions):
                    predictions = valid_predictions
                    # 重新计算权重
                    valid_models = self.ensemble_models[:len(valid_predictions)]
                    total_weight = sum(model['weight'] for model in valid_models)
                    for model in valid_models:
                        model['weight'] = model['weight'] / total_weight
                
                # 加权平均预测
                weighted_prediction = sum([
                    pred * model_info['weight'] 
                    for pred, model_info in zip(predictions, self.ensemble_models[:len(predictions)])
                ])
                
                # 检查加权预测结果
                if np.isnan(weighted_prediction):
                    return None
                
                # 计算预测方差
                prediction_variance = np.var(predictions)
                
            else:
                # 单模型预测
                if self.model is None:
                    return None
                
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(last_sequence)
                    weighted_prediction = float(prediction.cpu().numpy()[0][0])
                    model_confidences = [weighted_prediction]
                    prediction_variance = 0
            
            # 🎯 完整智能置信度计算 - 保持所有准确率优化
            # 1. 基础置信度 (距离0.5的程度)
            base_confidence = abs(weighted_prediction - 0.5) * 2
            
            # 2. 🔧 优化的防过拟合机制
            confidence_adjustments = []
            
            # 🔧 改进的模型分歧惩罚：使用软惩罚机制
            if prediction_variance > 0.01:  # 如果模型分歧较大
                # 使用对数函数进行软惩罚，避免过度惩罚
                variance_penalty = min(0.25, np.log(1 + prediction_variance * 20) * 0.1)  # 🔧 最大惩罚25%
                confidence_adjustments.append(f"分歧惩罚: -{variance_penalty:.3f}")
                base_confidence *= (1 - variance_penalty)
            
            # 🔧 改进的一致性惩罚：使用渐进惩罚
            if prediction_variance < 0.001 and len(predictions) > 1:
                # 根据一致性程度进行渐进惩罚
                consistency_level = 1 - prediction_variance * 1000  # 0-1之间
                consistency_penalty = min(0.2, consistency_level * 0.2)  # 🔧 最大惩罚20%
                confidence_adjustments.append(f"过度一致惩罚: -{consistency_penalty:.3f}")
                base_confidence *= (1 - consistency_penalty)
            
            # 🔧 改进的极值惩罚：渐进式惩罚
            extreme_distance = max(0, max(0.1 - weighted_prediction, weighted_prediction - 0.9))
            if extreme_distance > 0:
                extreme_penalty = min(0.15, extreme_distance * 1.5)  # 🔧 最大惩罚15%
                confidence_adjustments.append(f"极值惩罚: -{extreme_penalty:.3f}")
                base_confidence *= (1 - extreme_penalty)
            
            # 3. 技术指标验证加成
            try:
                tech_strength = self.calculate_enhanced_technical_strength()
                if tech_strength > 0:
                    confidence_adjustments.append(f"技术指标加成: +{tech_strength:.3f}")
                    base_confidence += tech_strength
            except Exception as e:
                tech_strength = 0
            
            # 4. 🔧 增强的市场情绪验证加成
            try:
                sentiment_strength = self.calculate_enhanced_sentiment_strength()  # 🔧 使用增强版本
                if sentiment_strength > 0:
                    confidence_adjustments.append(f"情绪分析加成: +{sentiment_strength:.3f}")
                    base_confidence += sentiment_strength
            except Exception as e:
                sentiment_strength = 0
            
            # 5. 🔧 新增：恐慌贪婪指数加成
            try:
                if hasattr(self.data_processor, 'sentiment_analyzer') and self.data_processor.sentiment_analyzer:
                    fear_greed_index = self.data_processor.sentiment_analyzer.get_fear_greed_index()
                    # 将恐慌贪婪指数转换为置信度调整
                    if fear_greed_index < 20:  # 极度恐慌
                        fg_adjustment = 0.05 if weighted_prediction < 0.5 else -0.02  # 恐慌时看跌更可信
                    elif fear_greed_index > 80:  # 极度贪婪
                        fg_adjustment = 0.05 if weighted_prediction > 0.5 else -0.02  # 贪婪时看涨更可信
                    else:
                        fg_adjustment = 0
                    
                    if fg_adjustment != 0:
                        confidence_adjustments.append(f"恐慌贪婪指数: {fg_adjustment:+.3f}")
                        base_confidence += fg_adjustment
            except Exception as e:
                pass
            
            # 6. 🔧 软下限机制：确保置信度不会过低
            soft_floor = self.config.get('soft_confidence_floor', 15) / 100  # 默认15%
            if base_confidence < soft_floor:
                floor_adjustment = soft_floor - base_confidence
                confidence_adjustments.append(f"软下限保护: +{floor_adjustment:.3f}")
                base_confidence = soft_floor
            
            # 7. 最终置信度限制 (软下限%-95%)
            final_confidence = max(soft_floor, min(0.95, base_confidence))
            
            # 8. 🔧 动态交易门槛检查
            min_threshold = self.config.get('min_confidence_threshold', 60) / 100  # 默认60%
            trade_recommended = final_confidence >= min_threshold
            
            # 9. 预测结果
            direction = "上涨" if weighted_prediction > 0.5 else "下跌"
            trade_signal = "📈 买涨" if weighted_prediction > 0.5 else "📉 买跌"
            
            # 🔧 根据置信度调整交易信号
            if not trade_recommended:
                trade_signal = "⏸️ 观望"
            
            # 🆕 新增：严格的技术指标过滤条件
            # 获取当前技术指标
            try:
                # 获取最新的技术指标
                current_price = self.api.get_latest_price(self.config['symbol'])
                if current_price is None:
                    logging.warning("⚠️ 无法获取当前价格，跳过技术指标过滤")
                else:
                    # 获取技术指标数据
                    klines_1m = self.api.client.get_klines(symbol=self.config['symbol'], interval='1m', limit=100)
                    if klines_1m:
                        df_tech = pd.DataFrame(klines_1m, columns=[
                            'open_time', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_asset_volume', 'number_of_trades',
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                        ])
                        
                        # 数据类型转换
                        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                        for col in numeric_columns:
                            df_tech[col] = pd.to_numeric(df_tech[col])
                        
                        # 计算技术指标
                        # RSI
                        rsi_14 = ta.momentum.rsi(df_tech['close'], window=14).iloc[-1]
                        
                        # MACD
                        macd = ta.trend.MACD(df_tech['close'])
                        macd_line = macd.macd().iloc[-1]
                        macd_signal = macd.macd_signal().iloc[-1]
                        macd_hist = macd.macd_diff().iloc[-1]
                        
                        # 布林带
                        bollinger = ta.volatility.BollingerBands(df_tech['close'])
                        bb_upper = bollinger.bollinger_hband().iloc[-1]
                        bb_middle = bollinger.bollinger_mavg().iloc[-1]
                        bb_lower = bollinger.bollinger_lband().iloc[-1]
                        
                        # 移动平均线
                        sma_20 = ta.trend.sma_indicator(df_tech['close'], window=20).iloc[-1]
                        sma_50 = ta.trend.sma_indicator(df_tech['close'], window=50).iloc[-1]
                        
                        # 成交量
                        volume_sma = df_tech['volume'].rolling(20).mean().iloc[-1]
                        current_volume = df_tech['volume'].iloc[-1]
                        
                        # 🆕 严格的技术指标过滤规则
                        tech_filters_passed = True
                        filter_messages = []
                        
                        # 规则1: 超买/超卖过滤
                        if direction == "上涨" and rsi_14 > 70:
                            tech_filters_passed = False
                            filter_messages.append(f"RSI过高({rsi_14:.1f} > 70)，不适合做多")
                        elif direction == "下跌" and rsi_14 < 30:
                            tech_filters_passed = False
                            filter_messages.append(f"RSI过低({rsi_14:.1f} < 30)，不适合做空")
                        
                        # 规则2: MACD方向与预测方向一致性检查
                        if direction == "上涨" and macd_hist < 0:
                            tech_filters_passed = False
                            filter_messages.append(f"MACD柱状图为负({macd_hist:.4f})，与做多信号不一致")
                        elif direction == "下跌" and macd_hist > 0:
                            tech_filters_passed = False
                            filter_messages.append(f"MACD柱状图为正({macd_hist:.4f})，与做空信号不一致")
                        
                        # 规则3: 布林带位置检查
                        current_price = df_tech['close'].iloc[-1]
                        if direction == "上涨" and current_price > bb_upper:
                            tech_filters_passed = False
                            filter_messages.append(f"价格已超过布林带上轨，不适合做多")
                        elif direction == "下跌" and current_price < bb_lower:
                            tech_filters_passed = False
                            filter_messages.append(f"价格已低于布林带下轨，不适合做空")
                        
                        # 规则4: 趋势方向检查
                        if direction == "上涨" and current_price < sma_20:
                            tech_filters_passed = False
                            filter_messages.append(f"价格低于20日均线，与做多信号不一致")
                        elif direction == "下跌" and current_price > sma_20:
                            tech_filters_passed = False
                            filter_messages.append(f"价格高于20日均线，与做空信号不一致")
                        
                        # 规则5: 成交量确认
                        if current_volume < volume_sma * 0.7:
                            tech_filters_passed = False
                            filter_messages.append(f"成交量过低，信号可靠性降低")
                        
                        # 如果没有通过技术指标过滤，则不推荐交易
                        if not tech_filters_passed:
                            trade_recommended = False
                            trade_signal = "⏸️ 观望 (技术指标过滤)"
                            logging.info(f"⚠️ 技术指标过滤: {'; '.join(filter_messages)}")
            except Exception as e:
                logging.warning(f"⚠️ 技术指标过滤出错: {e}")
            
            result = {
                'timestamp': datetime.now(),
                'current_price': self.api.get_latest_price('BTCUSDT'),
                'prediction': weighted_prediction,
                'confidence': final_confidence * 100,  # 转换为百分比
                'direction': direction,
                'trade_signal': trade_signal,
                'trade_recommended': trade_recommended,  # 🔧 新增
                'confidence_threshold': min_threshold * 100,  # 🔧 新增
                'model_predictions': model_confidences,
                'technical_strength': tech_strength,
                'sentiment_strength': sentiment_strength,
                'prediction_variance': prediction_variance,
                'confidence_adjustments': confidence_adjustments,
                'tech_filters_passed': locals().get('tech_filters_passed', True),
                'filter_messages': locals().get('filter_messages', [])
            }
            
            return result
            
        except Exception as e:
            return None
    
    def calculate_technical_strength(self):
        """计算技术指标强度，用于增强置信度"""
        try:
            # 获取最新的技术指标数据
            api = self.api
            klines_1m = api.client.get_klines(symbol='BTCUSDT', interval='1m', limit=100)
            
            if not klines_1m:
                return 0
            
            df = pd.DataFrame(klines_1m, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 数据类型转换
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # 计算关键技术指标强度
            strength_score = 0
            
            # RSI强度 (超买/超卖信号)
            try:
                rsi = ta.momentum.rsi(df['close'], window=14).iloc[-1]
                if rsi < 30 or rsi > 70:  # 超买超卖
                    strength_score += 0.1
            except:
                pass
            
            # MACD强度
            try:
                macd = ta.trend.MACD(df['close'])
                macd_line = macd.macd().iloc[-1]
                macd_signal = macd.macd_signal().iloc[-1]
                if abs(macd_line - macd_signal) > 0.1:  # MACD背离强烈
                    strength_score += 0.1
            except:
                pass
            
            # 成交量强度
            try:
                volume_ma = df['volume'].rolling(20).mean().iloc[-1]
                current_volume = df['volume'].iloc[-1]
                if current_volume > volume_ma * 1.5:  # 成交量放大
                    strength_score += 0.05
            except:
                pass
            
            # 布林带位置
            try:
                bollinger = ta.volatility.BollingerBands(df['close'])
                bb_position = bollinger.bollinger_pband().iloc[-1]
                if bb_position > 0.8 or bb_position < 0.2:  # 接近布林带边界
                    strength_score += 0.05
            except:
                pass
            
            return min(0.3, strength_score)  # 最多贡献30%置信度
            
        except Exception as e:
            logging.warning(f"计算技术强度失败: {e}")
            return 0
    
    def calculate_enhanced_technical_strength(self):
        """计算增强技术指标强度，用于增强置信度"""
        try:
            # 获取最新的技术指标数据
            api = self.api
            klines_1m = api.client.get_klines(symbol='BTCUSDT', interval='1m', limit=100)
            
            if not klines_1m:
                logging.warning("无法获取K线数据")
                return 0
            
            df = pd.DataFrame(klines_1m, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 数据类型转换
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            if len(df) < 50:  # 数据不足
                logging.warning("K线数据不足，无法计算技术指标")
                return 0
            
            # 🎯 简化但稳定的技术指标强度计算
            strength_score = 0
            
            # 1. 简单移动平均线趋势
            try:
                ma_5 = df['close'].rolling(5).mean().iloc[-1]
                ma_20 = df['close'].rolling(20).mean().iloc[-1]
                current_price = df['close'].iloc[-1]
                
                # 均线排列
                if ma_5 > ma_20 and current_price > ma_5:  # 上升趋势
                    strength_score += 0.08
                elif ma_5 < ma_20 and current_price < ma_5:  # 下降趋势
                    strength_score += 0.08
            except Exception as e:
                logging.warning(f"计算移动平均线失败: {e}")
            
            # 2. 价格动量（简化版RSI概念）
            try:
                price_changes = df['close'].diff().dropna()
                if len(price_changes) >= 14:
                    gains = price_changes.where(price_changes > 0, 0)
                    losses = -price_changes.where(price_changes < 0, 0)
                    
                    avg_gain = gains.rolling(14).mean().iloc[-1]
                    avg_loss = losses.rolling(14).mean().iloc[-1]
                    
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        # RSI信号
                        if rsi < 30 or rsi > 70:  # 超买超卖
                            strength_score += 0.06
            except Exception as e:
                logging.warning(f"计算价格动量失败: {e}")
            
            # 3. 成交量确认
            try:
                volume_ma = df['volume'].rolling(20).mean().iloc[-1]
                current_volume = df['volume'].iloc[-1]
                
                if current_volume > volume_ma * 1.5:  # 成交量放大
                    strength_score += 0.05
            except Exception as e:
                logging.warning(f"计算成交量确认失败: {e}")
            
            # 4. 价格波动率
            try:
                price_volatility = df['close'].rolling(20).std().iloc[-1]
                volatility_ma = df['close'].rolling(20).std().rolling(10).mean().iloc[-1]
                
                if price_volatility > volatility_ma * 1.2:  # 波动增强
                    strength_score += 0.04
            except Exception as e:
                logging.warning(f"计算波动率失败: {e}")
            
            final_score = min(0.2, strength_score)  # 最多贡献20%置信度
            logging.info(f"📊 技术指标强度: {final_score:.4f}")
            return final_score
            
        except Exception as e:
            logging.error(f"计算增强技术指标强度失败: {e}")
            return 0
    
    def calculate_enhanced_sentiment_strength(self):
        """🔧 新增：计算增强的情绪强度，整合多维度情绪指标"""
        try:
            if not hasattr(self.data_processor, 'sentiment_analyzer') or not self.data_processor.sentiment_analyzer:
                return 0
            
            sentiment_analyzer = self.data_processor.sentiment_analyzer
            
            # 1. 订单簿情绪分析
            book_sentiment = sentiment_analyzer.analyze_order_book_sentiment()
            
            # 基础情绪得分 (0-1)
            base_sentiment = book_sentiment.get('sentiment_score', 0.5)
            
            # 🔧 买墙/卖墙分析
            wall_ratio = book_sentiment.get('wall_ratio', 1.0)
            wall_strength = 0
            if wall_ratio > 1.5:  # 买墙强于卖墙
                wall_strength = min(0.05, (wall_ratio - 1) * 0.02)
            elif wall_ratio < 0.67:  # 卖墙强于买墙
                wall_strength = max(-0.05, (wall_ratio - 1) * 0.02)
            
            # 🔧 深度不平衡分析
            deep_imbalance = book_sentiment.get('deep_imbalance', 0)
            depth_strength = min(0.03, abs(deep_imbalance) * 0.1) if abs(deep_imbalance) > 0.1 else 0
            if deep_imbalance < 0:  # 卖盘深度更大
                depth_strength = -depth_strength
            
            # 2. 🔧 恐慌贪婪指数影响
            try:
                fear_greed_index = sentiment_analyzer.get_fear_greed_index()
                fg_strength = 0
                
                if fear_greed_index < 25:  # 恐慌区域
                    fg_strength = (25 - fear_greed_index) * 0.002  # 最大+0.05
                elif fear_greed_index > 75:  # 贪婪区域
                    fg_strength = (fear_greed_index - 75) * 0.002  # 最大+0.05
                
            except:
                fg_strength = 0
            
            # 3. 🔧 社交媒体情绪（占位符，可扩展）
            try:
                social_sentiment = sentiment_analyzer.get_social_sentiment()
                twitter_sentiment = social_sentiment.get('twitter_sentiment', 0.5)
                social_strength = (twitter_sentiment - 0.5) * 0.02  # 最大±0.01
            except:
                social_strength = 0
            
            # 4. 综合情绪强度计算
            # 基础情绪偏离中性的程度
            base_strength = abs(base_sentiment - 0.5) * 0.1  # 最大0.05
            
            # 综合所有情绪因子
            total_sentiment_strength = (
                base_strength +
                wall_strength +
                depth_strength +
                fg_strength +
                social_strength
            )
            
            # 限制在合理范围内
            total_sentiment_strength = max(-0.1, min(0.1, total_sentiment_strength))
            
            logging.info(f"   📊 情绪分析详情:")
            logging.info(f"      基础情绪: {base_sentiment:.3f}")
            logging.info(f"      买卖墙比: {wall_ratio:.3f} (强度: {wall_strength:+.3f})")
            logging.info(f"      深度不平衡: {deep_imbalance:+.3f} (强度: {depth_strength:+.3f})")
            logging.info(f"      恐慌贪婪: {fear_greed_index:.1f} (强度: {fg_strength:+.3f})")
            
            return total_sentiment_strength
            
        except Exception as e:
            logging.warning(f"⚠️ 增强情绪强度计算失败: {e}")
            return 0

    def calculate_sentiment_strength(self):
        """计算情绪强度，用于增强置信度 - 保持向后兼容"""
        return self.calculate_enhanced_sentiment_strength()
    
    def simulate_trade(self, prediction_result):
        """模拟事件合约交易 - 使用马丁格尔策略"""
        if prediction_result is None:
            return
        
        # 使用马丁格尔策略确定投注金额
        bet_amount = self.martingale_bet_amounts[self.current_bet_level]
        payout_ratio = self.config['payout_ratio']  # 80%盈利率
        
        trade_record = {
            'trade_id': len(self.simulation_records) + 1,
            'timestamp': prediction_result['timestamp'],
            'entry_price': prediction_result['current_price'],
            'prediction': prediction_result['direction'],
            'confidence': prediction_result['confidence'],
            'trade_signal': prediction_result['trade_signal'],
            'bet_amount': bet_amount,
            'bet_level': self.current_bet_level,  # 记录当前投注级别
            'potential_payout': bet_amount * (1 + payout_ratio),  # 本金 + 盈利 = 总回报
            'status': 'OPEN',
            'result_checked': False,
            'profit_loss_amount': 0,
            'result_timestamp': None,
            'exit_price': None,
            'accuracy': None,
            'result': None
        }
        
        # 设置10分钟后检查结果
        check_time = prediction_result['timestamp'] + pd.Timedelta(minutes=self.config['prediction_minutes'])
        schedule.every().day.at(check_time.strftime('%H:%M')).do(
            self.check_trade_result, 
            len(self.simulation_records)
        ).tag(f'trade_check_{len(self.simulation_records)}')
        
        self.simulation_records.append(trade_record)
        self.save_simulation_records()
        
        # 显示交易记录
        print(f"\n{'='*80}")
        print(f"🎯 事件合约交易记录 #{trade_record['trade_id']} - 马丁格尔策略 (级别 {self.current_bet_level})")
        print(f"{'='*80}")
        print(f"⏰ 投注时间: {trade_record['timestamp']}")
        print(f"💰 当前价格: ${trade_record['entry_price']:.2f}")
        print(f"📊 预测方向: {trade_record['prediction']}")
        print(f"🔮 交易信号: {trade_record['trade_signal']}")
        print(f"🎯 置信度: {trade_record['confidence']:.1f}%")
        print(f"💵 投注金额: {trade_record['bet_amount']} USDT")
        print(f"🏆 潜在回报: {trade_record['potential_payout']:.2f} USDT")
        print(f"📈 事件合约: 预测正确获得{trade_record['potential_payout']:.2f}u，错误归零")
        print(f"⏳ 将在{self.config['prediction_minutes']}分钟后验证结果...")
        print(f"{'='*80}\n")
    
    def check_trade_result(self, trade_index):
        """检查事件合约交易结果 - 更新马丁格尔策略状态"""
        if trade_index >= len(self.simulation_records):
            return schedule.CancelJob
        
        trade = self.simulation_records[trade_index]
        if trade['status'] != 'OPEN':
            return schedule.CancelJob
        
        current_price = self.api.get_latest_price(self.config['symbol'])
        if current_price is None:
            logging.error(f"❌ 无法获取当前价格，交易 #{trade_index+1} 结果检查失败")
            return schedule.CancelJob
        
        entry_price = trade['entry_price']
        prediction = trade['prediction']
        bet_amount = trade['bet_amount']
        potential_payout = trade['potential_payout']
        
        # 判断预测是否正确
        actual_direction = '上涨' if current_price > entry_price else '下跌'
        is_correct = prediction == actual_direction
        
        # 计算事件合约盈亏
        if is_correct:
            # 预测正确：获得全部回报
            profit_loss_amount = potential_payout - bet_amount  # 实际盈利 = 回报 - 本金
            final_amount = potential_payout  # 最终获得的金额
            result = 'WIN'
            
            # 赢了，重置为初始投注额
            self.current_bet_level = 0
        else:
            # 预测错误：失去全部投注
            profit_loss_amount = -bet_amount  # 损失全部本金
            final_amount = 0  # 最终获得0
            result = 'LOSS'
            
            # 输了，提高投注级别，但不超过最大级别
            if self.current_bet_level < len(self.martingale_bet_amounts) - 1:
                self.current_bet_level += 1
            else:
                logging.warning("⚠️ 已达到最大投注级别，重置为初始投注")
                self.current_bet_level = 0
        
        # 更新交易记录
        trade['status'] = 'CLOSED'
        trade['exit_price'] = current_price
        trade['result_timestamp'] = pd.Timestamp.now()
        trade['profit_loss_amount'] = profit_loss_amount
        trade['final_amount'] = final_amount
        trade['result'] = result
        trade['accuracy'] = is_correct
        trade['price_change'] = current_price - entry_price
        trade['price_change_pct'] = (current_price - entry_price) / entry_price * 100
        
        self.simulation_records[trade_index] = trade
        
        # 显示验证结果
        result_emoji = "🎉" if result == 'WIN' else "💸"
        result_text = "预测正确" if is_correct else "预测错误"
        
        print(f"\n{'🎰' * 20}")
        print(f"📊 事件合约 #{trade_index+1} 结算结果 📊")
        print(f"{'🎰' * 20}")
        print(f"{result_emoji} {result_text}")
        print(f"🎯 预测: {prediction} | 实际: {actual_direction}")
        print(f"💰 进场价格: ${entry_price:,.2f}")
        print(f"💰 出场价格: ${current_price:,.2f}")
        print(f"📊 价格变化: {trade['price_change']:+.2f} USDT ({trade['price_change_pct']:+.2f}%)")
        print(f"💵 投注金额: {bet_amount:.2f} USDT")
        if is_correct:
            print(f"🏆 获得回报: {final_amount:.2f} USDT")
            print(f"💰 净盈利: +{profit_loss_amount:.2f} USDT")
            print(f"🔄 马丁格尔策略: 赢了，下次投注回到 {self.martingale_bet_amounts[self.current_bet_level]} USDT")
        else:
            print(f"💸 失去本金: {bet_amount:.2f} USDT")
            print(f"💰 净亏损: {profit_loss_amount:.2f} USDT")
            print(f"🔄 马丁格尔策略: 输了，下次投注增加到 {self.martingale_bet_amounts[self.current_bet_level]} USDT")
        print(f"⏰ 结算时间: {trade['result_timestamp']}")
        print(f"{'🎰' * 20}\n")
        
        # 保存交易记录
        self.save_simulation_records()
        
        return schedule.CancelJob
    
    def get_total_pnl_stats(self):
        """获取总体盈亏统计"""
        if not self.simulation_records:
            return {
                'total_trades': 0,
                'pending_trades': 0,
                'closed_trades': 0,
                'total_invested': 0,
                'total_returned': 0,
                'net_pnl': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0,
                'recent_win_rate': 0,
                'avg_win_amount': 0,
                'avg_loss_amount': 0,
                'roi_percentage': 0
            }
        
        closed_trades = [trade for trade in self.simulation_records if trade.get('status', 'CLOSED') == 'CLOSED']
        
        if not closed_trades:
            return {
                'total_trades': len(self.simulation_records),
                'pending_trades': len([t for t in self.simulation_records if t.get('status', 'OPEN') == 'OPEN']),
                'closed_trades': 0,
                'message': '暂无已完成的交易'
            }
        
        # 兼容新旧交易记录格式
        total_trades = len(closed_trades)
        total_invested = 0
        total_returned = 0
        
        for trade in closed_trades:
            # 兼容旧记录格式
            if 'bet_amount' in trade:
                # 新格式：事件合约
                total_invested += trade['bet_amount']
                total_returned += trade.get('final_amount', 0)
            else:
                # 旧格式：传统交易，转换为事件合约格式
                bet_amount = 5  # 默认投注额
                total_invested += bet_amount
                if trade.get('result') == 'WIN' or trade.get('profit_loss_amount', 0) > 0:
                    total_returned += bet_amount * 1.8  # 9 USDT回报
                # 如果是输，total_returned += 0（什么都不加）
        
        net_pnl = total_returned - total_invested
        
        # 统计胜负
        win_trades = []
        loss_trades = []
        
        for trade in closed_trades:
            if 'result' in trade:
                # 新格式
                if trade['result'] == 'WIN':
                    win_trades.append(trade)
                else:
                    loss_trades.append(trade)
            else:
                # 旧格式 - 根据profit_loss_amount判断
                if trade.get('profit_loss_amount', 0) > 0:
                    win_trades.append(trade)
                else:
                    loss_trades.append(trade)
        
        win_count = len(win_trades)
        loss_count = len(loss_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # 计算平均盈亏
        avg_win_amount = 0
        avg_loss_amount = 0
        
        if win_count > 0:
            win_amounts = []
            for trade in win_trades:
                if 'bet_amount' in trade and 'profit_loss_amount' in trade:
                    # 新格式：事件合约
                    win_amounts.append(trade['profit_loss_amount'])
                else:
                    # 旧格式：统一转换为事件合约盈利
                    win_amounts.append(4)  # 事件合约盈利4 USDT
            avg_win_amount = sum(win_amounts) / len(win_amounts)
        
        if loss_count > 0:
            loss_amounts = []
            for trade in loss_trades:
                if 'bet_amount' in trade and 'profit_loss_amount' in trade:
                    # 新格式：事件合约
                    loss_amounts.append(trade['profit_loss_amount'])
                else:
                    # 旧格式：统一转换为事件合约亏损
                    loss_amounts.append(-5)  # 事件合约亏损5 USDT
            avg_loss_amount = sum(loss_amounts) / len(loss_amounts)
        
        # 最近交易统计
        recent_trades = closed_trades[-10:] if len(closed_trades) >= 10 else closed_trades
        recent_win_count = 0
        for trade in recent_trades:
            if 'result' in trade:
                if trade['result'] == 'WIN':
                    recent_win_count += 1
            else:
                if trade.get('profit_loss_amount', 0) > 0:
                    recent_win_count += 1
        
        recent_win_rate = recent_win_count / len(recent_trades) if recent_trades else 0
        
        return {
            'total_trades': total_trades,
            'pending_trades': len([t for t in self.simulation_records if t.get('status', 'OPEN') == 'OPEN']),
            'total_invested': total_invested,
            'total_returned': total_returned,
            'net_pnl': net_pnl,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'recent_win_rate': recent_win_rate,
            'avg_win_amount': avg_win_amount,
            'avg_loss_amount': avg_loss_amount,
            'roi_percentage': (net_pnl / total_invested * 100) if total_invested > 0 else 0
        }
    
    def display_pnl_stats(self):
        """显示总体盈亏统计"""
        stats = self.get_total_pnl_stats()
        
        if 'message' in stats:
            print(f"\n📊 事件合约交易统计")
            print(f"总交易数: {stats['total_trades']}")
            print(f"待结算: {stats['pending_trades']}")
            print(f"已完成: {stats['closed_trades']}")
            print(f"状态: {stats['message']}")
        else:
            # 颜色处理
            pnl_emoji = "🟢" if stats['net_pnl'] >= 0 else "🔴"
            pnl_sign = "+" if stats['net_pnl'] >= 0 else ""
            
            print(f"\n{'💰' * 25}")
            print(f"📊 事件合约交易总体统计 📊")
            print(f"{'💰' * 25}")
            print(f"🎯 总交易次数: {stats['total_trades']} 笔")
            print(f"⏳ 待结算交易: {stats['pending_trades']} 笔")
            print(f"✅ 成功交易: {stats['win_count']} 笔")
            print(f"❌ 失败交易: {stats['loss_count']} 笔")
            print(f"📈 整体胜率: {stats['win_rate']:.1%}")
            print(f"📈 近期胜率: {stats['recent_win_rate']:.1%} (最近10笔)")
            print(f"{'─' * 50}")
            print(f"💵 总投注金额: {stats['total_invested']} USDT")
            print(f"💰 总回收金额: {stats['total_returned']} USDT")
            print(f"{pnl_emoji} 净盈亏: {pnl_sign}{stats['net_pnl']} USDT")
            print(f"📊 投资回报率: {pnl_sign}{stats['roi_percentage']:.2f}%")
            print(f"{'─' * 50}")
            print(f"🏆 平均单笔盈利: +{stats['avg_win_amount']:.2f} USDT")
            print(f"💸 平均单笔亏损: {stats['avg_loss_amount']:.2f} USDT")
            print(f"{'💰' * 25}")
        
        # 添加返回机制
        print(f"\n{'🔙' * 20}")
        print("📌 按任意键返回监控状态...")
        print(f"{'🔙' * 20}")
        
        # 等待用户按键
        if sys.platform == 'win32':
            import msvcrt
            msvcrt.getch()  # Windows下等待按键
        else:
            import termios, tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.cbreak(fd)
                sys.stdin.read(1)  # Linux/Mac下等待按键
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        print("🔄 返回监控状态...\n")
    
    def calculate_performance(self):
        """计算绩效 - 适配事件合约"""
        stats = self.get_total_pnl_stats()
        
        if 'message' in stats:
            logging.info("📊 暂无已完成的交易，无法计算绩效")
            return stats
        
        logging.info(f"📊 事件合约绩效报告:")
        logging.info(f"   总交易次数: {stats['total_trades']}")
        logging.info(f"   胜率: {stats['win_rate']:.2%}")
        logging.info(f"   净盈亏: {stats['net_pnl']:+.2f} USDT")
        logging.info(f"   投资回报率: {stats['roi_percentage']:+.2f}%")
        
        return stats
    
    def save_simulation_records(self):
        with open('simulation_records.json', 'w') as f:
            json.dump(self.simulation_records, f, indent=4, default=str)
    
    def load_simulation_records(self):
        if os.path.exists('simulation_records.json'):
            try:
                with open('simulation_records.json', 'r') as f:
                    self.simulation_records = json.load(f)
                logging.info(f"加载了 {len(self.simulation_records)} 条交易记录")
            except Exception as e:
                logging.error(f"加载交易记录时出错: {e}")
    
    def plot_performance(self):
        if not self.simulation_records:
            logging.info("没有交易记录，无法绘制绩效图表")
            return
        
        closed_trades = [trade for trade in self.simulation_records if trade['status'] == 'CLOSED']
        if not closed_trades:
            logging.info("没有已完成的交易，无法绘制绩效图表")
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(closed_trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df = df.sort_values('timestamp')
        
        # 计算累计盈亏
        df['cumulative_profit_loss'] = df['profit_loss'].cumsum()
        
        # 设置颜色主题
        plt.style.use('dark_background')
        green_color = '#00FF7F'  # 绿色
        red_color = '#FF5555'    # 红色
        blue_color = '#00BFFF'   # 蓝色
        yellow_color = '#FFD700' # 黄色
        purple_color = '#BA55D3' # 紫色
        
        # 创建子图
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2)
        
        # 1. 累计盈亏曲线 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['exit_time'], df['cumulative_profit_loss'], color=blue_color, linewidth=2)
        ax1.fill_between(df['exit_time'], df['cumulative_profit_loss'], 0, 
                         where=(df['cumulative_profit_loss'] >= 0), color=green_color, alpha=0.3)
        ax1.fill_between(df['exit_time'], df['cumulative_profit_loss'], 0, 
                         where=(df['cumulative_profit_loss'] < 0), color=red_color, alpha=0.3)
        ax1.set_title('累计盈亏 (%)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # 2. 每笔交易盈亏 (右上)
        ax2 = fig.add_subplot(gs[0, 1])
        colors = [green_color if pl > 0 else red_color for pl in df['profit_loss']]
        ax2.bar(range(len(df)), df['profit_loss'], color=colors, alpha=0.7)
        ax2.set_title('每笔交易盈亏 (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('交易编号')
        ax2.set_ylabel('盈亏 (%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # 3. 胜率统计饼图 (左中)
        ax3 = fig.add_subplot(gs[1, 0])
        win_count = sum(1 for trade in closed_trades if trade['result'] == 'WIN')
        loss_count = len(closed_trades) - win_count
        ax3.pie([win_count, loss_count], 
                labels=['盈利', '亏损'], 
                colors=[green_color, red_color],
                autopct='%1.1f%%', 
                startangle=90,
                wedgeprops={'alpha': 0.7})
        ax3.set_title('交易胜率统计', fontsize=14, fontweight='bold')
        
        # 4. 盈亏分布直方图 (右中)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(df['profit_loss'], bins=20, color=blue_color, alpha=0.7)
        ax4.set_title('盈亏分布', fontsize=14, fontweight='bold')
        ax4.set_xlabel('盈亏 (%)')
        ax4.set_ylabel('频率')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # 5. 信心度与盈亏关系散点图 (底部跨列)
        ax5 = fig.add_subplot(gs[2, :])
        scatter = ax5.scatter(df['confidence'], df['profit_loss'], 
                              c=df['profit_loss'], cmap='coolwarm', 
                              s=100, alpha=0.7)
        ax5.set_title('置信度与盈亏关系', fontsize=14, fontweight='bold')
        ax5.set_xlabel('置信度')
        ax5.set_ylabel('盈亏 (%)')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        fig.colorbar(scatter, ax=ax5)
        
        # 添加统计数据文本框
        win_rate = win_count / len(closed_trades) if closed_trades else 0
        avg_profit = sum(trade['profit_loss'] for trade in closed_trades if trade['profit_loss'] > 0) / win_count if win_count else 0
        avg_loss = sum(trade['profit_loss'] for trade in closed_trades if trade['profit_loss'] <= 0) / loss_count if loss_count else 0
        total_pl = sum(trade['profit_loss'] for trade in closed_trades)
        
        stats_text = f"总交易次数: {len(closed_trades)}\n"
        stats_text += f"胜率: {win_rate:.2%}\n"
        stats_text += f"平均盈利: {avg_profit:.2f}%\n"
        stats_text += f"平均亏损: {avg_loss:.2f}%\n"
        stats_text += f"总盈亏: {total_pl:.2f}%"
        
        # 在图表空白处添加统计文本
        plt.figtext(0.92, 0.5, stats_text, 
                   bbox=dict(facecolor='gray', alpha=0.1),
                   fontsize=12, ha='right')
        
        # 添加标题和时间戳
        plt.suptitle('比特币价格预测交易系统绩效分析', fontsize=18, fontweight='bold')
        plt.figtext(0.5, 0.01, f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                   ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig('performance.png', dpi=300)
        logging.info("绩效图表已保存至 performance.png")
        
        # 额外生成策略效果图
        self.plot_strategy_performance(df)
    
    def plot_strategy_performance(self, df):
        """绘制策略效果对比图"""
        plt.style.use('dark_background')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 获取第一个和最后一个交易时间点
        start_time = df['timestamp'].min()
        end_time = df['exit_time'].max()
        
        # 重新从币安获取这段时间的比特币价格数据
        try:
            lookback_time = start_time.strftime("%d %b, %Y")
            klines = self.api.get_multi_timeframe_data(
                self.config['symbol'], 
                self.config['intervals'], 
                lookback_time
            )
            
            if klines:
                # 转换为DataFrame
                price_df = pd.DataFrame(klines['1m'], columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # 转换数据类型
                price_df['open_time'] = pd.to_datetime(price_df['open_time'], unit='ms')
                price_df['close'] = pd.to_numeric(price_df['close'])
                
                # 筛选日期范围
                price_df = price_df[(price_df['open_time'] >= start_time) & (price_df['open_time'] <= end_time)]
                
                if not price_df.empty:
                    # 计算价格变化百分比
                    initial_price = price_df['close'].iloc[0]
                    price_df['price_change_pct'] = (price_df['close'] / initial_price - 1) * 100
                    
                    # 绘制价格变化曲线
                    ax.plot(price_df['open_time'], price_df['price_change_pct'], 
                            color='gray', alpha=0.7, label='BTC价格变化(%)')
                    
                    # 绘制策略曲线
                    ax.plot(df['exit_time'], df['cumulative_profit_loss'], 
                            color='#00BFFF', linewidth=2, label='策略累计收益(%)')
                    
                    # 绘制买入点和卖出点
                    for _, trade in df.iterrows():
                        marker = '^' if trade['position'] == 'LONG' else 'v'
                        color = 'green' if trade['result'] == 'WIN' else 'red'
                        ax.scatter(trade['timestamp'], 0, marker=marker, color=color, s=100, alpha=0.7)
                    
                    # 设置图表
                    ax.set_title('策略收益与BTC价格对比', fontsize=16, fontweight='bold')
                    ax.set_xlabel('时间')
                    ax.set_ylabel('收益率 (%)')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    ax.legend()
                    
                    # 添加标注
                    plt.figtext(0.5, 0.01, '绿色三角形:盈利交易  红色三角形:亏损交易  向上三角:做多  向下三角:做空', 
                              ha='center', fontsize=10)
                    
                    plt.tight_layout()
                    plt.savefig('strategy_performance.png', dpi=300)
                    logging.info("策略对比图表已保存至 strategy_performance.png")
        except Exception as e:
            logging.error(f"绘制策略对比图表时出错: {e}")
    
    def backtest(self, start_date=None, end_date=None, period_days=30):
        """
        对历史数据进行回测，验证模型预测准确率
        
        参数:
            start_date: 回测开始日期，格式: 'YYYY-MM-DD'
            end_date: 回测结束日期，格式: 'YYYY-MM-DD'
            period_days: 如果未指定日期，回测最近的天数
        """
        if self.model is None:
            logging.error("模型未加载，无法进行回测")
            return
        
        # 设置回测日期范围
        if start_date is None:
            start_time = (datetime.now() - timedelta(days=period_days)).strftime("%d %b, %Y")
        else:
            start_time = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d %b, %Y")
        
        if end_date is None:
            end_time = datetime.now().strftime("%d %b, %Y")
        else:
            end_time = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d %b, %Y")
        
        logging.info(f"开始回测 - 时间范围: {start_time} 到 {end_time}")
        
        try:
            # 获取历史K线数据
            klines = self.api.get_multi_timeframe_data(
                self.config['symbol'], 
                self.config['intervals'], 
                start_time
            )
            
            if not klines or len(klines['1m']) < 100:  # 确保有足够的数据
                logging.error(f"回测数据不足，获取到 {len(klines['1m']) if klines['1m'] else 0} 条记录")
                return
            
            # 转换为DataFrame
            df = pd.DataFrame(klines['1m'], columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 转换数据类型
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                            'quote_asset_volume', 'number_of_trades',
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            # 创建技术指标
            df = self.data_processor.add_enhanced_technical_indicators(df, '1m')
            
            # 丢弃NaN值
            df = df.dropna().reset_index(drop=True)
            
            # 准备回测数据
            total_samples = len(df) - self.config['prediction_minutes']
            backtest_results = []
            
            logging.info(f"开始回测 {total_samples} 个时间点...")
            
            # 回测每个时间点
            for i in range(0, total_samples, 10):  # 每10个时间点采样一次，提高效率
                if i % 100 == 0:
                    logging.info(f"回测进度: {i}/{total_samples}")
                
                # 获取当前时间点的数据
                current_data = df.iloc[:i + self.config['prediction_minutes']]
                
                # 准备特征数据
                features = current_data[self.data_processor.features].values
                if len(features) < 10:  # 确保有足够的序列长度
                    continue
                
                # 归一化特征
                features_scaled = self.data_processor.feature_scalers['1m'].transform(features)
                
                # 创建序列
                X = []
                seq_length = 10
                if len(features_scaled) <= seq_length:
                    continue
                    
                X.append(features_scaled[-seq_length:])
                X = torch.FloatTensor(np.array(X))
                
                # 预测
                self.model.eval()
                with torch.no_grad():
                    X_tensor = X.to(self.device)
                    prediction = self.model(X_tensor)
                    probability = prediction.item()
                
                # 获取当前价格和未来价格
                current_price = float(current_data['close'].iloc[-1])
                future_price = float(df['close'].iloc[i + self.config['prediction_minutes']])
                
                # 确定预测结果
                predicted_direction = 'UP' if probability >= self.config['threshold'] else 'DOWN'
                actual_direction = 'UP' if future_price > current_price else 'DOWN'
                is_correct = predicted_direction == actual_direction
                confidence = abs(probability - 0.5) * 2  # 0-1之间的置信度
                
                # 计算价格变化
                price_change_pct = (future_price - current_price) / current_price * 100
                
                # 计算回测盈亏
                if predicted_direction == 'UP':
                    pnl = price_change_pct  # 做多的盈亏
                else:
                    pnl = -price_change_pct  # 做空的盈亏
                
                # 记录结果
                result = {
                    'timestamp': current_data['open_time'].iloc[-1],
                    'current_price': current_price,
                    'future_price': future_price,
                    'prediction': predicted_direction,
                    'actual': actual_direction,
                    'probability': probability,
                    'confidence': confidence,
                    'is_correct': is_correct,
                    'price_change_pct': price_change_pct,
                    'pnl': pnl
                }
                
                backtest_results.append(result)
            
            # 计算回测统计
            if not backtest_results:
                logging.error("回测结果为空")
                return
                
            backtest_df = pd.DataFrame(backtest_results)
            
            # 基本统计
            total_trades = len(backtest_df)
            correct_trades = sum(backtest_df['is_correct'])
            accuracy = correct_trades / total_trades if total_trades > 0 else 0
            
            # 按置信度分层分析
            backtest_df['confidence_bin'] = pd.cut(backtest_df['confidence'], 
                                                 bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                                 labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
            
            confidence_stats = backtest_df.groupby('confidence_bin').agg({
                'is_correct': ['count', 'mean'],
                'pnl': ['mean', 'sum', 'std']
            })
            
            # 计算累计收益
            backtest_df['cumulative_pnl'] = backtest_df['pnl'].cumsum()
            
            # 输出回测结果
            logging.info("\n" + "="*50)
            logging.info("回测结果摘要:")
            logging.info(f"回测时间范围: {backtest_df['timestamp'].min()} 到 {backtest_df['timestamp'].max()}")
            logging.info(f"总样本数: {total_trades}")
            logging.info(f"总体准确率: {accuracy:.4f}")
            logging.info(f"平均收益率: {backtest_df['pnl'].mean():.4f}%")
            logging.info(f"累计收益率: {backtest_df['pnl'].sum():.4f}%")
            logging.info(f"最大单次收益: {backtest_df['pnl'].max():.4f}%")
            logging.info(f"最大单次亏损: {backtest_df['pnl'].min():.4f}%")
            logging.info(f"收益标准差: {backtest_df['pnl'].std():.4f}%")
            logging.info("="*50)
            
            # 绘制回测图表
            self.plot_backtest_results(backtest_df)
            
            # 返回回测结果
            return backtest_df
            
        except Exception as e:
            logging.error(f"回测过程中出错: {e}")
            return None
    
    def plot_backtest_results(self, backtest_df):
        """绘制回测结果图表"""
        plt.style.use('dark_background')
        
        # 创建图表布局
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(4, 2)
        
        # 设置颜色
        blue_color = '#00BFFF'   # 蓝色
        green_color = '#00FF7F'  # 绿色
        red_color = '#FF5555'    # 红色
        yellow_color = '#FFD700' # 黄色
        
        # 1. 价格与预测 (顶部跨两列)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(backtest_df['timestamp'], backtest_df['current_price'], color='gray', alpha=0.7, label='BTC价格')
        
        # 标记正确和错误的预测
        correct_df = backtest_df[backtest_df['is_correct']]
        incorrect_df = backtest_df[~backtest_df['is_correct']]
        
        # 上涨预测
        up_correct = correct_df[correct_df['prediction'] == 'UP']
        up_incorrect = incorrect_df[incorrect_df['prediction'] == 'UP']
        
        # 下跌预测
        down_correct = correct_df[correct_df['prediction'] == 'DOWN']
        down_incorrect = incorrect_df[incorrect_df['prediction'] == 'DOWN']
        
        # 绘制预测点
        ax1.scatter(up_correct['timestamp'], up_correct['current_price'], 
                   color=green_color, marker='^', s=50, alpha=0.7, label='正确上涨预测')
        ax1.scatter(down_correct['timestamp'], down_correct['current_price'], 
                   color=green_color, marker='v', s=50, alpha=0.7, label='正确下跌预测')
        ax1.scatter(up_incorrect['timestamp'], up_incorrect['current_price'], 
                   color=red_color, marker='^', s=50, alpha=0.7, label='错误上涨预测')
        ax1.scatter(down_incorrect['timestamp'], down_incorrect['current_price'], 
                   color=red_color, marker='v', s=50, alpha=0.7, label='错误下跌预测')
        
        ax1.set_title('BTC价格与预测点', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 累计收益 (第二行左)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(backtest_df['timestamp'], backtest_df['cumulative_pnl'], color=blue_color, linewidth=2)
        ax2.fill_between(backtest_df['timestamp'], backtest_df['cumulative_pnl'], 0, 
                         where=(backtest_df['cumulative_pnl'] >= 0), color=green_color, alpha=0.3)
        ax2.fill_between(backtest_df['timestamp'], backtest_df['cumulative_pnl'], 0, 
                         where=(backtest_df['cumulative_pnl'] < 0), color=red_color, alpha=0.3)
        ax2.set_title('回测累计收益 (%)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # 3. 每次预测收益 (第二行右)
        ax3 = fig.add_subplot(gs[1, 1])
        colors = [green_color if pnl > 0 else red_color for pnl in backtest_df['pnl']]
        ax3.bar(range(len(backtest_df)), backtest_df['pnl'], color=colors, alpha=0.7)
        ax3.set_title('每次预测收益 (%)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('预测序号')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # 4. 置信度分布与准确率 (第三行左)
        ax4 = fig.add_subplot(gs[2, 0])
        confidence_groups = backtest_df.groupby('confidence_bin')
        
        conf_bins = []
        accuracies = []
        counts = []
        
        for conf_bin, group in confidence_groups:
            if not conf_bin or pd.isna(conf_bin):
                continue
                
            conf_bins.append(conf_bin)
            accuracies.append(group['is_correct'].mean())
            counts.append(len(group))
        
        # 绘制置信度与准确率关系
        bars = ax4.bar(conf_bins, accuracies, color=blue_color, alpha=0.7)
        
        # 添加交易次数标签
        for i, bar in enumerate(bars):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'n={counts[i]}', ha='center', va='bottom', fontsize=9)
        
        ax4.set_title('置信度与准确率关系', fontsize=14, fontweight='bold')
        ax4.set_xlabel('置信度区间')
        ax4.set_ylabel('准确率')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # 5. 收益分布直方图 (第三行右)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(backtest_df['pnl'], bins=30, color=blue_color, alpha=0.7)
        ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        ax5.set_title('收益分布', fontsize=14, fontweight='bold')
        ax5.set_xlabel('收益率 (%)')
        ax5.grid(True, alpha=0.3)
        
        # 6. 置信度与收益关系散点图 (第四行跨列)
        ax6 = fig.add_subplot(gs[3, :])
        scatter = ax6.scatter(backtest_df['confidence'], backtest_df['pnl'], 
                             c=backtest_df['pnl'], cmap='coolwarm', 
                             s=80, alpha=0.7)
        ax6.set_title('置信度与收益关系', fontsize=14, fontweight='bold')
        ax6.set_xlabel('置信度')
        ax6.set_ylabel('收益率 (%)')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        fig.colorbar(scatter, ax=ax6)
        
        # 添加统计数据文本框
        stats_text = f"总样本数: {len(backtest_df)}\n"
        stats_text += f"总体准确率: {backtest_df['is_correct'].mean():.2%}\n"
        stats_text += f"平均收益率: {backtest_df['pnl'].mean():.2f}%\n"
        stats_text += f"累计收益率: {backtest_df['pnl'].sum():.2f}%\n"
        stats_text += f"最佳置信度阈值: {conf_bins[np.argmax(accuracies)]}"
        
        plt.figtext(0.92, 0.5, stats_text, 
                   bbox=dict(facecolor='gray', alpha=0.1),
                   fontsize=12, ha='right')
        
        # 添加标题和时间戳
        plt.suptitle('比特币价格预测系统回测结果', fontsize=18, fontweight='bold')
        plt.figtext(0.5, 0.01, f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                   ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig('backtest_results.png', dpi=300)
        logging.info("回测结果图表已保存至 backtest_results.png")

    def continuous_predict_and_trade(self):
        """实时监控模式 - 保持完整预测准确率的同时优化响应速度"""
        last_price = None
        last_prediction_time = None
        loop_count = 0
        last_trade_time = None
        
        print(f"\n{'🚀' * 50}")
        print(f"🎯 启动实时监控模式 - 完整预测系统")
        print(f"💡 持续监控价格变化，完整技术分析，发现高置信度交易机会立即提醒")
        print(f"🔧 保持所有技术指标和情绪分析，确保最高预测准确率")
        print(f"💰 马丁格尔投注策略: 初始5U，输则翻倍(10U→30U→90U→250U)，赢则回到5U")
        print(f"⚡ 按 's' 查看统计 | 按 'q' 退出程序")
        print(f"{'🚀' * 50}\n")
        
        def check_keyboard_input():
            """检查键盘输入的函数"""
            if sys.platform == 'win32':
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    if key == 's':
                        self.display_pnl_stats()
                    elif key == 'q':
                        logging.info("👋 用户选择退出程序")
                        return True
            else:
                # Linux/Mac系统的非阻塞输入检查
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    key = sys.stdin.read(1).lower()
                    if key == 's':
                        self.display_pnl_stats()
                    elif key == 'q':
                        logging.info("👋 用户选择退出程序")
                        return True
            return False
        
        def has_pending_trades():
            """检查是否有待结算的交易"""
            pending_trades = [t for t in self.simulation_records if t.get('status', 'OPEN') == 'OPEN']
            return len(pending_trades) > 0
        
        def should_trade_now(current_time):
            """检查是否可以进行新交易"""
            # 检查是否有待结算交易
            if has_pending_trades():
                return False, "有待结算交易"
            
            # 检查交易冷却期（10分钟）
            if last_trade_time and (current_time - last_trade_time).total_seconds() < 600:
                time_left = 600 - (current_time - last_trade_time).total_seconds()
                return False, f"交易冷却期，还需{time_left:.0f}秒"
            
            return True, "可以交易"
        
        def should_run_prediction(current_time, current_price):
            """智能判断是否需要运行完整预测"""
            # 价格变化触发
            price_changed = last_price is None or current_price != last_price
            
            # 时间触发 - 每30秒至少预测一次，确保不错过机会
            time_trigger = (last_prediction_time is None or 
                          (current_time - last_prediction_time).total_seconds() >= 30)
            
            # 显著价格变化触发 - 价格变化超过0.05%立即预测
            significant_change = False
            if last_price is not None:
                price_change_pct = abs(current_price - last_price) / last_price * 100
                if price_change_pct >= 0.05:
                    significant_change = True
            
            return price_changed or time_trigger or significant_change
        
        while True:
            try:
                # 检查用户输入
                if check_keyboard_input():
                    break
                
                current_time = datetime.now()
                
                # 获取当前价格 - 快速API调用
                current_price = self.api.get_latest_price('BTCUSDT')
                if current_price is None:
                    print("⚠️ 获取价格失败，重试中...")
                    time.sleep(1)
                    continue
                
                # 检查是否可以交易
                can_trade, trade_status = should_trade_now(current_time)
                
                # 显示实时监控信息（每100次循环显示一次，避免刷屏）
                loop_count += 1
                if loop_count % 100 == 0:
                    status_emoji = "✅" if can_trade else "⏳"
                    print(f"{status_emoji} 监控中... 价格: ${current_price:.2f} | 状态: {trade_status} | 循环: #{loop_count}")
                
                # 智能预测触发 - 只在需要时进行完整预测
                if should_run_prediction(current_time, current_price):
                    last_prediction_time = current_time
                    last_price = current_price
                    
                    # 执行完整预测分析 - 保持所有技术指标和情绪分析
                    try:
                        prediction_result = self.predict()
                        
                        if prediction_result is not None:
                            confidence = prediction_result['confidence']
                            
                            # 根据置信度级别进行不同处理
                            if confidence >= 85:  # 极高置信度 - 立即交易
                                if can_trade:
                                    prediction_result['bet_amount'] = 5
                                    
                                    # 立即提醒 - 多次提示音
                                    for _ in range(3):
                                        print('\a')  # 系统提示音
                                        time.sleep(0.1)
                                    
                                    print(f"\n{'🚨' * 60}")
                                    print(f"🚨🚨🚨 发现极高置信度交易机会！🚨🚨🚨")
                                    print(f"⏰ 时间: {current_time.strftime('%H:%M:%S')}")
                                    print(f"💰 价格: ${current_price:,.2f}")
                                    print(f"🎯 信号: {prediction_result['trade_signal']}")
                                    print(f"📊 方向: 预测10分钟后价格{prediction_result['direction']}")
                                    print(f"🔥 置信度: {confidence:.1f}% (极高)")
                                    print(f"💵 投注: 5 USDT")
                                    print(f"🏆 预期回报: 9 USDT (盈利4u)")
                                    
                                    # 显示详细分析信息
                                    if 'technical_strength' in prediction_result:
                                        print(f"📈 技术强度: {prediction_result['technical_strength']:.3f}")
                                    if 'sentiment_strength' in prediction_result:
                                        print(f"😊 情绪强度: {prediction_result['sentiment_strength']:.3f}")
                                    if 'confidence_adjustments' in prediction_result:
                                        print(f"🔧 置信度调整: {', '.join(prediction_result['confidence_adjustments'])}")
                                    
                                    # 显示马丁格尔策略信息
                                    print(f"💰 马丁格尔投注: {self.martingale_bet_amounts[self.current_bet_level]} USDT (级别 {self.current_bet_level})")
                                    if self.current_bet_level > 0:
                                        print(f"📊 马丁格尔策略: 之前亏损，增加投注额以追回损失")
                                    else:
                                        print(f"📊 马丁格尔策略: 初始投注额")
                                    
                                    print(f"{'🚨' * 60}\n")
                                    
                                    # 执行交易
                                    self.simulate_trade(prediction_result)
                                    last_trade_time = current_time
                                else:
                                    print(f"🔥 极高置信度信号: {prediction_result['trade_signal']} | 置信度: {confidence:.1f}% | 价格: ${current_price:.2f} | {trade_status}")
                                
                            elif confidence >= 75:  # 高置信度 - 提醒但不交易
                                print(f"🔥 高置信度信号: {prediction_result['trade_signal']} | 置信度: {confidence:.1f}% | 价格: ${current_price:.2f}")
                                
                            elif confidence >= 65:  # 中等置信度 - 简单提醒
                                if loop_count % 50 == 0:  # 减少显示频率
                                    print(f"📊 中等置信度: {prediction_result['trade_signal']} | 置信度: {confidence:.1f}% | 价格: ${current_price:.2f}")
                            
                            # 低置信度不显示，避免刷屏
                            
                    except Exception as e:
                        if loop_count % 200 == 0:  # 每200次循环才显示一次错误，避免刷屏
                            print(f"⚠️ 预测分析出错: {e}")
                
                # 处理定时任务（检查交易结果）
                schedule.run_pending()
                
                # 优化的等待时间 - 0.5秒确保实时性，但不会过度消耗资源
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                logging.info("⚠️ 用户中断监控")
                break
            except Exception as e:
                logging.error(f"❌ 监控出错: {e}")
                time.sleep(2)  # 错误时稍微等待一下

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='🚀 比特币价格预测系统 - 智能合约交易助手')
    parser.add_argument('--train', action='store_true', help='训练新模型')
    parser.add_argument('--backtest', action='store_true', help='进行历史回测')
    parser.add_argument('--start_date', type=str, help='回测开始日期，格式: YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, help='回测结束日期，格式: YYYY-MM-DD')
    parser.add_argument('--days', type=int, default=30, help='回测天数(如果未指定日期)')
    parser.add_argument('--interval', type=int, default=5, help='数据刷新间隔(秒)，默认5秒')
    parser.add_argument('--continuous', action='store_true', help='启用连续实时分析模式')
    parser.add_argument('--stats', action='store_true', help='显示总体盈亏统计')
    args = parser.parse_args()
    
    logging.info("🚀 启动比特币价格预测系统 - 智能合约交易助手")
    logging.info("📊 支持多时间周期分析 | 🔍 大单检测 | 📈 市场深度分析")
    logging.info("🎰 事件合约模式: 投注5u，预测正确获得9u，错误归零")
    
    # 加载配置
    config = load_config()
    
    # 检查API密钥配置
    if not config['api_key'] or not config['api_secret']:
        print("🔑 请输入您的币安API信息:")
        api_key = input("API Key: ")
        api_secret = input("API Secret: ")
        config['api_key'] = api_key
        config['api_secret'] = api_secret
        save_config(config)
    
    # 创建预测器
    predictor = BitcoinPredictor(config)
    
    # 加载历史交易记录
    predictor.load_simulation_records()
    
    # 如果只是查看统计信息
    if args.stats:
        predictor.display_pnl_stats()
        return
    
    # 如果指定了训练模式
    if args.train:
        logging.info("🔧 强制训练新模型...")
        # 删除旧模型文件（如果存在）
        if os.path.exists('model.pth'):
            logging.info("📝 删除旧模型文件...")
            try:
                os.remove('model.pth')
                logging.info("✅ 旧模型文件已删除")
            except Exception as e:
                logging.error(f"❌ 删除旧模型文件失败: {e}")
        
        if predictor.train():
            logging.info("🎉 模型训练完成！")
        else:
            logging.error("❌ 模型训练失败")
        return
    
    # 如果没有找到模型，训练一个新模型
    if predictor.model is None:
        logging.info("🔧 没有找到预训练模型，开始训练新模型...")
        
        # 如果加载模型失败但文件存在，可能是模型与当前代码不兼容
        # 删除旧模型文件并重新训练
        if os.path.exists('model.pth'):
            logging.warning("⚠️ 发现不兼容的模型文件，删除并重新训练...")
            try:
                os.remove('model.pth')
            except Exception as e:
                logging.error(f"❌ 删除旧模型文件失败: {e}")
        
        if not predictor.train():
            logging.error("❌ 模型训练失败，退出程序")
            return
    
    # 如果指定了回测模式
    if args.backtest:
        logging.info("📊 开始执行历史回测...")
        predictor.backtest(args.start_date, args.end_date, args.days)
        return
    
    # 获取数据刷新间隔
    refresh_interval = args.interval
    
    # 记录上次预测时间和价格，避免重复预测
    last_prediction_time = 0
    last_price = 0
    prediction_cooldown = 60  # 预测冷却时间60秒
    
    # 程序结束时计算最终绩效
    try:
        predictor.calculate_performance()
        predictor.plot_performance()
    except:
        pass
    logging.info("🔚 程序结束")

    # 启动连续模式或定时模式
    # 默认启动连续监控模式，除非明确指定其他模式
    if not args.backtest and not args.stats and not args.train:
        logging.info("🔄 启动连续实时分析模式（默认模式）")
        logging.info("👁️ 完整预测过程展示: 每次分析都会显示详细步骤")
        logging.info("📊 包含: 数据获取→技术指标→AI模型→置信度计算→交易决策")
        logging.info("⚡ 只在极高置信度(≥85%)时才会提示交易")
        logging.info("🚨 发现极佳机会时会连续提示音+醒目提示")
        logging.info("⏰ 每笔交易将在10分钟后自动验证结果")
        logging.info("💰 马丁格尔投注策略: 初始5U，输则翻倍(10U→30U→90U→250U)，赢则回到5U")
        logging.info("⌨️  运行中按键功能: 's' - 查看统计  'q' - 退出")
        
        # 定时任务 - 每天计算一次绩效
        schedule.every().day.at("00:00").do(predictor.calculate_performance)
        schedule.every().day.at("00:01").do(predictor.plot_performance)
        
        # 立即进行一次预测和启动连续监控
        predictor.continuous_predict_and_trade()
    elif args.continuous:
        # 兼容旧的--continuous参数
        logging.info("🔄 启动连续实时分析模式（通过--continuous参数）")
        logging.info("👁️ 完整预测过程展示: 每次分析都会显示详细步骤")
        logging.info("📊 包含: 数据获取→技术指标→AI模型→置信度计算→交易决策")
        logging.info("⚡ 只在极高置信度(≥85%)时才会提示交易")
        logging.info("🚨 发现极佳机会时会连续提示音+醒目提示")
        logging.info("⏰ 每笔交易将在10分钟后自动验证结果")
        logging.info("💰 马丁格尔投注策略: 初始5U，输则翻倍(10U→30U→90U→250U)，赢则回到5U")
        logging.info("⌨️  运行中按键功能: 's' - 查看统计  'q' - 退出")
        
        # 定时任务 - 每天计算一次绩效
        schedule.every().day.at("00:00").do(predictor.calculate_performance)
        schedule.every().day.at("00:01").do(predictor.plot_performance)
        
        # 立即进行一次预测和启动连续监控
        predictor.continuous_predict_and_trade()

if __name__ == "__main__":
    main() 