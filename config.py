"""
比特币预测系统配置文件
"""

# API配置
API_CONFIG = {
    'api_key': '',  # 请填入您的币安API密钥
    'api_secret': '',  # 请填入您的币安API密钥
}

# 交易参数
TRADING_CONFIG = {
    'symbol': 'BTCUSDT',  # 交易对
    'intervals': ['1m', '5m', '15m'],  # 使用的时间周期
    'lookback_hours': 96,  # 增加历史数据获取范围（小时）
    'prediction_minutes': 10,  # 预测未来多少分钟
    'trade_amount': 100,  # 每次交易金额（美元）
    'stop_loss': 0.02,  # 止损比例 (2%)
    'take_profit': 0.03,  # 止盈比例 (3%)
    'min_confidence_threshold': 85,  # 提高最低置信度门槛 (85%)
    'soft_confidence_floor': 15,  # 置信度软下限 (15%)
    'big_trade_threshold': 0.01,  # 大单交易阈值，单位BTC
    'bet_amount': 5,  # 固定投注金额为5u
    'payout_ratio': 0.8,  # 事件合约盈利率80%
}

# 马丁格尔策略参数
MARTINGALE_CONFIG = {
    'enabled': True,  # 是否启用马丁格尔策略
    'bet_amounts': [5, 10, 30, 90, 250],  # 马丁格尔投注金额序列
    'max_level': 4,  # 最大马丁格尔级别 (0-4)
    'reset_after_wins': 2,  # 连续获胜多少次后重置级别
}

# 模型参数
MODEL_CONFIG = {
    'hidden_size': 128,  # LSTM隐藏层大小
    'num_layers': 2,  # LSTM层数
    'dropout': 0.3,  # 增加Dropout比例
    'learning_rate': 0.001,  # 学习率
    'batch_size': 32,  # 批处理大小
    'epochs': 70,  # 增加训练轮数
    'sequence_length': 15,  # 增加序列长度
    'train_size': 0.8,  # 训练集比例
    'ensemble_models': 5,  # 集成模型数量
}

# 技术指标参数
TECHNICAL_CONFIG = {
    'rsi_oversold': 30,  # RSI超卖阈值
    'rsi_overbought': 70,  # RSI超买阈值
    'ma_short': 20,  # 短期均线
    'ma_long': 50,  # 长期均线
    'bb_period': 20,  # 布林带周期
    'bb_std': 2,  # 布林带标准差
    'macd_fast': 12,  # MACD快线
    'macd_slow': 26,  # MACD慢线
    'macd_signal': 9,  # MACD信号线
    'volume_threshold': 1.2,  # 成交量放大阈值
    'tech_score_threshold': 3,  # 提高技术指标评分阈值
    'high_confidence_override': 0.92,  # 高置信度覆盖阈值
    'trend_confirmation': True,  # 启用趋势确认
    'multi_timeframe_check': True,  # 启用多时间周期检查
}

# 情绪分析参数
SENTIMENT_CONFIG = {
    'enhanced_sentiment_enabled': True,  # 是否启用增强情绪分析
    'fear_greed_weight': 0.15,  # 增加恐慌贪婪指数权重
    'order_book_depth': 50,  # 订单簿深度
    'social_sentiment_enabled': True,  # 启用社交媒体情绪分析
}

# 交易时间过滤
TIME_FILTER_CONFIG = {
    'enabled': True,  # 启用交易时间过滤
    'avoid_high_volatility_hours': True,  # 避开高波动时段
    'avoid_news_hours': True,  # 避开重要新闻发布时段
    'preferred_trading_hours': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21],  # 优先交易时段（UTC小时）
}

# 系统参数
SYSTEM_CONFIG = {
    'log_level': 'INFO',  # 日志级别
    'save_model': True,  # 是否保存模型
    'backtest_days': 30,  # 回测天数
    'max_trades_per_day': 8,  # 每日最大交易次数
    'min_trade_interval_minutes': 60,  # 交易间隔最小分钟数
}

# 合并所有配置
DEFAULT_CONFIG = {
    **API_CONFIG,
    **TRADING_CONFIG,
    **MARTINGALE_CONFIG,
    **MODEL_CONFIG,
    **TECHNICAL_CONFIG,
    **SENTIMENT_CONFIG,
    **TIME_FILTER_CONFIG,
    **SYSTEM_CONFIG,
} 