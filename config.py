def get_default_config():
    """获取默认配置"""
    return {
        # 基础配置
        'symbol': 'BTCUSDT',
        'timeframe': '1m',
        'api_key': '',
        'api_secret': '',
        'test_mode': True,
        'log_level': 'INFO',
        
        # 模型配置
        'sequence_length': 60,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'train_size': 0.8,
        
        # 预测和交易配置
        'min_confidence_threshold': 75,  # 提高置信度阈值，从60%提高到75%
        'soft_confidence_floor': 15,
        'check_interval': 60,  # 秒
        'trade_enabled': False,
        'trade_amount': 5,  # USDT
        
        # 风控配置
        'max_daily_trades': 10,
        'max_daily_loss': 100,  # USDT
        'stop_loss_pct': 1.5,  # 百分比
        'take_profit_pct': 1.0,  # 百分比
        
        # 马丁格尔策略配置
        'martingale_enabled': True,
        'martingale_base_amount': 5,  # 基础投注额 (USDT)
        'martingale_levels': [1, 2, 6, 18, 50],  # 投注倍数序列
        
        # 大单检测配置
        'large_order_threshold': 0.5,  # BTC
        'large_order_influence_time': 60,  # 秒
        'auto_threshold_adjustment': True,  # 自动调整大单阈值
        
        # 技术指标过滤配置
        'tech_filter_enabled': True,  # 启用技术指标过滤
        'rsi_overbought': 70,  # RSI超买阈值
        'rsi_oversold': 30,  # RSI超卖阈值
        'volume_threshold': 0.7,  # 成交量阈值（相对于20日均值）
        
        # 高级配置
        'feature_engineering': {
            'use_technical_indicators': True,
            'use_sentiment_analysis': True,
            'use_market_data': True,
            'use_order_book': True,
            'use_large_trades': True,
        },
        
        # 回测配置
        'backtest': {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_balance': 1000,
            'fee_rate': 0.001,
        }
    } 