import numpy as np
import pandas as pd
import logging
import ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class DataProcessor:
    def prepare_enhanced_features(self, df):
        """准备增强特征集，包含更多技术指标和市场情绪特征"""
        try:
            # 确保数据类型正确
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            # 基本价格特征
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # 价格波动特征
            df['price_range'] = df['high'] - df['low']
            df['price_range_pct'] = df['price_range'] / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            
            # 🆕 高级价格特征
            df['hl_ratio'] = df['high'] / df['low']  # 高低比
            df['co_ratio'] = df['close'] / df['open']  # 收开比
            
            # 移动平均线
            for window in [5, 10, 20, 50, 100]:
                df[f'sma_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
                df[f'ema_{window}'] = ta.trend.ema_indicator(df['close'], window=window)
                
                # 🆕 相对于均线的位置
                df[f'close_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
                df[f'close_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
                
                # 🆕 均线交叉信号
                if window > 5:
                    df[f'sma_5_{window}_cross'] = np.where(
                        (df[f'sma_5'].shift(1) < df[f'sma_{window}'].shift(1)) & 
                        (df[f'sma_5'] > df[f'sma_{window}']), 
                        1, np.where(
                            (df[f'sma_5'].shift(1) > df[f'sma_{window}'].shift(1)) & 
                            (df[f'sma_5'] < df[f'sma_{window}']), 
                            -1, 0
                        )
                    )
            
            # 动量指标
            df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
            df['rsi_21'] = ta.momentum.rsi(df['close'], window=21)
            
            # 🆕 RSI变化率
            df['rsi_14_change'] = df['rsi_14'] - df['rsi_14'].shift(1)
            df['rsi_14_slope'] = df['rsi_14'].diff(3) / 3  # 3周期RSI斜率
            
            # 🆕 RSI超买超卖区域指示
            df['rsi_overbought'] = np.where(df['rsi_14'] > 70, 1, 0)
            df['rsi_oversold'] = np.where(df['rsi_14'] < 30, 1, 0)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # 🆕 MACD交叉信号
            df['macd_cross'] = np.where(
                (df['macd'].shift(1) < df['macd_signal'].shift(1)) & 
                (df['macd'] > df['macd_signal']), 
                1, np.where(
                    (df['macd'].shift(1) > df['macd_signal'].shift(1)) & 
                    (df['macd'] < df['macd_signal']), 
                    -1, 0
                )
            )
            
            # 布林带
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # 🆕 布林带位置指标
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 🆕 布林带突破信号
            df['bb_upper_break'] = np.where(df['close'] > df['bb_upper'], 1, 0)
            df['bb_lower_break'] = np.where(df['close'] < df['bb_lower'], 1, 0)
            
            # 🆕 Keltner通道
            keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
            df['kc_upper'] = keltner.keltner_channel_hband()
            df['kc_middle'] = keltner.keltner_channel_mband()
            df['kc_lower'] = keltner.keltner_channel_lband()
            
            # 🆕 Squeeze Momentum指标 (布林带与Keltner通道的关系)
            df['squeeze'] = np.where(
                (df['bb_upper'] < df['kc_upper']) & 
                (df['bb_lower'] > df['kc_lower']), 
                1, 0
            )
            
            # 成交量指标
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # 🆕 高成交量突破信号
            df['volume_breakout'] = np.where(
                (df['volume'] > df['volume_sma_20'] * 2) & 
                (abs(df['returns']) > df['returns'].rolling(20).std()), 
                np.sign(df['returns']), 
                0
            )
            
            # 🆕 价格波动性指标
            df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            df['atr_percent'] = df['atr_14'] / df['close'] * 100
            
            # 🆕 波动率变化
            df['volatility_change'] = df['atr_14'] / df['atr_14'].shift(5) - 1
            
            # 🆕 趋势强度指标 - ADX
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
            
            # 🆕 趋势强度分类
            df['trend_strength'] = np.where(
                df['adx'] < 20, 0,  # 无趋势
                np.where(
                    df['adx'] < 40, 1,  # 弱趋势
                    np.where(
                        df['adx'] < 60, 2,  # 强趋势
                        3  # 极强趋势
                    )
                )
            )
            
            # 🆕 趋势方向
            df['trend_direction'] = np.where(
                df['adx_pos'] > df['adx_neg'], 1,  # 上升趋势
                -1  # 下降趋势
            )
            
            # 🆕 Ichimoku云图指标
            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            
            # 🆕 云图信号
            df['cloud_green'] = np.where(df['ichimoku_a'] > df['ichimoku_b'], 1, 0)  # 绿云
            df['above_cloud'] = np.where(
                df['close'] > df['ichimoku_a'].shift(26), 
                np.where(df['close'] > df['ichimoku_b'].shift(26), 1, 0), 
                0
            )
            df['below_cloud'] = np.where(
                df['close'] < df['ichimoku_a'].shift(26), 
                np.where(df['close'] < df['ichimoku_b'].shift(26), 1, 0), 
                0
            )
            
            # 🆕 Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # 🆕 Stochastic交叉信号
            df['stoch_cross'] = np.where(
                (df['stoch_k'].shift(1) < df['stoch_d'].shift(1)) & 
                (df['stoch_k'] > df['stoch_d']), 
                1, np.where(
                    (df['stoch_k'].shift(1) > df['stoch_d'].shift(1)) & 
                    (df['stoch_k'] < df['stoch_d']), 
                    -1, 0
                )
            )
            
            # 🆕 CCI (Commodity Channel Index)
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
            
            # 🆕 OBV (On-Balance Volume)
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['obv_sma'] = df['obv'].rolling(20).mean()
            df['obv_ratio'] = df['obv'] / df['obv_sma']
            
            # 🆕 Chaikin Money Flow
            df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
            
            # 🆕 Money Flow Index
            df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
            
            # 🆕 Williams %R
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
            
            # 🆕 Parabolic SAR
            df['psar'] = ta.trend.psar_up(df['high'], df['low'], df['close'])
            df['psar_down'] = ta.trend.psar_down(df['high'], df['low'], df['close'])
            
            # 🆕 SAR信号
            df['psar_signal'] = np.where(
                pd.notna(df['psar']), 1,  # 上升趋势
                np.where(pd.notna(df['psar_down']), -1, 0)  # 下降趋势或无信号
            )
            
            # 🆕 市场情绪特征 (如果有外部数据)
            if hasattr(self, 'sentiment_analyzer') and self.sentiment_analyzer:
                try:
                    # 恐慌贪婪指数
                    fear_greed = self.sentiment_analyzer.get_fear_greed_index()
                    if fear_greed is not None:
                        df['fear_greed'] = fear_greed
                        
                        # 情绪分类
                        df['extreme_fear'] = np.where(df['fear_greed'] < 20, 1, 0)
                        df['fear'] = np.where((df['fear_greed'] >= 20) & (df['fear_greed'] < 40), 1, 0)
                        df['neutral'] = np.where((df['fear_greed'] >= 40) & (df['fear_greed'] < 60), 1, 0)
                        df['greed'] = np.where((df['fear_greed'] >= 60) & (df['fear_greed'] < 80), 1, 0)
                        df['extreme_greed'] = np.where(df['fear_greed'] >= 80, 1, 0)
                    
                    # 社交媒体情绪
                    social_sentiment = self.sentiment_analyzer.get_social_sentiment()
                    if social_sentiment is not None:
                        df['social_sentiment'] = social_sentiment
                except Exception as e:
                    pass
            
            # 🆕 大单交易特征
            if hasattr(self, 'large_trade_detector') and self.large_trade_detector:
                try:
                    # 大单买入/卖出压力
                    large_buy_pressure, large_sell_pressure = self.large_trade_detector.get_large_trade_pressure()
                    if large_buy_pressure is not None and large_sell_pressure is not None:
                        df['large_buy_pressure'] = large_buy_pressure
                        df['large_sell_pressure'] = large_sell_pressure
                        df['large_trade_ratio'] = large_buy_pressure / (large_sell_pressure + 0.0001)
                except Exception as e:
                    pass
            
            # 🆕 订单簿特征
            if hasattr(self, 'order_book_analyzer') and self.order_book_analyzer:
                try:
                    # 买卖压力比
                    bid_ask_ratio = self.order_book_analyzer.get_bid_ask_ratio()
                    if bid_ask_ratio is not None:
                        df['bid_ask_ratio'] = bid_ask_ratio
                        
                    # 订单簿深度
                    book_depth = self.order_book_analyzer.get_order_book_depth()
                    if book_depth is not None:
                        df['order_book_depth'] = book_depth
                        
                    # 价格墙
                    support_level, resistance_level = self.order_book_analyzer.get_price_walls()
                    if support_level is not None and resistance_level is not None:
                        df['support_distance'] = (df['close'] - support_level) / df['close']
                        df['resistance_distance'] = (resistance_level - df['close']) / df['close']
                except Exception as e:
                    pass
            
            # 🆕 时间特征
            df['hour'] = pd.to_datetime(df.index).hour
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            
            # 🆕 小时交易活跃度
            active_hours = [0, 8, 12, 16, 20]  # UTC时间，对应主要交易时段
            df['active_hour'] = df['hour'].apply(lambda x: 1 if x in active_hours else 0)
            
            # 🆕 特征组合
            # RSI与成交量的结合
            df['rsi_volume'] = df['rsi_14'] * df['volume_ratio']
            
            # 价格动量与成交量的结合
            df['price_volume_trend'] = df['returns'] * df['volume_ratio']
            
            # 趋势与波动性的结合
            df['trend_volatility'] = df['trend_direction'] * df['atr_percent']
            
            # 填充缺失值
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill')
            df = df.fillna(0)  # 剩余的NaN填充为0
            
            return df
            
        except Exception as e:
            logging.error(f"特征工程失败: {e}")
            return df 