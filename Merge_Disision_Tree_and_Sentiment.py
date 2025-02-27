
# import the price data of bitcoin and analyze the 124 tech indicators
import pandas as pd
data = pd.read_csv('OHLC.csv')  # 读取数据
data['SMA5'] = talib.SMA(data['Close'], timeperiod=5)  # 计算5天简单移动平均
data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)  # 计算MACD

import pandas as pd
import numpy as np
from transformers import pipeline
from talib import RSI, SMA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. 加载比特币价格数据
btc_data = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')  # 从Kaggle下载
btc_data['Timestamp'] = pd.to_datetime(btc_data['Timestamp'], unit='s')
btc_data.set_index('Timestamp', inplace=True)
btc_daily = btc_data.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume_(BTC)': 'sum'})

# 2. 计算技术指标（示例：RSI和SMA）
btc_daily['RSI_14'] = RSI(btc_daily['Close'], timeperiod=14)
btc_daily['SMA_20'] = SMA(btc_daily['Close'], timeperiod=20)
# 注意：实际需计算124个技术指标，这里仅展示2个，需扩展其他指标

# 3. 模拟Twitter情绪数据（假设已处理）
# 这里假设你已使用Twitter-roBERTa模型生成了情绪得分CSV
# 示例格式：Date, Sentiment_Score
sentiment_data = pd.DataFrame({
    'Date': btc_daily.index,
    'Sentiment_Score': np.random.uniform(0, 1, len(btc_daily))  # 模拟数据，需替换为真实数据
})
sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])
sentiment_data.set_index('Date', inplace=True)

# 4. 合并数据
features = btc_daily[['Close', 'RSI_14', 'SMA_20']].join(sentiment_data['Sentiment_Score'], how='inner')

# 5. 标准化特征
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
features_df = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

# 6. 保存数据集
features_df.to_csv('btc_features_125.csv')

# 7. 可视化图1
plt.figure(figsize=(12, 6))
plt.plot(features_df.index, features_df['Close'], label='Bitcoin Close Price')
plt.plot(features_df.index, features_df['RSI_14'], label='RSI (14)')
plt.plot(features_df.index, features_df['SMA_20'], label='SMA (20)')
plt.plot(features_df.index, features_df['Sentiment_Score'], label='Twitter Sentiment Score')
plt.xlabel('Date')
plt.ylabel('Normalized Values')
plt.title('Bitcoin Price, Technical Indicators, and Twitter Sentiment (2019-2021)')
plt.legend()
plt.show()
