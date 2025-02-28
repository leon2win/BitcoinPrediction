import pandas as pd
import talib
import numpy as np

# 加载比特币价格数据
data = pd.read_csv('BitcoinData.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# 计算技术指标（这里展示部分指标，实际需扩展到124个）
# 1. 简单移动平均线 (SMA)
data['SMA_5'] = talib.SMA(data['Close'], timeperiod=5)
data['SMA_10'] = talib.SMA(data['Close'], timeperiod=10)
data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)

# 2. 指数移动平均线 (EMA)
data['EMA_12'] = talib.EMA(data['Close'], timeperiod=12)
data['EMA_26'] = talib.EMA(data['Close'], timeperiod=26)

# 3. 相对强弱指数 (RSI)
data['RSI_14'] = talib.RSI(data['Close'], timeperiod=14)

# 4. 布林带 (Bollinger Bands)
upper, middle, lower = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
data['BB_upper'] = upper
data['BB_middle'] = middle
data['BB_lower'] = lower

# 5. MACD (移动平均线收敛-发散指标)
macd, macdsignal, macdhist = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
data['MACD'] = macd
data['MACD_signal'] = macdsignal
data['MACD_hist'] = macdhist

# 6. 动量 (Momentum)
data['Momentum_10'] = talib.MOM(data['Close'], timeperiod=10)

# 7. 随机指标 (Stochastic Oscillator)
slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'],
                           fastk_period=14, slowk_period=3, slowk_matype=0,
                           slowd_period=3, slowd_matype=0)
data['Stoch_K'] = slowk
data['Stoch_D'] = slowd

# 8. 平均真实波动幅度 (ATR)
data['ATR_14'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

# 9. 抛物线转向 (Parabolic SAR)
data['SAR'] = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)

# 10. K线形态识别 (Doji)
data['Doji'] = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])

# 注意：这里仅列出10个指标。您需要根据论文扩展到124个指标，例如添加更多周期的SMA、EMA，或其他TA-Lib支持的指标。

# 移除NaN值（技术指标计算会引入NaN）
data = data.dropna()

# 准备特征和目标变量
# 特征（这里仅列出示例中的指标，实际需包含所有124个）
features = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI_14',
            'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'MACD_signal',
            'MACD_hist', 'Momentum_10', 'Stoch_K', 'Stoch_D', 'ATR_14',
            'SAR', 'Doji']
X = data[features]

# 定义目标变量：次日价格波动范围（21个区间）
data['Price_Change'] = data['Close'].pct_change().shift(-1)  # 次日价格变化百分比
data['Target'] = pd.qcut(data['Price_Change'], 21, labels=range(-10, 11))  # 分成21个区间
data = data.dropna()  # 移除最后一行（无次日数据）

# 更新训练数据
X = data[features]
y = data['Target']

# 输出数据预览
print("特征数据预览：")
print(X.head())
print("\n目标变量预览：")
print(y.head())

# 接下来可以训练CART决策树模型（示例未包含训练部分）
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# model.fit(X, y)
