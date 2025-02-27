import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 读取数据
data = pd.read_csv('OHLC.csv')

# 计算简单特征（例如5天和10天移动平均）
data['SMA5'] = data['Close'].rolling(5).mean()
data['SMA10'] = data['Close'].rolling(10).mean()

# 计算明天的回报率
data['Return'] = (data['Close'].shift(-1) - data['Close']) / data['Close']

# 定义标签（简化版，涨=1，跌=0）
data['Label'] = data['Return'].apply(lambda x: 1 if x > 0 else 0)

# 删除空值
data = data.dropna()

# 准备特征（X）和标签（y）
X = data[['SMA5', 'SMA10']]  # 示例特征
y = data['Label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练决策树模型
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
