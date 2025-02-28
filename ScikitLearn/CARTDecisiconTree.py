import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 加载预处理后的数据
data = pd.read_csv('processed_data.csv')
data['Date'] = pd.to_datetime(data['Date'])

# 选择特征和目标变量
# 假设特征包括归一化的价格和情绪得分，目标是价格范围
features = ['Price_Normalized', 'Sentiment_Normalized']
X = data[features]
y = data['price_range']  # 目标变量，假设为 -10 至 10 的 21 个区间

# 处理缺失值
X = X.dropna()
y = y.dropna()

# 确保 X 和 y 长度一致
X = X.loc[y.index]

# 数据分割（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化CART模型
model = DecisionTreeClassifier(random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

# 10折交叉验证（验证模型稳定性）
cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print(f"10折交叉验证得分: {cv_scores.mean():.2f} (±{cv_scores.std() * 2:.2f})")

# 预测新数据（示例）
new_data = pd.DataFrame({
    'Price_Normalized': [0.5],
    'Sentiment_Normalized': [0.7]
})
new_prediction = model.predict(new_data)
print(f"对新数据的预测结果: {new_prediction[0]}")

# 输出特征重要性（仅限分类任务）
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})
print("\n特征重要性：")
print(feature_importance)
