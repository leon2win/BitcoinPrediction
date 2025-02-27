import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 生成模拟数据（真实情况应使用比特币技术指标数据）
np.random.seed(42)
X = np.random.rand(2000, 124)  # 2000 行数据，124 个技术指标
y = np.random.randint(-10, 11, 2000)  # 目标变量为 21 个区间（-10 到 10）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 1000 棵决策树
n_trees = 1000
predictions = np.zeros((len(y_test), n_trees))  # 用于存储每棵树的预测结果

for i in range(n_trees):
    tree = DecisionTreeClassifier(criterion="entropy", max_features="sqrt", random_state=i)
    tree.fit(X_train, y_train)
    predictions[:, i] = tree.predict(X_test)  # 记录每棵树的预测值

# 计算 1000 棵树的平均预测结果
final_predictions = np.round(predictions.mean(axis=1))

# 输出前 10 个测试样本的预测值
print("前 10 个测试样本的预测值:", final_predictions[:10])
