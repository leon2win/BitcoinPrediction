import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 读取数据（假设有124个技术指标）
data = pd.read_csv('bitcoin_data.csv')  # 包含日期、124个指标和回报区间
X = data.drop(['date', 'return_range'], axis=1)  # 124个特征
y = data['return_range']  # 目标（回报区间）

# 分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练CART模型
model = DecisionTreeClassifier(criterion='entropy', max_features=11)
model.fit(X_train, y_train)

# 测试准确率
accuracy = model.score(X_test, y_test)
print(f"准确率: {accuracy}")



# 步骤1：加载Twitter-roBERTa模型
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 步骤2：定义情绪分析函数
def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    return scores[2] - scores[0]  # 正向 - 负向，得分范围[-1, 1]

# 步骤3：读取Twitter数据并计算每日情绪得分
tweet_data = pd.read_csv('Bitcoin_tweets.csv')  # 从Kaggle下载的推文数据
tweet_data['date'] = pd.to_datetime(tweet_data['date']).dt.date  # 转换为日期格式
tweet_data['sentiment'] = tweet_data['text'].apply(get_sentiment)  # 计算每条推文情绪
daily_sentiment = tweet_data.groupby('date')['sentiment'].mean().reset_index()  # 每日平均情绪

# 步骤4：读取原始高维参数数据
market_data = pd.read_csv('bitcoin_data.csv')  # 假设包含124个指标
market_data['date'] = pd.to_datetime(market_data['date']).dt.date

# 步骤5：合并情绪得分作为第125个指标
merged_data = market_data.merge(daily_sentiment, on='date', how='left')
merged_data['sentiment'].fillna(0, inplace=True)  # 缺失值填0

# 步骤6：准备特征和目标
X = merged_data.drop(['date', 'return_range'], axis=1)  # 现在有125个特征
y = merged_data['return_range']

# 步骤7：分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤8：训练CART模型
model = DecisionTreeClassifier(criterion='entropy', max_features=11)
model.fit(X_train, y_train)

# 步骤9：测试准确率
accuracy = model.score(X_test, y_test)
print(f"加入情绪分析后的准确率: {accuracy}")
