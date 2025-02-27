# 安装工具：pip install transformers pandas
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# 加载情绪分析模型
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# 读取推文数据
tweet_data = pd.read_csv('SentimentAnalysis.csv')  # 假设这是你的推文文件

# 计算每条推文的情绪得分
def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1).detach().numpy()[0]
    return probs[2] - probs[0]  # 正向概率 - 负向概率

tweet_data['sentiment'] = tweet_data['text'].apply(get_sentiment)

# 按日期算每天平均情绪
daily_sentiment = tweet_data.groupby('date')['sentiment'].mean()
daily_sentiment.to_csv('daily_sentiment.csv')  # 保存结果



# 读取市场数据和情绪数据
market_data = pd.read_csv('market_data.csv')  # 你的价格数据
market_data['date'] = pd.to_datetime(market_data['date'])

daily_sentiment = pd.read_csv('daily_sentiment.csv', names=['date', 'Sentiment_Score'])
daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

# 合并
merged_data = market_data.merge(daily_sentiment, on='date', how='left')
merged_data['Sentiment_Score'].fillna(0, inplace=True)  # 没数据的填0
