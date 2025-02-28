# Import necessary libraries
import pandas as pd
import numpy as np
import re
from textblob import TextBlob  # For sentiment analysis
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 1. Load and preprocess Bitcoin and Twitter data files
try:
    bitcoin_data = pd.read_csv('BitcoinData.csv')  # Bitcoin historical price data
    twitter_data = pd.read_csv('TwitterData.csv')  # Twitter sentiment data
except FileNotFoundError:
    print("Data files not found, please check the file names and paths!")
    exit()

# 2. Preprocess BitcoinData
def preprocess_bitcoin_data(df):
    # Convert date format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Handle missing values
    df = df.dropna(subset=['Price'])  # Drop rows with missing prices
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(df['Volume'].mean())  # Fill missing volume with mean

    # Remove outliers (assume price range is 0 to 100,000, adjust as needed)
    df = df[(df['Price'] > 0) & (df['Price'] < 100000)]

    # Normalize price data
    scaler = MinMaxScaler()
    df['Price_Normalized'] = scaler.fit_transform(df[['Price']])

    # Sort by date
    df = df.sort_values('Date')

    # Reset index
    df = df.reset_index(drop=True)
    return df

bitcoin_data_processed = preprocess_bitcoin_data(bitcoin_data)

# 3. Preprocess TwitterData
def preprocess_twitter_data(df):
    # Convert date format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Clean tweet text
    def clean_tweet(text):
        if pd.isna(text):  # Handle null values
            return ""
        # Remove URLs, @mentions, hashtags, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
        text = re.sub(r'@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    # Apply text cleaning
    df['Tweet'] = df['Tweet'].apply(clean_tweet)

    # Drop rows with missing tweets
    df = df.dropna(subset=['Tweet'])

    # Calculate sentiment score if not present
    if 'Sentiment_Score' not in df.columns:
        def get_sentiment(text):
            if pd.isna(text) or text == "":
                return 0
            return TextBlob(text).sentiment.polarity  # Return sentiment polarity from -1 to 1
        df['Sentiment_Score'] = df['Tweet'].apply(get_sentiment)

    # Aggregate sentiment scores by date (daily mean)
    daily_sentiment = df.groupby('Date')['Sentiment_Score'].mean().reset_index()

    # Normalize sentiment scores
    scaler = MinMaxScaler()
    daily_sentiment['Sentiment_Normalized'] = scaler.fit_transform(daily_sentiment[['Sentiment_Score']])

    return daily_sentiment

twitter_data_processed = preprocess_twitter_data(twitter_data)

# 4. Feature extraction from Twitter and Bitcoin data
# Combine Bitcoin and Twitter data by date for feature extraction
combined_data = pd.merge(bitcoin_data_processed, twitter_data_processed, on='Date', how='inner')

# Use TF-IDF to extract features from tweet text (if tweet text is still needed)
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')  # Limit to top 100 features
tfidf_features = vectorizer.fit_transform(twitter_data['Tweet'].dropna()).toarray()

# Create a DataFrame with TF-IDF features
tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

# Feature engineering with time-based and numerical features
combined_data['hour'] = combined_data['Date'].dt.hour  # Extract hour from date
combined_data['tweet_length'] = twitter_data['Tweet'].dropna().apply(len)  # Tweet length as feature

# Combine all features
processed_features = pd.concat([combined_data[['Price_Normalized', 'Sentiment_Normalized', 'hour', 'tweet_length']], tfidf_df], axis=1)

# Feature transformation
scaler = StandardScaler()
numerical_features = ['Price_Normalized', 'Sentiment_Normalized', 'hour', 'tweet_length']
processed_features[numerical_features] = scaler.fit_transform(processed_features[numerical_features])

# Feature selection
# Select top 10 features based on their relevance to a target variable (e.g., price range)
# Assuming 'price_range' is a target variable (you need to define or calculate it)
if 'price_range' not in combined_data.columns:
    # Example: Create a dummy price range based on price changes (you can adjust this logic)
    combined_data['price_change'] = combined_data['Price_Normalized'].pct_change().shift(-1) * 100
    combined_data['price_range'] = pd.qcut(combined_data['price_change'], 21, labels=range(-10, 11))
y = combined_data['price_range'].dropna()  # Target variable
X = processed_features.dropna()  # Ensure alignment with y

selector = SelectKBest(score_func=f_classif, k=10)
selected_features = selector.fit_transform(X, y)

# Get the names of selected features
selected_mask = selector.get_support()
selected_feature_names = X.columns[selected_mask].tolist()
print("Selected features:", selected_feature_names)

# Feature reduction with PCA
pca = PCA(n_components=5)
pca_features = pca.fit_transform(selected_features)

# Create a DataFrame for PCA features
pca_df = pd.DataFrame(pca_features, columns=[f'pca_{i}' for i in range(5)])

# 5. CART Decision Tree Modeling and Prediction
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(pca_df, y, test_size=0.2, random_state=42)

# Initialize CART model (classification decision tree)
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Prediction on new data
# Example: Predict for a new sample (replace with actual new data)
new_data = [[0.1, 0.2, 0.3, 0.4, 0.5]]  # Example PCA-transformed data
prediction = model.predict(new_data)
print(f"Prediction for new data: {prediction[0]}")

# Save processed features and predictions
pca_df.to_csv('extracted_features_final.csv', index=False)
print("Features and predictions saved to 'extracted_features_final.csv'")





np.random.seed(42)
X = np.random.rand(2000, 124)  # 2000 行数据，124 个技术指标
y = np.random.randint(-10, 11, 2000)  # 目标变量为 21 个区间（-10 到 10）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 5000 棵决策树，从1000修改为5000 in order to improve the accuracy
n_trees = 5000
predictions = np.zeros((len(y_test), n_trees))  # 用于存储每棵树的预测结果

for i in range(n_trees):
    tree = DecisionTreeClassifier(criterion="entropy", max_features="sqrt", random_state=i)
    tree.fit(X_train, y_train)
    predictions[:, i] = tree.predict(X_test)

# 计算 5000 棵树的平均预测结果 get the avg results of these 5000 trees
final_predictions = np.round(predictions.mean(axis=1))

# 输出前 10 个测试样本的预测值
print("test the results in 10 samples:", final_predictions[:10])
