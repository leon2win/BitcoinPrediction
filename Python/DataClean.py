# Import necessary libraries
import pandas as pd
import numpy as np
import re
from textblob import TextBlob  # For sentiment analysis
from sklearn.preprocessing import MinMaxScaler

# 1. Load data files
try:
    bitcoin_data = pd.read_csv('BitcoinData.csv')  # Bitcoin data
    twitter_data = pd.read_csv('TwitterData.csv')  # Twitter data
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

# 4. Save processed data
bitcoin_data_processed.to_csv('BitcoinData_Processed.csv', index=False)
twitter_data_processed.to_csv('TwitterData_Processed.csv', index=False)

print("Data preprocessing completed! Processed files have been saved as CSV format.")
