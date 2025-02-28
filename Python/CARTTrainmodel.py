# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Step 1: Load and preprocess data
# Load data from a CSV file (assumed columns: 'tweet_text', 'timestamp', 'price_range')
data = pd.read_csv('twitter_data.csv')

# Handle missing values by dropping rows with NaN
data = data.dropna()

# Convert timestamp to datetime format for time-based features
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Step 2: Feature extraction from text data
# Use TF-IDF to extract features from tweet text
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')  # Limit to top 100 features
tfidf_features = vectorizer.fit_transform(data['tweet_text']).toarray()

# Create a DataFrame with TF-IDF features
tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

# Step 3: Feature engineering with time-based features
# Extract hour of the day from timestamp as a new feature
data['hour'] = data['timestamp'].dt.hour

# Create a feature for tweet length (number of characters in tweet)
data['tweet_length'] = data['tweet_text'].apply(len)

# Step 4: Combine all features
# Combine original data (numerical and categorical) with TF-IDF features
processed_data = pd.concat([data[['hour', 'tweet_length']], tfidf_df], axis=1)

# Step 5: Feature transformation
# Standardize numerical features (hour and tweet_length)
scaler = StandardScaler()
numerical_features = ['hour', 'tweet_length']
processed_data[numerical_features] = scaler.fit_transform(processed_data[numerical_features])

# Step 6: Feature selection
# Select top 10 features based on their relevance to the target variable (price_range)
selector = SelectKBest(score_func=f_classif, k=10)
selected_features = selector.fit_transform(processed_data, data['price_range'])

# Get the names of selected features
selected_mask = selector.get_support()
selected_feature_names = processed_data.columns[selected_mask].tolist()
print("Selected features:", selected_feature_names)

# Step 7: Feature reduction with PCA
# Apply PCA to reduce dimensionality to 5 components
pca = PCA(n_components=5)
pca_features = pca.fit_transform(selected_features)

# Create a DataFrame for PCA features
pca_df = pd.DataFrame(pca_features, columns=[f'pca_{i}' for i in range(5)])

# Step 8: Save the extracted features
# Save the processed features to a CSV file
pca_df.to_csv('extracted_features.csv', index=False)
print("Features saved to 'extracted_features.csv'")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch




# roBERTa test sample
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 示例推文
text = "I love Bitcoin so much!"

# 预处理并预测
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]

# 输出结果
labels = ['Negative', 'Neutral', 'Positive']
for label, score in zip(labels, scores):
    print(f"{label}: {score:.4f}")
