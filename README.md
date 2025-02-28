# BitcoinPrediction
This prejct denotes a new feature of predicting the Next-day Bitcoin price range. It uses the CART decision tree model with 124 general finance high-demonsional indicators wo analysis the Bitcoin price trend. Consider to external factor's impact, such as social media sentiment, 

# Tools
TA-Lib sources(124 technical-indicators): https://ta-lib.org/install/

Hugging Face twitter based roBERTa sentiment analysis(the 125th sentiment indicator): https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

Scikit-learn(Decision Tree Model): https://scikit-learn.org/stable/modules/tree.html#tree

# Data Collection
Bitcoin Historical Data: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data

Bitcoin Sentiment Analysis: https://www.kaggle.com/code/abdalrahmanshahrour/sgd-bitcoin-sentiment-analysis/input

Twitter-Roberta-Base-Sentiment-Analysis： https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

BitcoinData (Bitcoin Historical Price Data)
Load data: Load data from BitcoinData.csv.
Date format conversion: Convert the “Date” column to datetime format for time series analysis.
Handle missing values: Remove rows with missing prices, and fill missing volume rows with the mean.
Remove outliers: Filter out unreasonable prices (e.g., < 0 or > 100,000).
Normalize data: Use MinMaxScaler to scale price data to the [0,1] interval.
TwitterData (Twitter Sentiment Data)

Load data: Load data from TwitterData.csv.
Clean text data: Remove URLs, @users, punctuation, and other irrelevant characters.
Calculate sentiment scores: If sentiment scores are not present in the data, use TextBlob to calculate the sentiment polarity of each tweet (ranging from -1 to 1).
Aggregate daily sentiment: Group by date and calculate the daily average sentiment score.
Date format conversion: Convert the “Date” column to datetime format.
Normalize sentiment scores: Scale sentiment scores to the [0,1] interval.

#Feature Extraction and Fusion
Based on the TA-Lib method and Twitter roBERTa method to analysis the data, then merge the twitter score as the 125th indicator. Make sure the results keep in the same type in order to train the CART model

# CART Decision Tree Model Testing and Prediction

#Results
Results summarize and demonstrate as few charts and tables(see them in the img folder).


![comparision_table](https://github.com/user-attachments/assets/47ac9b38-a561-4090-bddc-492af6762a2f)
