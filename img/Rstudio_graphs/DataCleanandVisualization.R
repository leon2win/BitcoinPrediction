# 加载必要的R包
library(readr)    # 读取CSV文件
library(dplyr)    # 数据处理
library(tidyr)    # 数据整形
library(ggplot2)  # 数据可视化
library(lubridate) # 日期处理

# 1. 加载原始数据
bitcoin_data <- read_csv("BitcoinData.csv")  # 加载比特币价格数据
twitter_data <- read_csv("TwitterData.csv")  # 加载Twitter数据

# 2. 预处理BitcoinData
# 转换日期格式
bitcoin_data$Date <- ymd(bitcoin_data$Date)

# 处理缺失值：删除价格缺失的行，填充交易量缺失值
bitcoin_data <- bitcoin_data %>%
  drop_na(Price) %>%
  mutate(Volume = replace_na(Volume, mean(Volume, na.rm = TRUE)))

# 移除异常值（假设价格范围为0到100,000）
bitcoin_data <- bitcoin_data %>%
  filter(Price > 0 & Price < 100000)

# 计算每日价格变化（用于后续分析）
bitcoin_data <- bitcoin_data %>%
  arrange(Date) %>%
  mutate(Price_Change = (Price - lag(Price)) / lag(Price) * 100)  # 百分比变化

# 3. 预处理TwitterData
# 转换日期格式
twitter_data$Date <- ymd(twitter_data$Date)

# 清理推文文本（假设推文列名为'Tweet'）
clean_tweet <- function(text) {
  if (is.na(text)) return("")
  # 移除URL、@用户、#话题和特殊字符
  text <- gsub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)
  text <- gsub("@\\w+", "", text)
  text <- gsub("#\\w+", "", text)
  text <- gsub("[^[:alnum:][:space:]]", "", text)
  return(trimws(text))
}

# 应用文本清理
twitter_data <- twitter_data %>%
  mutate(Tweet = clean_tweet(Tweet)) %>%
  drop_na(Tweet)

# 聚合每日Twitter情绪（假设需要简单统计推文数量或情感得分）
# 如果没有情感得分，这里计算推文数量作为代理
daily_tweets <- twitter_data %>%
  group_by(Date) %>%
  summarise(Tweet_Count = n()) %>%
  ungroup()

# 4. 合并Bitcoin和Twitter数据
combined_data <- bitcoin_data %>%
  left_join(daily_tweets, by = "Date")

# 5. 数据可视化 - 比特币价格与Twitter数据的关系
# 比特币价格随时间变化图（线图）
p1 <- ggplot(combined_data, aes(x = Date, y = Price)) +
  geom_line(color = "blue", size = 1) +
  labs(title = "Bitcoin Price Over Time (2019-2024)",
       x = "Date",
       y = "Price (USD)") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  theme_minimal()

# 保存比特币价格图
ggsave("Bitcoin_Price_Over_Time.png", plot = p1, width = 10, height = 6, dpi = 300)

# Twitter推文数量随时间变化图（柱状图）
p2 <- ggplot(daily_tweets, aes(x = Date, y = Tweet_Count)) +
  geom_bar(stat = "identity", fill = "red", alpha = 0.6) +
  labs(title = "Daily Twitter Activity Related to Bitcoin (2019-2024)",
       x = "Date",
       y = "Tweet Count") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  theme_minimal()

# 保存Twitter推文数量图
ggsave("Twitter_Activity_Over_Time.png", plot = p2, width = 10, height = 6, dpi = 300)

# 6. 合并数据后可视化 - 比特币价格和Twitter推文数量的关系
p3 <- ggplot(combined_data, aes(x = Tweet_Count, y = Price)) +
  geom_point(color = "purple", alpha = 0.5) +
  geom_smooth(method = "lm", color = "black") +
  labs(title = "Relationship Between Bitcoin Price and Twitter Activity",
       x = "Daily Tweet Count",
       y = "Bitcoin Price (USD)") +
  theme_minimal()

# 保存关系图
ggsave("Bitcoin_Price_vs_Twitter.png", plot = p3, width = 10, height = 6, dpi = 300)

# 7. 保存预处理后的数据
write_csv(bitcoin_data, "BitcoinData_Processed.csv")
write_csv(twitter_data, "TwitterData_Processed.csv")
write_csv(combined_data, "CombinedData_Processed.csv")

print("Data preprocessing and visualization completed! Files saved as CSV and PNG.")
