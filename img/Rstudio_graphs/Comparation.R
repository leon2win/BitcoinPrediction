# 加载必要的包
library(ggplot2)
library(dplyr)

# 创建示例数据（请替换为您的实际数据）
set.seed(123)  # 确保结果可重复
dates <- seq(as.Date("2019-01-01"), as.Date("2024-01-01"), by = "day")
n <- length(dates)
bitcoin_price <- cumsum(rnorm(n, mean = 0.001, sd = 0.02))  # 模拟比特币价格
bitcoin_price <- (bitcoin_price - min(bitcoin_price)) / (max(bitcoin_price) - min(bitcoin_price))  # 归一化
eii <- runif(n, 0, 1)  # 模拟 EII 数据
data <- data.frame(Date = dates, Bitcoin_Price = bitcoin_price, EII = eii)

# 绘制图表
p1 <- ggplot(data, aes(x = Date)) +
  geom_bar(aes(y = EII), stat = "identity", fill = "red", alpha = 0.6) +  # EII 柱状图
  geom_line(aes(y = Bitcoin_Price), color = "blue", size = 1) +  # 比特币价格线图
  labs(title = "Bitcoin Price and EII Over Time (2019-2024)",
       x = "Date",
       y = "Normalized Value") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +  # X 轴按年显示
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +  # Y 轴范围和刻度
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 10))

# 显示图表
print(p1)

# 保存图表为 PNG 文件
ggsave("Bitcoin_Price_and_EII.png", plot = p1, width = 10, height = 6, dpi = 300)