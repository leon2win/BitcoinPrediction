# 加载 ggplot2 包
library(ggplot2)

# 创建数据框，假设包含年化回报率 (rG)、信息比率 (IR) 和胜率 (Win/Loss Ratio)
data <- data.frame(
  Strategy = c("AI Model", "Buy & Hold", "Cutoff 1", "Cutoff 2", "Cutoff 3"),
  Annualized_Return = c(577.9, 305.5, 116.98, 131.35, 196.86),
  Info_Ratio = c(8.99, 3.43, 1.43, 1.63, 2.47),
  Win_Loss_Ratio = c(1.71, 1.38, 1.17, 1.19, 1.28)
)

# 将数据转换为长格式，以便 ggplot 处理多个指标
library(reshape2)
data_long <- melt(data, id.vars = "Strategy")

# 画柱状图
ggplot(data_long, aes(x = Strategy, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparison of Different Strategies",
       x = "Strategy",
       y = "Value",
       fill = "Metric") +
  theme_minimal()
