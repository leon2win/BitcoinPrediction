# 加载必要的包
library(ggplot2)
library(dplyr)
library(tidyr)

# 创建示例数据（请替换为您的实际数据）
return_intervals <- -10:10
baseline_accuracy <- c(0.65, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 
                       0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.65, 0.65)  # 基础模型准确率
enhanced_accuracy <- c(0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 
                       0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.75)  # 增强模型准确率
data <- data.frame(Return_Interval = return_intervals,
                   Baseline_Accuracy = baseline_accuracy,
                   Enhanced_Accuracy = enhanced_accuracy)

# 将数据转换为长格式
data_long <- data %>%
  pivot_longer(cols = c(Baseline_Accuracy, Enhanced_Accuracy),
               names_to = "Model",
               values_to = "Accuracy")

# 绘制柱状图
p2 <- ggplot(data_long, aes(x = factor(Return_Interval), y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  labs(title = "Performance Comparison of Models by Return Interval",
       x = "Return Interval",
       y = "Accuracy") +
  scale_fill_manual(values = c("Baseline_Accuracy" = "lightblue", "Enhanced_Accuracy" = "coral"),
                    labels = c("Baseline (124 Features)", "Enhanced (125 Features)")) +
  scale_y_continuous(limits = c(0, 0.8), breaks = seq(0, 0.8, 0.2)) +
  theme_minimal() +
  theme(legend.position = "top",
        legend.title = element_blank())

# 显示图表
print(p2)

# 保存图表为 PNG 文件
ggsave("Model_Accuracy_by_Return_Interval.png", plot = p2, width = 10, height = 6, dpi = 300)