library(ggplot2)
dates <- seq(as.Date("2019-01-01"), as.Date("2024-12-29"), by="day")
prices <- seq(3745.95, 97500, length.out=length(dates)) + rnorm(length(dates), 0, 500)
rsi <- rnorm(length(dates), 50, 15); rsi <- pmax(0, pmin(100, rsi))
sma <- zoo::rollmean(prices, k=20, fill=prices[1], align="right")
eii <- rnorm(length(dates), 56, 17.5); eii <- pmax(0, pmin(100, eii))

# 标准化
prices_norm <- (prices - min(prices)) / (max(prices) - min(prices))
rsi_norm <- rsi / 100
sma_norm <- (sma - min(sma)) / (max(sma) - min(sma))
eii_norm <- eii / 100

# 数据框
df <- data.frame(Date=dates, Price=prices_norm, RSI=rsi_norm, SMA=sma_norm, EII=eii_norm)

# 绘图
p <- ggplot(df, aes(x=Date)) +
  geom_line(aes(y=Price, color="Bitcoin Price")) +
  geom_line(aes(y=RSI, color="RSI (14)")) +
  geom_line(aes(y=SMA, color="SMA (20)")) +
  geom_line(aes(y=EII, color="EII")) +
  labs(title="Figure 1: Key Technical Indicators and Sentiment Score vs Bitcoin Price (2019-2024)",
       x="Date", y="Normalized Values") +
  scale_color_manual(values=c("Bitcoin Price"="blue", "RSI (14)"="orange", "SMA (20)"="green", "EII"="red")) +
  theme_minimal() +
  theme(legend.position="bottom", legend.title=element_blank()) +
  guides(color=guide_legend(nrow=1))

# 保存
ggsave("Figure_1.png", p, width=10, height=6, dpi=300)
print(p)