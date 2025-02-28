library(DiagrammeR)
# 绘制流程图
grViz("
digraph G {
  graph [layout = dot, rankdir = LR]  # 上下布局
  node [shape = box, style = filled, fillcolor = lightblue]  # 节点为矩形，蓝色填充
  edge [arrowhead = vee]  # 箭头样式

  # 节点（步骤）
  Start [shape = ellipse]  # 开始用椭圆，绿色填充
  End [shape = ellipse, fillcolor = lightgreen]  # 结束用椭圆，绿色填充
  DataCollection [label = 'Data Collection\n(Bitcoin Price & Twitter Data)']
  DataPreprocessing [label = 'Data Preprocessing\n(Clean and format data)']
  FeatureExtraction [label = 'Feature Extraction\n(124 Indicators & Sentiment)']
  CARTModel [label = 'CART Decision Tree Model\n(Predict with 124 indicators)']
  SentimentAnalysis [label = 'Sentiment Analysis\n(Generate sentiment scores)']
  FeatureFusion [label = 'Feature Fusion\n(Merge 125 features)']
  PricePrediction [label = 'Price Range Prediction\n(Predict Bitcoin range)']

  # 连接（箭头）
  Start -> DataCollection
  DataCollection -> DataPreprocessing
  DataPreprocessing -> FeatureExtraction
  FeatureExtraction -> CARTModel
  FeatureExtraction -> SentimentAnalysis
  SentimentAnalysis -> FeatureFusion
  CARTModel -> FeatureFusion
  FeatureFusion -> PricePrediction
  PricePrediction -> End
}
")