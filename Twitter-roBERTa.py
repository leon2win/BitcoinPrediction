from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载模型和分词器
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
