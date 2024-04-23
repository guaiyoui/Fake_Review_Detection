import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax


from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd

# 定义情感分析函数
def analyze_sentiment(text):
    # 添加特殊标记并进行tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # 使用模型进行情感分析
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # 使用softmax获取概率分布
    probabilities = softmax(logits, dim=1)

    # 获取预测的情感
    _, predicted_class = torch.max(probabilities, 1)
    sentiment = "Positive" if predicted_class == 1 else "Negative"

    return sentiment, probabilities[0][predicted_class].item()


if __name__ == "__main__":
    # download the pre-trained model to the path
    model_id = "./sentiment_model/"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_id)

    # 测试
    text = "这部电影太棒了，我喜欢它！"
    sentiment, confidence = analyze_sentiment(text)
    print(f"文本：{text}")
    print(f"情感：{sentiment}，置信度：{confidence:.4f}")

    text = "这个产品质量很差，我不喜欢它。"
    sentiment, confidence = analyze_sentiment(text)
    print(f"文本：{text}")
    print(f"情感：{sentiment}，置信度：{confidence:.4f}")

    comment_data = pd.read_csv('comment_data.csv', encoding='gbk', sep='\t', header=None, index_col=None)
    # 消除缺失数据 NaN为缺失数据
    comment_data = comment_data.dropna().values.tolist() 
    
    new_list = []
    for sentence in comment_data:
        # 去除换行符、缩进
        sentence = str(sentence[0])
        sentence = sentence.replace('\n', '')
        sentence = sentence.replace('\t', '')
        sentence = sentence.replace(' ', '')
        new_list.append(sentence)
    
    sentiment_analysis = []
    confidence_analysis = []
    for review in new_list:
        sentiment, confidence = analyze_sentiment(review)
        sentiment_analysis.append(sentiment)
        confidence_analysis.append(confidence)
        print(f"文本：{review}", f"情感：{sentiment}，置信度：{confidence:.4f}")

    results = pd.DataFrame({'Review': new_list, 'Sentiment_analysis': sentiment_analysis, 'Confidence': confidence_analysis})

    results.to_csv("./results/sentiment_analysis_results.csv", index=False)
