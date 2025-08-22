import os
from transformers import pipeline
from langchain.tools import Tool
from collections import Counter

# اختيار الموديل حسب اللغة (بناءً على ENV variable)
ARABIC_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
ENGLISH_MODEL = "siebert/sentiment-roberta-large-english"
MODEL_NAME = ARABIC_MODEL if os.getenv(
    "LANGUAGE", "arabic") == "arabic" else ENGLISH_MODEL


def load_pipeline():
    print(f"🌟 Using pre-trained model: {MODEL_NAME}")
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        return_all_scores=False
    )


sentiment_pipeline = load_pipeline()


def sentiment_tool_func(text, max_length=512):
    words = text.split()
    chunks = [" ".join(words[i:i+max_length])
              for i in range(0, len(words), max_length)]

    results = []
    for chunk in chunks:
        res = sentiment_pipeline(chunk)
        results.append(res[0]) 

    if len(results) > 1:
        labels = [r["label"] for r in results]
        most_common = Counter(labels).most_common(1)[0][0]

        avg_score = sum(r["score"] for r in results if r["label"] == most_common) / \
            sum(1 for r in results if r["label"] == most_common)

        return f"{most_common} ({round(avg_score * 100, 2)}%)"
    else:
        return f"{results[0]['label']} ({round(results[0]['score'] * 100, 2)}%)"


sentiment_tool = Tool(
    name="SentimentAnalyzer",
    func=sentiment_tool_func,
    description="Analyze sentiment of Arabic or English text using pretrained models."
)

if __name__ == "__main__":
    print("✨ Example:", sentiment_tool_func("اليوم جميل جدًا"))
    long_text = " ".join(["هذا نص طويل جدًا ومكرر."] * 300)
    print("✨ Long Example:", sentiment_tool_func(long_text))
