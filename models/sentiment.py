import os
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from langchain.tools import Tool

MODEL_PATH = "./my_sentiment_model"
CSV_PATH = r"C:\Users\PCM\Desktop\taki_analysis\data\cr.csv"
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"


def train_and_save_model():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()  
    df['label'] = df['label'].astype(str) 
    labels = df['label'].unique()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df['label_id'] = df['label'].map(label2id)

    dataset = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch['text'],
            truncation=True,
            padding='max_length',
            max_length=256
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.train_test_split(test_size=0.2)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )

    trainer.train()

    
    preds = trainer.predict(dataset['test'])
    y_pred = preds.predictions.argmax(axis=1)
    y_true = preds.label_ids
    print(classification_report(y_true, y_pred, target_names=labels))

    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("✅ Model trained & saved!")


def load_pipeline():
    return pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        return_all_scores=False
    )


if not os.path.exists(MODEL_PATH):
    print("🚀 Training new model...")
    train_and_save_model()
else:
    print("📦 Model already exists — loading...")

sentiment_pipeline = load_pipeline()


def sentiment_tool_func(text):
    result = sentiment_pipeline(text)
    return f"{result[0]['label']} ({round(result[0]['score'] * 100, 2)}%)"


sentiment_tool = Tool(
    name="SentimentAnalyzer",
    func=sentiment_tool_func,
    description="Analyze sentiment of Arabic text using a fine-tuned CAMeLBERT model."
)


if __name__ == "__main__":
    print("✨ Example: ", sentiment_tool_func("اليوم جميل جدًا"))
