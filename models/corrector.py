import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline
)
from langchain.tools import Tool


MODEL_PATH = "./my_grammar_corrector"
CSV_PATH = r"C:\Users\PCM\Desktop\taki_analysis\data\arabic_academic_linguistic_errors_1000.csv"
CSV_PATH2 = r"C:\Users\PCM\Desktop\taki_analysis\data\arabic_linguistic_errors_1500.csv"
if os.path.exists(CSV_PATH2):
    CSV_PATH = CSV_PATH2  
PRETRAINED_MODEL = "CAMeL-Lab/arabart-qalb15-gec-ged-13"


def train_and_save_model():
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise ValueError("CSV file is empty or not found.")
    dd = pd.read_csv(CSV_PATH2)
    if not dd.empty:
        df = pd.concat([df, dd], ignore_index=True)
    df = df.dropna()
    df = df.rename(columns={"wrong_text": "input_text", "correct_text": "target_text"})
    dataset = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    def preprocess(batch):
        inputs = tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=128)
        labels = tokenizer(batch["target_text"], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = labels["input_ids"]
        return inputs

    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.train_test_split(test_size=0.2)

    model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_strategy="epoch",
        predict_with_generate=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("✅ Model trained & saved!")


def load_pipeline():
    if os.path.exists(MODEL_PATH):
        model_dir = MODEL_PATH
        print("📦 Using trained model.")
    else:
        model_dir = PRETRAINED_MODEL
        print("🌟 Using pre-trained model.")
    return pipeline(
        "text2text-generation",
        model=model_dir,
        tokenizer=model_dir
    )


if os.path.exists(CSV_PATH) and not os.path.exists(MODEL_PATH):
    print("🚀 Training model on your CSV...")
    train_and_save_model()
else:
    if os.path.exists(MODEL_PATH):
        print("📦 Trained model already exists.")
    else:
        print("⚡ No CSV found, will use pre-trained model.")


grammar_pipeline = load_pipeline()


def correct_text(text):
    result = grammar_pipeline(text, max_length=128, num_beams=4, early_stopping=True)
    return result[0]["generated_text"]


correction_tool = Tool(
    name="ArabicGrammarCorrector",
    func=correct_text,
    description="Corrects Arabic grammar and spelling mistakes in text."
)


if __name__ == "__main__":
    sample = "انا بحب البرمجه جدا و اريد ان اتعلم"
    print("📝 Original:", sample)
    print("✅ Corrected:", correct_text(sample))
