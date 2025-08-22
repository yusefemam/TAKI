import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.tools import Tool
import nltk

nltk.download("punkt")

MODEL_ENGLISH = "hassaanik/grammar-correction-model"
MODEL_ARABIC = "CAMeL-Lab/arabart-qalb15-gec-ged-13"


def load_pipeline(model_name):
    print(f"🌟 Loading model: {model_name}")
    return pipeline(
        "text2text-generation",
        model=AutoModelForSeq2SeqLM.from_pretrained(model_name),
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        device=0 if torch.cuda.is_available() else -1
    )


grammar_pipeline = load_pipeline(MODEL_ARABIC)


def correct_text(text: str) -> str:
    """
    تصلح النصوص العربية (بتقسيمها لجمل قصيرة لو أطول من حد الموديل)
    """
    sentences = nltk.sent_tokenize(text) 
    corrected_sentences = []

    for sent in sentences:
        result = grammar_pipeline(
            sent,
            max_length=128,      
            truncation=True,    
            num_beams=4,
            early_stopping=True
        )
        corrected_sentences.append(result[0]["generated_text"])

    return " ".join(corrected_sentences)


correction_tool = Tool(
    name="ArabicGrammarCorrector",
    func=correct_text,
    description="Corrects Arabic grammar and spelling mistakes in text."
)


if __name__ == "__main__":
    sample = "انا بحب البرمجه جدا و اريد ان اتعلم. هذا نص طويل جدا قد يسبب مشاكل اذا لم نقسمه."
    print("📝 Original:", sample)
    print("✅ Corrected:", correct_text(sample))
