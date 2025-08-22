from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect

model_name = "facebook/m2m100_418M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def translate_text(text, target_language="en"):
    if not text.strip():
        return "Please provide valid text for translation."

    try:
        source_language = detect(text)
        print(f"Detected source language: {source_language}")

        tokenizer.src_lang = source_language

        encoded = tokenizer(text, return_tensors="pt")

        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_language],
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

        translated_text = tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True)
        return translated_text

    except Exception as e:
        return f"An error occurred during translation: {str(e)}"
