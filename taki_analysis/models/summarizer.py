from transformers import MT5Tokenizer, AutoModelForSeq2SeqLM

sum_model_name = "csebuetnlp/mT5_multilingual_XLSum"
sum_tokenizer = MT5Tokenizer.from_pretrained(sum_model_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)


def summarize_text(text):
    if not text.strip():
        return "الرجاء إدخال نص صالح للتلخيص"

    if len(text.split()) < 10:
        return "النص قصير جدًا، أدخل فقرة تحتوي على أكثر من 10 كلمات."

    try:
        input_text = "summarize: " + text
        inputs = sum_tokenizer(input_text, return_tensors="pt",
                               truncation=True, max_length=512)

        summary_ids = sum_model.generate(
            inputs['input_ids'],
            max_length=150,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=2.5
        )

        summary = sum_tokenizer.decode(
            summary_ids[0], skip_special_tokens=True)

        unwanted_phrases = [
            "تعرض عليكم أكثر القضايا التي تداولها مستخدمو مواقع التواصل"]
        for phrase in unwanted_phrases:
            summary = summary.replace(phrase, "")

        return summary.strip()

    except Exception as e:
        return f"حدث خطأ أثناء التلخيص: {str(e)}"
