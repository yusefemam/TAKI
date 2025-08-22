from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MT5Tokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.tools import Tool

sum_model_name = "csebuetnlp/mT5_multilingual_XLSum"
sum_tokenizer = MT5Tokenizer.from_pretrained(sum_model_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)


def summarize_text(text):
    if len(text.split()) < 10:
        return "النص قصير جدًا، أدخل فقرة تحتوي على أكثر من 10 كلمات."
    input_text = "summarize: " + text
    inputs = sum_tokenizer(input_text, return_tensors="pt",
                           truncation=True, max_length=512)

    summary_ids = sum_model.generate(
        inputs['input_ids'],
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    return sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# 📌 Tool 1: Summarizer
summarizer_tool = Tool(
    name="ArabicSummarizer",
    func=summarize_text,
    description="Generates a short Arabic summary of the input text."
)

# 📌 Tool 2: Bullet Point Converter


def to_bullet_points(text):
    points = text.split(". ")
    points = [f"- {point.strip()}" for point in points if point.strip()]
    return "\n".join(points)


bullet_tool = Tool(
    name="BulletPointConverter",
    func=to_bullet_points,
    description="Converts a paragraph into bullet points."
)


# 📌 Tool 3: Grammar Corrector

grammar_model_name = "CAMeL-Lab/arabart-qalb15-gec-ged-13"
grammar_tokenizer = AutoTokenizer.from_pretrained(grammar_model_name)
grammar_model = AutoModelForSeq2SeqLM.from_pretrained(grammar_model_name)


def correct_text(text):
    input_ids = grammar_tokenizer(text, return_tensors="pt").input_ids
    output_ids = grammar_model.generate(
        input_ids, max_length=128, num_beams=4, early_stopping=True)
    corrected = grammar_tokenizer.decode(
        output_ids[0], skip_special_tokens=True)
    return corrected


grammar_tool = Tool(
    name="ArabicGrammarCorrector",
    func=correct_text,
    description="Corrects Arabic grammar and spelling mistakes in text."
)

# 📌 LangChain Tools
tools = [
    summarizer_tool,
    bullet_tool,
    grammar_tool
]
# Example usage of the summarizer, bullet point converter, and grammar corrector
# This is just for demonstration purposes; in practice, you would use these tools in a Lang

if __name__ == "__main__":
    article = """
    هذا المقال يتحدث عن أهمية الذكاء الاصطناعي في حياتنا اليومية.
    يساعد الذكاء الاصطناعي في تحسين الإنتاجية وتسهيل المهام.
    ولكنه يثير أيضًا مخاوف تتعلق بالخصوصية وفقدان الوظائف.
    """
    print("🔷 Original Article:")
    print(article)

    print("\n🔷 Summary:")
    print(summarize_text(article))

    print("\n🔷 Bullet Points:")
    print(to_bullet_points(summarize_text(article)))

    print("\n🔷 Grammar Corrected:")
    print(correct_text(article))
