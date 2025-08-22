from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

try:
    from models.summarizer import summarize_text
    from models.corrector import correct_text
    from models.sentiment import sentiment_tool_func as analyze_sentiment
    from models.diagnostics import analyze_text
    from models.machine_translation import translate_text
except ImportError as e:
    print(f"Error importing model: {e}")


def is_arabic(text):
    for ch in text:
        if '\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F' or '\u08A0' <= ch <= '\u08FF':
            return True
    return False


@app.route("/")
def home():
    return redirect(url_for("chat_page"))


@app.route("/chat_page", methods=["GET", "POST"])
def chat_page():
    if request.method == "POST":
        try:
            data = request.get_json()
            input_text = data.get("text", "")
            selected_task = data.get("task", "")
            result = None

            if not input_text.strip():
                result = "⚠️ الرجاء إدخال نص صالح للمعالجة"
            elif not is_arabic(input_text):
                result = "⚠️ ندعم فقط اللغة العربية"
            elif not selected_task:
                result = "⚠️ الرجاء اختيار مهمة"
            else:
                if selected_task == "correction":
                    result = correct_text(input_text)
                elif selected_task == "summary":
                    result = summarize_text(input_text)
                elif selected_task == "sentiment":
                    result = analyze_sentiment(input_text)
                elif selected_task == "diagnostics":
                    result = analyze_text(input_text)
                elif selected_task == "translation":
                    result = translate_text(input_text)

            return jsonify({"result": result})

        except Exception as e:
            return jsonify({"result": f"❌ Error: {str(e)}"})

    return render_template("chat.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
