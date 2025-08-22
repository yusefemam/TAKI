from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from dotenv import load_dotenv

# تحميل المتغيرات من .env
load_dotenv()

app = Flask(__name__)

# استيراد الموديلات (لو موجودة)
try:
    from models.summarizer import summarize_text
    from models.corrector import correct_text
    from models.sentiment import sentiment_tool_func as analyze_sentiment
    from models.diagnostics import analyze_text
    from models.machine_translation import translate_text
except ImportError as e:
    print(f"Error importing model: {e}")


@app.route("/")
def home():
    # إعادة التوجيه لصفحة الشات
    return redirect(url_for("chat_page"))


@app.route("/chat_page", methods=["GET", "POST"])
def chat_page():
    if request.method == "POST":
        try:
            data = request.get_json()  # استلام JSON من الفرونت
            input_text = data.get("text", "")
            selected_task = data.get("task", "")
            result = None

            if not input_text.strip():
                result = "⚠️ الرجاء إدخال نص صالح للمعالجة"
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

    # لو دخلت بالرابط مباشرة → يفتح الـ HTML
    return render_template("chat.html")


if __name__ == "__main__":
    # خلي السيرفر يشتغل على أي host علشان ينفع ترفعه
    app.run(host="0.0.0.0", port=5000, debug=True)
