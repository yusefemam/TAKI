from flask import Flask, render_template, request, redirect, url_for
from models.summarizer import summarize_text
from models.corrector import correct_text
from models.sentiment import sentiment_tool_func as analyze_sentiment
from models.diagnostics import analyze_text

app = Flask(__name__)
app.secret_key = 'Yousef_emam1272005_no_db_atelier'


@app.route("/")
def home():
    return redirect(url_for('chat_page'))


@app.route("/chat_page", methods=["GET", "POST"])
def chat_page():
    result = None
    input_text = ""
    selected_task = ""

    if request.method == "POST":
        input_text = request.form.get("text", "")
        selected_task = request.form.get("task", "")

        if not input_text.strip():
            result = "الرجاء إدخال نص صالح للمعالجة"
        elif not selected_task:
            result = "الرجاء اختيار مهمة"
        else:
            if selected_task == "correction":
                result = correct_text(input_text)
            elif selected_task == "summary":
                result = summarize_text(input_text)
            elif selected_task == "sentiment":
                result = analyze_sentiment(input_text)
            elif selected_task == "diagnostics":
                result = analyze_text(input_text)

    return render_template("chat.html",
                           result=result,
                           input_text=input_text,
                           selected_task=selected_task)


# @app.route("/tasks_page")
# def tasks_page():
#     tasks = [
#         {"name": "correction", "description": "تصحيح النصوص"},
#         {"name": "summary", "description": "تلخيص النصوص"},
#         {"name": "sentiment", "description": "تحليل المشاعر"},
#         {"name": "diagnostics", "description": "تشخيص النصوص"}
#     ]
#     return render_template("tasks.html", tasks=tasks)


if __name__ == "__main__":
    app.run(debug=True)
