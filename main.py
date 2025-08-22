from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

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
    return jsonify({"message": "TAKI API is running ✅"})


@app.route("/chat_page", methods=["POST"])
def chat_page():
    data = request.json
    input_text = data.get("text", "")
    selected_task = data.get("task", "")
    result = None

    if not input_text.strip():
        result = "الرجاء إدخال نص صالح للمعالجة"
    elif not selected_task:
        result = "الرجاء اختيار مهمة"
    else:
        try:
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
        except Exception as e:
            result = f"Error in {selected_task}: {str(e)}"

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)
