from flask import Flask, render_template, request, redirect, url_for
import mysql.connector
from models.corrector import correct_text
from models.summarizer import summarize_text
from models.sentiment import analyze_sentiment
from models.diagnostics import analyze_text

app = Flask(__name__)
app.secret_key = 'Yousef_emam1272005_db_atelier@$@5%'

db_config = {
    'host': "localhost",
    'user': "root",
    'password': "",   
    'database': "taki"
}


def get_db_connection():
    conn = mysql.connector.connect(**db_config)
    return conn


@app.route("/")
def home():
    return redirect(url_for('signup_page'))


@app.route("/signup_page", methods=["GET", "POST"])
def signup_page():
    result = None
    if request.method == "POST":
        firstname = request.form.get("firstname")
        lastname = request.form.get("lastname")
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not firstname or not lastname or not username or not email or not password or not confirm_password:
            result = "يرجى ملء جميع الحقول."
        elif password != confirm_password:
            result = "كلمة المرور وتأكيدها غير متطابقين."
        else:
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO users (firstname, lastname, username, email, password) VALUES (%s, %s, %s, %s, %s)",
                (firstname, lastname, username, email, password)
            )
            connection.commit()
            cursor.close()
            connection.close()
            result = "تم التسجيل بنجاح! يمكنك الآن تسجيل الدخول."

    return render_template("signup.html", result=result)


@app.route("/login_page", methods=["GET", "POST"])
def login_page():
    result = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            result = "يرجى إدخال اسم المستخدم وكلمة المرور."
        else:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM users WHERE username = %s AND password = %s",
                (username, password)
            )
            user = cursor.fetchone()
            cursor.close()
            connection.close()

            if user:
                result = "تم تسجيل الدخول بنجاح!"
            else:
                result = "اسم المستخدم أو كلمة المرور غير صحيحة."

    return render_template("login.html", result=result)


@app.route("/chat_page", methods=["GET", "POST"])
def chat_page():
    result = None
    if request.method == "POST":
        input_text = request.form.get("text")
        selected_task = request.form.get("task")

        if not input_text or not selected_task:
            result = "يرجى إدخال نص وتحديد مهمة."
        else:
            if selected_task == "correction":
                result = correct_text(input_text)
            elif selected_task == "summary":
                result = summarize_text(input_text)
            elif selected_task == "sentiment":
                result = analyze_sentiment(input_text)
            elif selected_task == "diagnostics":
                result = analyze_text(input_text)
            else:
                result = "مهمة غير معروفة."

    return render_template("chat.html", result=result)


@app.route("/tasks_page")
def tasks_page():
    tasks = [
        {"name": "correction", "description": "تصحيح النصوص"},
        {"name": "summary", "description": "تلخيص النصوص"},
        {"name": "sentiment", "description": "تحليل المشاعر"},
        {"name": "diagnostics", "description": "تشخيص النصوص"}
    ]
    return render_template("tasks.html", tasks=tasks)


if __name__ == "__main__":
    app.run(debug=True)
