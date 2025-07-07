
app = Flask(__name__, template_folder='templates')


def get_db_connection():
    if not db_config['password']:
        raise Exception(
            "⚠️ لا يوجد كلمة مرور للـ DB! من فضلك ضعها في .env أو متغير بيئة.")
    return mysql.connector.connect(**db_config)


@app.route("/")
def home():
    # أول ما يفتح الموقع يروح على signup
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
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, password)
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
            cursor = connection.cursor()
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


# ⚠️ لازم تعرف أو تستورد هذه الدوال
def correct_text(text):
    return f"[تصحيح] {text}"


def summarize_text(text):
    return f"[تلخيص] {text}"


def analyze_sentiment(text):
    return f"[تحليل مشاعر] {text}"


def analyze_text(text):
    return f"[تشخيص] {text}"


if __name__ == "__main__":
    app.run(debug=True)
