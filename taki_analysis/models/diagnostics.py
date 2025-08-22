import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')


def analyze_text(text, lang='english'):
    if not text.strip():
        return "الرجاء إدخال نص صالح للتحليل"


    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower()  

    words = word_tokenize(text)

    words = [w for w in words]

    if not words:
        return "لا توجد كلمات في النص بعد إزالة الكلمات الشائعة"

    avg_len = np.mean([len(w) for w in words])

    counter = Counter(words)
    unique_count = len(counter)
    repeated_count = sum(1 for w, c in counter.items() if c > 1)

    return f"كلمات: {len(words)}, متوسط الطول: {round(avg_len, 2)}, كلمات فريدة: {unique_count}, الكلمات المكرره: {repeated_count}"
