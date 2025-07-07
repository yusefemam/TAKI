import numpy as np

def analyze_text(text):
    words = text.split()
    avg_len = np.mean([len(w) for w in words]) if words else 0
    unique = len(set(words))
    redundancy = round((1 - unique / len(words)) * 100, 2) if words else 0
    return f"كلمات: {len(words)}, متوسط الطول: {round(avg_len, 2)}, كلمات فريدة: {unique}, تكرار: {redundancy}%"
