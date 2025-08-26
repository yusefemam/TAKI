import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'F:\Desktop\taki_analysis\data\spam_ham_dataset.csv')
data = data.dropna(subset=['text', 'label'])

data.loc[data['label'] == 'spam', 'label'] = 0
data.loc[data['label'] == 'ham', 'label'] = 1
data['label'] = data['label'].astype(int)

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
pipeline.fit(X, y)


def mail(text):
    prediction = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    confidence = np.max(probabilities)

    label = "Not spam" if prediction == 1 else "Spam"

    return {"label": label}



if __name__ == "__main__":
    test_text = "Congratulations! You've won a free ticket to Bahamas. Click here to claim."
    result = mail(test_text)
    print(f"label: {result['label']}")
