# Import necessary libraries
import os
import sys
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
from datetime import datetime

# Download stopwords and initialize stop words set
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# Define file paths: dataset, saved models, vectorizer, and log file
DATA_PATH         = "cyberbullying_tweets.csv"
VECTORIZER_PATH   = "tfidf_vectorizer.pkl"
LOGREG_MODEL_PATH = "logreg_model.pkl"
NB_MODEL_PATH     = "nb_model.pkl"
LOG_FILE          = "logs/cyberbully.log"

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def train_and_save_models():
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"[Error] Файл не найден: {DATA_PATH}")
        sys.exit(1)

    data = data.rename(columns=lambda c: c.strip().lower())
    if 'tweet_text' in data.columns and 'cyberbullying_type' in data.columns:
        data['text'] = data['tweet_text']
        data['label'] = data['cyberbullying_type'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)
    else:
        print("[Error] Ожидаются колонки 'tweet_text' и 'cyberbullying_type'")
        sys.exit(1)

    data['cleaned'] = data['text'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned'], data['label'],
        test_size=0.2, random_state=42, stratify=data['label']
    )

    vect = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.7,
        stop_words=stopwords.words('english')
    )
    X_train_tfidf = vect.fit_transform(X_train)
    X_test_tfidf  = vect.transform(X_test)

    os.makedirs(os.path.dirname(VECTORIZER_PATH) or '.', exist_ok=True)
    joblib.dump(vect, VECTORIZER_PATH)
    print(f"[Model] Vectorizer saved to {VECTORIZER_PATH}")

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_tfidf, y_train)
    joblib.dump(logreg, LOGREG_MODEL_PATH)
    print(f"[Model] LogisticRegression saved to {LOGREG_MODEL_PATH}")

    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    joblib.dump(nb, NB_MODEL_PATH)
    print(f"[Model] MultinomialNB saved to {NB_MODEL_PATH}")

    print("\n[Evaluation]")
    for name, model in [('LogReg', logreg), ('NaiveBayes', nb)]:
        preds = model.predict(X_test_tfidf)
        print(f"--- {name} ---")
        print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
        print(classification_report(y_test, preds, digits=4))

def load_models():
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(LOGREG_MODEL_PATH) and os.path.exists(NB_MODEL_PATH)):
        print("[Setup] Модели не найдены — тренируем заново")
        train_and_save_models()
    vect   = joblib.load(VECTORIZER_PATH)
    logreg = joblib.load(LOGREG_MODEL_PATH)
    nb     = joblib.load(NB_MODEL_PATH)
    return vect, logreg, nb


def predict_and_log(text: str, model_type: str, vect, logreg, nb):
    clean = preprocess_text(text)
    vec   = vect.transform([clean])
    if model_type == 'nb':
        pred = nb.predict(vec)[0]
        prob = nb.predict_proba(vec)[0][1]
    else:
        pred = logreg.predict(vec)[0]
        prob = logreg.predict_proba(vec)[0][1]

    label_str = 'BULLYING' if pred == 1 else 'SAFE'
    print(f"[{label_str}] (вероятность: {prob:.4f})")

    if pred == 1:
        os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | {model_type} | {prob:.4f} | {text}\n")


def console_interface():
    vect, logreg, nb = load_models()
    print("=== CyberBully Console ===")
    print("Выбери модель: [1] LogisticRegression, [2] NaiveBayes")
    choice = input("Введите 1 или 2: ").strip()
    model_type = 'nb' if choice == '2' else 'logreg'

    print("Введите сообщение (или 'exit' для выхода):")
    while True:
        text = input("> ").strip()
        if text.lower() == 'exit':
            print("До встречи!")
            break
        predict_and_log(text, model_type, vect, logreg, nb)

if __name__ == "__main__":
    console_interface()

