import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Загрузка необходимых ресурсов NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Загрузка датасета
# Предполагается, что файл 'cyberbullying_tweets.csv' находится в текущей директории
try:
    data = pd.read_csv("cyberbullying_tweets.csv")
    print("Датасет успешно загружен")
    print(f"Размер датасета: {data.shape}")
    print(data.head())
except FileNotFoundError:
    print("Файл не найден. Убедитесь, что 'cyberbullying_tweets.csv' находится в текущей директории.")
    # Можно создать тестовый датасет, если настоящий файл не найден
    data = pd.DataFrame({
        'tweet_text': [
            "You are so ugly and stupid",
            "I had a great day today!",
            "I will destroy you next time we meet",
            "The weather is nice today",
            "You're worthless, nobody likes you"
        ],
        'cyberbullying_type': [
            "other_cyberbullying",
            "not_cyberbullying",
            "other_cyberbullying",
            "not_cyberbullying",
            "other_cyberbullying"
        ]
    })
    print("Создан тестовый датасет для демонстрации")


# Предобработка данных
def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Удаление упоминаний пользователей
    text = re.sub(r'@\w+', '', text)

    # Удаление хэштегов
    text = re.sub(r'#\w+', '', text)

    # Удаление специальных символов
    text = re.sub(r'[^\w\s]', '', text)

    # Удаление чисел
    text = re.sub(r'\d+', '', text)

    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Применение предобработки к текстовым данным
data['cleaned_text'] = data['tweet_text'].apply(preprocess_text)

# Создание бинарного признака: является ли сообщение кибербуллингом
data['is_cyberbullying'] = data['cyberbullying_type'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_text'],
    data['is_cyberbullying'],
    test_size=0.2,
    random_state=42
)

# Создание TF-IDF векторизатора
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.7,
    stop_words=stopwords.words('english')
)

# Преобразование текста в TF-IDF признаки
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 1. Обучение модели Logistic Regression
print("\nОбучение Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_tfidf, y_train)

# Оценка модели Logistic Regression
y_pred_log = log_reg.predict(X_test_tfidf)
print(f"Accuracy (Logistic Regression): {accuracy_score(y_test, y_pred_log):.4f}")
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_log))

# 2. Обучение модели MultinomialNB
print("\nОбучение MultinomialNB...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Оценка модели MultinomialNB
y_pred_nb = nb_model.predict(X_test_tfidf)
print(f"Accuracy (MultinomialNB): {accuracy_score(y_test, y_pred_nb):.4f}")
print("Classification Report (MultinomialNB):")
print(classification_report(y_test, y_pred_nb))

# Сохранение моделей и векторизатора
joblib.dump(log_reg, 'logreg_model.pkl')
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print("\nМодели и векторизатор сохранены.")


# Функция для предсказания, является ли текст кибербуллингом
def predict_cyberbullying(text,model_type='logreg'):
    # Предобработка текста
    processed_text = preprocess_text(text)

    # Загрузка векторизатора
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Преобразование текста в TF-IDF признаки
    text_tfidf = vectorizer.transform([processed_text])

    # Выбор модели
    if model_type.lower() == 'nb':
        model = joblib.load('nb_model.pkl')
    else:
        model = joblib.load('logreg_model.pkl')

    # Предсказание
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0][1]

    return {
        'is_cyberbullying': bool(prediction),
        'probability': probability,
        'text': text
    }

# Примеры использования
test_texts = [
    "You're a wonderful person, thank you for your help!",
    "I hate you so much, you should delete your account",
    "The weather is beautiful today",
    "You're a complete failure and nobody likes you"
]

print("\nПримеры классификации новых сообщений:")
for text in test_texts:
    result = predict_cyberbullying(text)
    cyberbullying_status = "КИБЕРБУЛЛИНГ" if result['is_cyberbullying'] else "НЕ кибербуллинг"
    print(f"Текст: '{text}'")
    print(f"Результат: {cyberbullying_status} (вероятность: {result['probability']:.4f})")
    print("-" * 50)