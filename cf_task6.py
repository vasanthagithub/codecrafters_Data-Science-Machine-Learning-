import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
import random

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

text = [" ".join(document) for document, category in documents]
labels = [category for document, category in documents]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# 5. Model evaluation
from sklearn.metrics import accuracy_score, classification_report

y_pred = nb_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

